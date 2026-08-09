"""Resolve a cited paper to a verified open-access PDF URL.

Source-tools node for the citation-verification pipeline. Queries every free
scholarly API concurrently, scores each hit against the requested title, and
returns ranked candidates rather than a single guess — a wrong PDF is worse
than no PDF when the downstream verifier will treat it as ground truth.
"""

from __future__ import annotations

import asyncio
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import httpx

CONTACT_EMAIL = "richyboy170@gmail.com"  # Crossref/OpenAlex polite pool + Unpaywall requirement
USER_AGENT = f"citation-verifier/1.0 (mailto:{CONTACT_EMAIL})"

# Below this normalized-title similarity, a hit is treated as a different paper.
TITLE_MATCH_THRESHOLD = 0.82

ATOM = {"a": "http://www.w3.org/2005/Atom"}

# arXiv mints a DataCite DOI per submission, so a PDF served from arxiv.org can be
# identified from its URL alone without another request.
ARXIV_PDF_RE = re.compile(
    r"arxiv\.org/(?:pdf|abs)/(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})", re.I)


def location_doi(pdf_url: str | None, work_doi: str | None) -> str | None:
    """Identity of the copy being downloaded, not of the merged work record.

    Indexes hang one work-level DOI on every location they have merged into a
    record, so a predatory mirror's DOI can end up naming a file that genuinely
    came from arxiv.org. Where the host mints its own identifiers, prefer those.
    """
    match = ARXIV_PDF_RE.search(pdf_url or "")
    return f"10.48550/arXiv.{match.group(1)}" if match else work_doi


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Candidate:
    source: str
    title: str | None = None
    doi: str | None = None            # work-level, as the index reported it
    location_doi: str | None = None   # this copy's own, where the host mints one
    pdf_url: str | None = None
    landing_url: str | None = None
    title_score: float = 0.0
    pdf_verified: bool = False
    # True when this record was retrieved *by DOI lookup*, so it is the record for
    # the DOI that was asked for even if it echoes no DOI back. A hit from a title
    # search is not, and may not stand in for one.
    doi_keyed: bool = False

    @property
    def identity(self) -> str | None:
        """The DOI of the copy on offer: the location's own where one is derivable,
        otherwise the index's work-level DOI."""
        return self.location_doi or self.doi

    @property
    def confidence(self) -> float:
        """Title agreement, discounted when the PDF URL was not confirmed fetchable."""
        return round(self.title_score * (1.0 if self.pdf_verified else 0.7), 3)

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "title": self.title,
            "doi": self.identity,
            "work_doi": self.doi,
            "pdf_url": self.pdf_url,
            "landing_url": self.landing_url,
            "title_score": round(self.title_score, 3),
            "pdf_verified": self.pdf_verified,
            "confidence": self.confidence,
        }


def pick_best(candidates: list[Candidate]) -> Candidate | None:
    """Highest identity wins; a fetchable PDF only breaks ties inside that tier.

    Ranking on `confidence` alone lets the 0.7 unverified discount overrule
    identity — a title-only match with a reachable PDF (1.0) beats an exact DOI
    match whose PDF failed its HEAD check (0.7), and the wrong paper is chosen. A
    PDF we cannot confirm becomes a download that fails loudly; the wrong paper is
    a silent wrong answer, which is the one outcome this tool exists to prevent.
    """
    if not candidates:
        return None
    top = max(c.title_score for c in candidates)
    tier = [c for c in candidates if c.title_score >= top - 1e-9]
    return next((c for c in tier if c.pdf_verified), tier[0])


@dataclass
class Resolution:
    query_title: str | None
    query_doi: str | None
    best: Candidate | None
    candidates: list[Candidate] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def pdf_url(self) -> str | None:
        return self.best.pdf_url if self.best else None

    def as_dict(self) -> dict:
        return {
            "pdf_url": self.pdf_url,
            # The identity of the copy on offer, never the identity that was merely
            # requested — a resolution must not echo back an assumption as a finding.
            "doi": self.best.identity if self.best else None,
            "work_doi": self.best.doi if self.best else None,
            "matched_title": self.best.title if self.best else None,
            "confidence": self.best.confidence if self.best else 0.0,
            "source": self.best.source if self.best else None,
            "alternatives": [c.as_dict() for c in self.candidates[1:4]],
            "errors": self.errors,
        }


# --------------------------------------------------------------------------- #
# Title matching
# --------------------------------------------------------------------------- #

def normalize_title(title: str) -> str:
    """Fold accents, drop punctuation and collapse whitespace so that
    'Attention Is All You Need!' and 'attention is all you need' compare equal."""
    folded = unicodedata.normalize("NFKD", title)
    ascii_only = "".join(c for c in folded if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", ascii_only.lower())).strip()


def title_similarity(requested: str, found: str | None) -> float:
    if not found:
        return 0.0
    a, b = normalize_title(requested), normalize_title(found)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # Indexes truncate titles and drop subtitles, so a *shorter* found title sitting
    # inside the requested one usually means "same paper" — but only when it is most
    # of it, or "Deep learning" would match "Deep learning for image recognition".
    #
    # The other direction earns nothing. A candidate that ADDS words is normally a
    # different work: "Tensor Product Attention Is All You Need" contains "Attention
    # Is All You Need" whole and is a different paper by different authors.
    if b in a and len(b) / len(a) >= 0.6:
        return 0.95
    ratio = SequenceMatcher(None, a, b).ratio()
    # Prepended words change the claim, however much of the string survives:
    # "Not All Attention Is All You Need" reaches 0.86 on raw ratio alone and argues
    # the opposite of the paper it would be standing in for. Hold it under the bar.
    if a in b and not b.startswith(a):
        return min(ratio, TITLE_MATCH_THRESHOLD - 0.01)
    return ratio


def identity_score(candidate: Candidate, title: str | None, doi: str | None) -> float:
    """How strongly a hit proves it is the paper that was asked for."""
    if not doi:
        return title_similarity(title, candidate.title) if title else 0.0

    if candidate.doi and candidate.doi.strip().lower() == doi.strip().lower():
        return 1.0  # exact identity; nothing left to check

    if not candidate.doi_keyed:
        # A title-search hit scored against a DOI-pinned query. Its title may match,
        # but nothing here ties it to the DOI that was asked for — and a generic
        # title carries no weight at all: arXiv 1807.07987 is titled "Deep Learning"
        # and is not 10.1038/nature14539.
        return 0.0

    if not candidate.doi:
        # We asked a DOI-keyed endpoint for this DOI and it answered without echoing
        # one back. The record is still the record for that DOI.
        return 0.9

    # The DOIs disagree on a record we fetched *by* DOI — normally a preprint and its
    # published version, so the title still gets to vouch for the match.
    return title_similarity(title, candidate.title) if title else 0.0


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #

class RateLimiter:
    """Minimum-interval gate. NCBI caps unkeyed clients at 3 req/s and arXiv
    asks for one request every 3s; exceeding either gets the IP throttled."""

    def __init__(self, min_interval: float):
        self._min_interval = min_interval
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def __aenter__(self):
        async with self._lock:
            wait = self._min_interval - (time.monotonic() - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()

    async def __aexit__(self, *exc):
        return False


# --------------------------------------------------------------------------- #
# Resolver
# --------------------------------------------------------------------------- #

class PaperResolver:
    def __init__(self, *, email: str = CONTACT_EMAIL, timeout: float = 15.0,
                 verify_pdfs: bool = True, max_retries: int = 3):
        self.email = email
        self.verify_pdfs = verify_pdfs
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
        self._limiters = {
            "eutils.ncbi.nlm.nih.gov": RateLimiter(0.34),
            "www.ncbi.nlm.nih.gov": RateLimiter(0.34),
            "export.arxiv.org": RateLimiter(3.0),
            "api.semanticscholar.org": RateLimiter(1.0),
        }

    async def __aenter__(self) -> "PaperResolver":
        return self

    async def __aexit__(self, *exc):
        await self._client.aclose()

    async def _get(self, url: str, **params) -> httpx.Response:
        """GET with per-host pacing and backoff on throttling, transport failure, or a
        transient server error."""
        limiter = self._limiters.get(httpx.URL(url).host)
        response: httpx.Response | None = None
        transport_error: httpx.TransportError | None = None

        for attempt in range(self.max_retries):
            try:
                if limiter:
                    async with limiter:
                        response = await self._client.get(url, params=params or None)
                else:
                    response = await self._client.get(url, params=params or None)
            except httpx.TransportError as exc:
                # A throttling API often stalls or drops the connection instead of
                # answering 429. Retrying only on status codes lets one hiccup delete
                # a whole source from the fan-out, and its papers are never found.
                response, transport_error = None, exc
                await asyncio.sleep(2 ** attempt)
                continue

            transport_error = None
            if response.status_code not in (429, 500, 502, 503, 504):
                response.raise_for_status()
                return response

            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt
            await asyncio.sleep(delay)

        if response is not None:
            response.raise_for_status()
            return response
        raise transport_error or httpx.HTTPError(f"no response from {url}")

    # -- individual sources -------------------------------------------------- #

    async def _openalex(self, title: str | None, doi: str | None) -> list[Candidate]:
        if doi:
            url, params = f"https://api.openalex.org/works/doi:{doi}", {"mailto": self.email}
        else:
            url = "https://api.openalex.org/works"
            params = {"search": title, "per_page": 3, "mailto": self.email}
        payload = (await self._get(url, **params)).json()
        works = payload.get("results", [payload]) if "results" in payload else [payload]

        out = []
        for w in works:
            work_title = w.get("display_name") or w.get("title")
            work_doi = (w.get("doi") or "").replace("https://doi.org/", "") or None

            # OpenAlex picks best_oa_location by its own criteria, which routinely
            # prefers a predatory mirror over the canonical arXiv copy sitting in the
            # same payload. Emit every location it knows and let the allowlist choose
            # the host — these links cost no extra request.
            seen: set[str] = set()
            for loc in (w.get("best_oa_location"), w.get("primary_location"),
                        *(w.get("locations") or [])):
                pdf = (loc or {}).get("pdf_url")
                if not pdf or pdf in seen:
                    continue
                seen.add(pdf)
                out.append(Candidate(
                    source="openalex",
                    title=work_title,
                    doi=work_doi,
                    location_doi=location_doi(pdf, work_doi),
                    pdf_url=pdf,
                    landing_url=loc.get("landing_page_url"),
                    doi_keyed=bool(doi),
                ))

            if not seen:
                oa_url = (w.get("open_access") or {}).get("oa_url")
                if oa_url:
                    out.append(Candidate(
                        source="openalex",
                        title=work_title,
                        doi=work_doi,
                        location_doi=location_doi(oa_url, work_doi),
                        pdf_url=oa_url,
                        landing_url=(w.get("best_oa_location") or {}).get("landing_page_url"),
                        doi_keyed=bool(doi),
                    ))
        return out

    async def _semantic_scholar(self, title: str | None, doi: str | None) -> list[Candidate]:
        fields = "title,openAccessPdf,externalIds"
        if doi:
            url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
            papers = [(await self._get(url, fields=fields)).json()]
        else:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            papers = (await self._get(url, query=title, fields=fields, limit=3)).json().get("data", [])

        out = []
        for p in papers:
            if not p:
                continue
            external = p.get("externalIds") or {}
            pdf = (p.get("openAccessPdf") or {}).get("url")
            if not pdf and external.get("ArXiv"):
                # S2 frequently knows the arXiv id without filling in openAccessPdf.
                # ("ArXiv" arrives inside externalIds — it is not a requestable field,
                # and naming it in `fields` makes the whole call 400.)
                pdf = f"https://arxiv.org/pdf/{external['ArXiv']}"
            out.append(Candidate(
                source="semantic_scholar",
                title=p.get("title"),
                doi=external.get("DOI"),
                location_doi=location_doi(pdf, external.get("DOI")),
                pdf_url=pdf,
                doi_keyed=bool(doi),
            ))
        return out

    async def _unpaywall(self, title: str | None, doi: str | None) -> list[Candidate]:
        if not doi:
            return []
        data = (await self._get(f"https://api.unpaywall.org/v2/{doi}", email=self.email)).json()
        loc = data.get("best_oa_location") or {}
        return [Candidate(
            source="unpaywall",
            title=data.get("title"),
            doi=doi,
            location_doi=location_doi(loc.get("url_for_pdf"), doi),
            pdf_url=loc.get("url_for_pdf"),
            landing_url=loc.get("url_for_landing_page"),
            doi_keyed=True,  # this endpoint is DOI-keyed and takes nothing else
        )]

    async def _crossref(self, title: str | None, doi: str | None) -> list[Candidate]:
        if doi:
            items = [(await self._get(f"https://api.crossref.org/works/{doi}",
                                      mailto=self.email)).json()["message"]]
        else:
            resp = await self._get("https://api.crossref.org/works",
                                   **{"query.bibliographic": title, "rows": 3, "mailto": self.email})
            items = resp.json()["message"]["items"]

        out = []
        for item in items:
            pdf = next((l["URL"] for l in item.get("link", [])
                        if l.get("content-type") == "application/pdf"), None)
            out.append(Candidate(
                source="crossref",
                title=(item.get("title") or [None])[0],
                doi=item.get("DOI"),
                location_doi=location_doi(pdf, item.get("DOI")),
                pdf_url=pdf,  # frequently publisher-gated; PDF verification will catch it
                landing_url=item.get("URL"),
                doi_keyed=bool(doi),
            ))
        return out

    async def _arxiv(self, title: str | None, doi: str | None) -> list[Candidate]:
        if not title:
            return []
        # Quotes in a Lucene field query break arXiv's parser, so strip them.
        safe = normalize_title(title)
        resp = await self._get("https://export.arxiv.org/api/query",
                               search_query=f'ti:"{safe}"', max_results=3)
        out = []
        for entry in ET.fromstring(resp.text).findall("a:entry", ATOM):
            pdf = next((l.get("href") for l in entry.findall("a:link", ATOM)
                        if l.get("title") == "pdf"), None)
            work_doi = entry.findtext("a:doi", namespaces=ATOM)
            out.append(Candidate(
                source="arxiv",
                title=(entry.findtext("a:title", default="", namespaces=ATOM) or "").strip(),
                doi=work_doi,
                location_doi=location_doi(pdf, work_doi),
                pdf_url=pdf,
                landing_url=entry.findtext("a:id", namespaces=ATOM),
                # This source only ever runs a *title* search, so its hits can never
                # stand in for a DOI-pinned query. doi_keyed stays False.
            ))
        return out

    async def _pmc_oa_pdf(self, uid: str) -> str | None:
        """Open-access PDF link for one PMC record, or None if it is not in the OA subset."""
        oa = await self._get("https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi", id=f"PMC{uid}")
        link = ET.fromstring(oa.text).find(".//link[@format='pdf']")
        href = link.get("href") if link is not None else None
        # The OA service returns ftp:// URLs; rewrite to https so httpx can fetch them.
        if href and href.startswith("ftp://"):
            href = href.replace("ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov", 1)
        return href

    async def _pubmed(self, title: str | None, doi: str | None) -> list[Candidate]:
        # An unqualified db=pmc term is a *full-text* search — "Attention Is All
        # You Need" matches 738k articles that merely use those words. A DOI is
        # distinctive enough to search bare; a title has to be pinned to the title
        # field as a quoted phrase or the query returns pure noise.
        if doi:
            term = doi
        elif title:
            phrase = normalize_title(title)
            if not phrase:
                return []
            term = f'"{phrase}"[title]'
        else:
            return []

        search = await self._get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                                 db="pmc", term=term, retmode="json", retmax=3, sort="relevance")
        ids = search.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        # Even a pinned search returns articles that only *cite* the query — a DOI
        # lookup for a paper PMC does not hold comes back with the papers citing
        # it. So each hit has to prove its identity from the record's own
        # metadata; echoing the query back as the candidate's title would score it
        # 1.0 against itself and defeat the title guard entirely.
        summary = await self._get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                                  db="pmc", id=",".join(ids), retmode="json")
        result = summary.json().get("result", {})

        out = []
        for uid in result.get("uids", []):
            record = result.get(uid) or {}
            record_title = record.get("title")
            record_doi = next((a.get("value") for a in record.get("articleids", [])
                               if a.get("idtype") == "doi"), None)

            # Same policy as identity_score(), applied here so a rejected hit never
            # costs a rate-limited OA round trip.
            if doi and record_doi:
                if record_doi.strip().lower() != doi.strip().lower():
                    continue  # a different DOI is a different paper
            elif title:
                if title_similarity(title, record_title) < TITLE_MATCH_THRESHOLD:
                    continue
            else:
                # DOI query, and PMC publishes no DOI for this record — with no
                # title to fall back on, nothing here proves identity.
                continue

            href = await self._pmc_oa_pdf(uid)
            if not href:
                continue
            out.append(Candidate(
                source="pubmed",
                title=record_title,
                doi=record_doi,  # the record's own, already identity-checked above
                pdf_url=href,
                landing_url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{uid}/",
                # A DOI term here is a *full-text* search, not a record lookup, so
                # these hits get no DOI-keyed trust — the filter above is what vouches
                # for them, and it only passes a hit whose own DOI matched.
            ))
        return out

    # -- verification and orchestration -------------------------------------- #

    async def _verify_pdf(self, url: str) -> bool:
        """Confirm the URL actually serves a PDF. Publishers routinely return a
        200 HTML paywall from a .pdf path, which would otherwise poison the pipeline."""
        try:
            head = await self._client.head(url)
            if head.status_code >= 400 or "pdf" not in head.headers.get("content-type", "").lower():
                # Some servers reject HEAD; fall back to sniffing the magic bytes.
                async with self._client.stream("GET", url) as stream:
                    if stream.status_code >= 400:
                        return False
                    if "pdf" in stream.headers.get("content-type", "").lower():
                        return True
                    async for chunk in stream.aiter_bytes(5):
                        return chunk.startswith(b"%PDF-")
                return False
            return True
        except httpx.HTTPError:
            return False

    async def resolve(self, title: str | None = None, doi: str | None = None) -> Resolution:
        if not (title or doi):
            raise ValueError("resolve() requires a title, a doi, or both")

        sources = {
            "openalex": self._openalex,
            "semantic_scholar": self._semantic_scholar,
            "unpaywall": self._unpaywall,
            "crossref": self._crossref,
            "arxiv": self._arxiv,
            "pubmed": self._pubmed,
        }
        results = await asyncio.gather(
            *(fn(title, doi) for fn in sources.values()), return_exceptions=True
        )

        candidates: list[Candidate] = []
        errors: dict[str, str] = {}
        for name, result in zip(sources, results):
            if isinstance(result, Exception):
                errors[name] = f"{type(result).__name__}: {result}"
                continue
            for c in result:
                if not c.pdf_url:
                    continue
                score = identity_score(c, title, doi)
                if score >= TITLE_MATCH_THRESHOLD:
                    candidates.append(Candidate(**{**c.__dict__, "title_score": score}))

        if self.verify_pdfs and candidates:
            verified = await asyncio.gather(*(self._verify_pdf(c.pdf_url) for c in candidates))
            candidates = [Candidate(**{**c.__dict__, "pdf_verified": ok})
                          for c, ok in zip(candidates, verified)]

        # Identity first, fetchability second — see pick_best().
        candidates.sort(key=lambda c: (c.title_score, c.pdf_verified), reverse=True)
        best = pick_best(candidates)
        if best is not None:
            # as_dict() publishes candidates[1:4] as `alternatives`, so the winner has
            # to sit at index 0 or it is listed twice and a runner-up is hidden.
            candidates = [best] + [c for c in candidates if c is not best]

        return Resolution(query_title=title, query_doi=doi, best=best,
                          candidates=candidates, errors=errors)


# --------------------------------------------------------------------------- #
# Agent tool surface
# --------------------------------------------------------------------------- #

FIND_PAPER_PDF_TOOL = {
    "name": "find_paper_pdf",
    "description": (
        "Resolve a cited research paper to a verified open-access PDF URL by querying "
        "OpenAlex, Semantic Scholar, Unpaywall, Crossref, arXiv and PubMed Central. "
        "Returns the best match with a confidence score plus alternative candidates. "
        "A null pdf_url means no open-access copy was found, not that the paper does not exist."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Paper title exactly as it appears in the citation"},
            "doi": {"type": "string", "description": "DOI if the citation supplies one; enables exact-identity resolution"},
        },
        "anyOf": [{"required": ["title"]}, {"required": ["doi"]}],
    },
}


async def find_paper_pdf(title: str | None = None, doi: str | None = None) -> dict:
    """Tool entrypoint. Opens a client per call — for batch verification, hold a
    single PaperResolver open and call .resolve() directly to reuse connections."""
    async with PaperResolver() as resolver:
        return (await resolver.resolve(title=title, doi=doi)).as_dict()


if __name__ == "__main__":
    import json

    async def main():
        async with PaperResolver() as resolver:
            for kwargs in (
                {"title": "Attention Is All You Need"},
                {"doi": "10.1038/nature14539"},
                {"title": "A paper title that does not exist anywhere at all"},
            ):
                result = await resolver.resolve(**kwargs)
                print(json.dumps(result.as_dict(), indent=2))

    asyncio.run(main())
