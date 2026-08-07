"""Find open-access papers from a description of what you want, not a citation.

The semantic counterpart to source-tool. Where that tool resolves a paper you can
already name — title, DOI, or both — this one takes free text like "transformer
architectures applied to protein structure prediction" and returns the top-k
nearest open-access papers by embedding similarity.

    description
      -> expand into several queries (local LLM, optional)
      -> fan out to OpenAlex / Semantic Scholar / Europe PMC / arXiv / Crossref
      -> dedup by DOI, then by normalized title
      -> embed and rank by cosine similarity against the original description
      -> drop everything below the similarity floor, keep the top k

WHAT THIS TOOL DOES NOT GUARANTEE
---------------------------------
source-tool's safety property is identity: a hit must prove it is *the paper you
asked for*, via a matching DOI or a title similarity above 0.82. That guard works
by comparing against a requested title, and in semantic search there is no
requested title — so it is structurally unavailable here.

What replaces it is a similarity floor plus provenance. A result is the nearest
match among what retrieval happened to return; it is NOT verified to be any
particular paper, and an empty result means nothing cleared the floor, not that
no such research exists. Every record carries its similarity, its abstract, and
which sources and query variants surfaced it, so a human or a downstream verifier
can judge relevance instead of trusting the rank.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import httpx

from embedder import DEFAULT_MODEL, Embedder, cosine
from query_expander import expand as expand_queries

CONTACT_EMAIL = "richyboy170@gmail.com"  # Crossref/OpenAlex polite pool
USER_AGENT = f"citation-verifier-semantic/1.0 (mailto:{CONTACT_EMAIL})"

# Calibrated 2026-08-07 against DEFAULT_MODEL — see the README's "Calibrating the
# floor". Measured: correct answers land 0.47-0.83, pool medians 0.24-0.29.
# 0.35 sits above the tail and below the weakest correct answer observed.
#
# This is a TAIL FILTER, not an identity guard. Measurement showed no absolute
# threshold separates a correct answer to a loosely-worded query (0.468) from the
# top hit of a deliberately nonsensical one (0.529) — cosine is not comparable
# across queries. It removes obvious noise; it cannot certify a top result.
# Changing --model invalidates this number.
MIN_SIMILARITY = 0.35

# A candidate with no abstract is scored on its title alone. Dropping those would
# silently lose real papers (arXiv and Crossref hits often arrive bare), but an
# unpenalized bare title can outrank a genuine abstract match on short queries.
TITLE_ONLY_PENALTY = 0.85

DEFAULT_TOP_K = 5
DEFAULT_PER_SOURCE = 20

# How far down the ranking to probe for fetchable PDFs before filling the k
# slots. Probing widens in batches of FACTOR x k and stops as soon as k
# fetchable papers are found, so an easy query pays one batch. The cap bounds
# the worst case: a query where nothing below the floor is fetchable would
# otherwise HEAD the entire pool, which runs to hundreds.
VERIFY_WINDOW_FACTOR = 3
VERIFY_WINDOW_CAP = 45
# Much shorter than the client's request timeout — see _verify_pdf().
VERIFY_TIMEOUT = 8.0

# Longest Retry-After we will actually wait out. Past this the source is treated
# as unavailable for this run rather than slept on — see _get().
RETRY_AFTER_CAP = 30.0

# Simultaneous PDF probes allowed against any one host. Selection probes a
# widening window, and a single query's candidates cluster heavily on a handful
# of hosts (11 of 12 results in one measured batch were arxiv.org), so without a
# bound one run can open dozens of connections to the same server at once. A
# small cap keeps bursts civil without serialising the window into minutes.
PROBE_CONCURRENCY = 3

ATOM = {"a": "http://www.w3.org/2005/Atom"}


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #

def normalize_title(title: str) -> str:
    """Fold accents, drop punctuation, collapse whitespace — used for dedup keys."""
    folded = unicodedata.normalize("NFKD", title)
    ascii_only = "".join(c for c in folded if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", ascii_only.lower())).strip()


def title_similarity(a: str | None, b: str | None) -> float:
    """Only used to merge near-identical titles during dedup, never to rank."""
    if not a or not b:
        return 0.0
    na, nb = normalize_title(a), normalize_title(b)
    if not na or not nb:
        return 0.0
    return 1.0 if na == nb else SequenceMatcher(None, na, nb).ratio()


def reconstruct_abstract(inverted: dict | None) -> str | None:
    """OpenAlex ships abstracts as {word: [positions]} and no plain text field."""
    if not isinstance(inverted, dict) or not inverted:
        return None
    placed = [(pos, word) for word, positions in inverted.items()
              for pos in (positions or []) if isinstance(pos, int)]
    return " ".join(word for _, word in sorted(placed)) or None


def strip_markup(text: str | None) -> str | None:
    """Crossref abstracts arrive as JATS XML; the tags would pollute the embedding."""
    if not text:
        return None
    cleaned = re.sub(r"<[^>]+>", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


def clean_query(text: str) -> str:
    """Quotes and Lucene operators break arXiv's parser and skew the others."""
    return re.sub(r"\s+", " ", re.sub(r'["\'()\[\]:+^~]', " ", text)).strip()


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Candidate:
    """One hit from one source for one query, before dedup."""
    source: str
    query_index: int
    title: str | None = None
    abstract: str | None = None
    doi: str | None = None
    pdf_url: str | None = None
    landing_url: str | None = None


@dataclass
class Paper:
    """A deduplicated work, merged from every candidate that pointed at it."""
    key: str
    title: str | None = None
    abstract: str | None = None
    doi: str | None = None
    pdf_urls: list[str] = field(default_factory=list)
    landing_url: str | None = None
    retrieval_sources: list[str] = field(default_factory=list)
    matched_queries: list[int] = field(default_factory=list)
    similarity: float = 0.0
    title_only: bool = False
    pdf_verified: bool = False
    rank: int = 0
    # Where this paper sat under pure similarity ordering. Under the default
    # PDF-preferring selection the kept set is not the top k by score, so without
    # this the manifest cannot explain why rank 1 is rank 1.
    similarity_rank: int = 0

    @property
    def pdf_url(self) -> str | None:
        return self.pdf_urls[0] if self.pdf_urls else None

    @property
    def confidence(self) -> float:
        """Similarity discounted when the PDF URL was not confirmed fetchable.

        Same shape and meaning as source-tool's Candidate.confidence, so the two
        manifests stay comparable — but the underlying score is a cosine, not a
        title match. `similarity` is always the raw, undiscounted number.
        """
        return round(self.similarity * (1.0 if self.pdf_verified else 0.7), 3)

    def as_dict(self) -> dict:
        return {
            "rank": self.rank,
            "similarity_rank": self.similarity_rank,
            "similarity": round(self.similarity, 4),
            "confidence": self.confidence,
            "title": self.title,
            "doi": self.doi,
            "abstract": self.abstract,
            "title_only": self.title_only,
            "pdf_url": self.pdf_url,
            "pdf_urls": self.pdf_urls,
            "landing_url": self.landing_url,
            "pdf_verified": self.pdf_verified,
            "retrieval_sources": self.retrieval_sources,
            "matched_queries": self.matched_queries,
        }


@dataclass
class Resolution:
    query_description: str
    expanded_queries: list[str] = field(default_factory=list)
    expansion_error: str | None = None
    embedding_model: str = DEFAULT_MODEL
    min_similarity: float = MIN_SIMILARITY
    papers: list[Paper] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    truncated_sources: list[str] = field(default_factory=list)
    pool_size: int = 0
    scored: list[float] = field(default_factory=list)  # every score, for calibration
    selection: str = "pdf_preferring"  # or "rank_only"
    verified_window: int = 0  # candidates whose PDF URL was probed before selecting

    def as_dict(self) -> dict:
        return {
            "query_description": self.query_description,
            "expanded_queries": self.expanded_queries,
            "expansion_error": self.expansion_error,
            "embedding_model": self.embedding_model,
            "min_similarity": self.min_similarity,
            "selection": self.selection,
            "verified_window": self.verified_window,
            "pool_size": self.pool_size,
            "results": [p.as_dict() for p in self.papers],
            "truncated_sources": self.truncated_sources,
            "errors": self.errors,
        }


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #

class RateLimiter:
    """Minimum-interval gate. These are the APIs' own published limits — raising
    them gets the IP throttled."""

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

class SemanticResolver:
    def __init__(
        self,
        *,
        email: str = CONTACT_EMAIL,
        timeout: float = 30.0,
        max_retries: int = 3,
        model: str = DEFAULT_MODEL,
        per_source: int = DEFAULT_PER_SOURCE,
        verify_pdfs: bool = True,
        expand: bool = True,
        rank_only: bool = False,
        url_filter: Callable[[str], bool] | None = None,
        ollama_model: str | None = None,
        ollama_url: str | None = None,
        variants: int = 4,
    ):
        self.email = email
        self.max_retries = max_retries
        self.per_source = per_source
        self.verify_pdfs = verify_pdfs
        self.expand = expand
        self.rank_only = rank_only
        # Lets the caller say which URLs it would actually be willing to fetch —
        # fetch_semantic passes its allowlist check. Without it, selection
        # happily spends its slots on papers that verify fine and are then
        # reported BLOCK, which is the same wasted-slot bug PDF-preferring
        # selection exists to fix (seen with a one-host allowlist: both results
        # verified, both blocked). Kept as a predicate rather than a host set so
        # the resolver stays agnostic about how the runner decides.
        self.url_filter = url_filter
        self.variants = variants
        self.embedder = Embedder(model)
        self._expander_kwargs = {
            k: v for k, v in (("model", ollama_model), ("url", ollama_url)) if v
        }
        self._client = httpx.AsyncClient(
            timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT}
        )
        self._limiters = {
            "export.arxiv.org": RateLimiter(3.0),
            "api.semanticscholar.org": RateLimiter(1.0),
            "www.ebi.ac.uk": RateLimiter(0.34),
            # PDF hosts, reached by _verify_pdf rather than _get. NCBI publishes
            # 3 requests/second for unauthenticated clients and Europe PMC mirrors
            # it; selection probes enough URLs at once to exceed that easily.
            "www.ncbi.nlm.nih.gov": RateLimiter(0.34),
            "europepmc.org": RateLimiter(0.34),
        }
        # Per-host concurrency caps for PDF probing, created on demand. Separate
        # from _limiters because most PDF hosts publish no rate at all — the goal
        # there is only to avoid opening a burst of connections at once.
        self._probe_gates: dict[str, asyncio.Semaphore] = {}
        # Optional and never required — every other source works without it. A key
        # moves S2 off the shared unauthenticated pool, which is the only way that
        # source answers reliably (see the 429 note in `_get`).
        self._s2_key = os.environ.get("S2_API_KEY") or None
        self._auth_headers = (
            {"api.semanticscholar.org": {"x-api-key": self._s2_key}}
            if self._s2_key else {}
        )

    async def __aenter__(self) -> "SemanticResolver":
        return self

    async def __aexit__(self, *exc):
        await self._client.aclose()

    async def _get(self, url: str, **params) -> httpx.Response:
        """GET with per-host pacing and backoff on throttling, transport failure,
        or a transient server error.

        Scholarly APIs frequently throttle by stalling the connection rather than
        answering 429, so transport errors retry too — without that, one hiccup
        silently deletes a whole source from the fan-out.
        """
        host = httpx.URL(url).host
        limiter = self._limiters.get(host)
        headers = self._auth_headers.get(host)
        response: httpx.Response | None = None
        transport_error: httpx.TransportError | None = None

        for attempt in range(self.max_retries):
            try:
                if limiter:
                    async with limiter:
                        response = await self._client.get(
                            url, params=params or None, headers=headers)
                else:
                    response = await self._client.get(
                        url, params=params or None, headers=headers)
            except httpx.TransportError as exc:
                response, transport_error = None, exc
                await asyncio.sleep(2 ** attempt)
                continue

            transport_error = None
            if response.status_code not in (429, 500, 502, 503, 504):
                response.raise_for_status()
                return response

            # Semantic Scholar's *unauthenticated* search pool is shared globally
            # and returns 429 to an idle `curl` (verified 2026-08-07, three probes,
            # no load). Backing off 1s/2s/4s cannot outwait a quota that is already
            # exhausted before we arrive — it only adds ~7s of dead time per query
            # to a source that will not answer. Fail fast, let the error land in
            # `errors`, and keep the other four sources' pacing untouched.
            #
            # With S2_API_KEY set we are on our own quota, so a 429 means *we*
            # exceeded it — that is exactly the case backoff is designed for, and
            # the fast path is skipped.
            if (response.status_code == 429
                    and host == "api.semanticscholar.org"
                    and not self._s2_key):
                response.raise_for_status()

            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt
            if delay > RETRY_AFTER_CAP:
                # Honouring Retry-After literally hands an API the power to stop
                # the run: OpenAlex answers an exhausted quota with
                # `Retry-After: 36494` (~10 hours, verified 2026-08-07) and the
                # bare sleep obeyed it, hanging the whole tool with no output.
                # A wait this long means the source cannot help *this* run, so
                # drop it into `errors` and let the other four finish — the same
                # thing we already do for a source that errors outright.
                response.raise_for_status()
            await asyncio.sleep(delay)

        if response is not None:
            response.raise_for_status()
            return response
        raise transport_error or httpx.HTTPError(f"no response from {url}")

    # -- sources ------------------------------------------------------------- #

    async def _openalex(self, query: str, qi: int) -> list[Candidate]:
        payload = (await self._get(
            "https://api.openalex.org/works",
            search=query, per_page=self.per_source, mailto=self.email,
        )).json()

        out = []
        for work in payload.get("results", []):
            title = work.get("display_name") or work.get("title")
            doi = (work.get("doi") or "").replace("https://doi.org/", "") or None
            abstract = reconstruct_abstract(work.get("abstract_inverted_index"))

            # Emit every OA location, as source-tool does: OpenAlex's own
            # best_oa_location routinely ranks a predatory mirror above the
            # canonical copy sitting in the same payload. More hosts means more
            # chances one is on the allowlist, and each still faces every guard.
            seen: set[str] = set()
            for loc in (work.get("best_oa_location"), work.get("primary_location"),
                        *(work.get("locations") or [])):
                pdf = (loc or {}).get("pdf_url")
                if not pdf or pdf in seen:
                    continue
                seen.add(pdf)
                out.append(Candidate("openalex", qi, title, abstract, doi, pdf,
                                     loc.get("landing_page_url")))
            if not seen:
                out.append(Candidate("openalex", qi, title, abstract, doi, None,
                                     (work.get("best_oa_location") or {}).get("landing_page_url")))
        return out

    async def _europepmc(self, query: str, qi: int) -> list[Candidate]:
        payload = (await self._get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            query=query, format="json", pageSize=self.per_source, resultType="core",
        )).json()

        out = []
        for rec in payload.get("resultList", {}).get("result", []):
            # Only documentStyle == "pdf" entries are actual PDFs. The rest are
            # landing pages that would fail the %PDF- check and report a
            # misleading FAIL — better to contribute the abstract with no
            # pdf_url and let the record report MISS honestly.
            urls = (rec.get("fullTextUrlList") or {}).get("fullTextUrl") or []
            pdfs = [u.get("url") for u in urls
                    if u.get("documentStyle") == "pdf" and u.get("url")]
            landing = next((u.get("url") for u in urls if u.get("url")), None)

            common = dict(
                title=rec.get("title"),
                abstract=rec.get("abstractText"),
                doi=rec.get("doi"),
                landing_url=landing,
            )
            if pdfs:
                out.extend(Candidate("europepmc", qi, pdf_url=p, **common) for p in pdfs)
            else:
                out.append(Candidate("europepmc", qi, **common))
        return out

    async def _semantic_scholar(self, query: str, qi: int) -> list[Candidate]:
        # "ArXiv" is NOT a requestable field name — naming it here returns
        # 400 Unrecognized fields and kills the source. The id arrives inside
        # externalIds regardless.
        payload = (await self._get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            query=query,
            fields="title,abstract,externalIds,openAccessPdf",
            limit=min(self.per_source, 100),
        )).json()

        out = []
        for paper in payload.get("data") or []:
            if not paper:
                continue
            external = paper.get("externalIds") or {}
            pdf = (paper.get("openAccessPdf") or {}).get("url")
            landing = None
            # S2 routinely puts a doi.org link in openAccessPdf. That is a
            # redirector to a landing page, not a PDF — treating it as one makes
            # every such hit report BLOCK against the host "doi.org", which is
            # noise: doi.org is not a site that serves papers.
            if pdf and (httpx.URL(pdf).host or "").endswith("doi.org"):
                pdf, landing = None, pdf
            if not pdf and external.get("ArXiv"):
                # S2 frequently knows the arXiv id without filling openAccessPdf.
                pdf = f"https://arxiv.org/pdf/{external['ArXiv']}"
            out.append(Candidate("semantic_scholar", qi, paper.get("title"),
                                 paper.get("abstract"), external.get("DOI"), pdf, landing))
        return out

    async def _arxiv(self, query: str, qi: int) -> list[Candidate]:
        safe = clean_query(query)
        if not safe:
            return []
        # `all:` searches title, abstract and full text — this is topic search.
        # source-tool uses `ti:` because it is looking up a known title.
        response = await self._get(
            "https://export.arxiv.org/api/query",
            search_query=f"all:{safe}", max_results=self.per_source,
        )
        out = []
        for entry in ET.fromstring(response.text).findall("a:entry", ATOM):
            pdf = next((l.get("href") for l in entry.findall("a:link", ATOM)
                        if l.get("title") == "pdf"), None)
            out.append(Candidate(
                "arxiv", qi,
                (entry.findtext("a:title", default="", namespaces=ATOM) or "").strip(),
                (entry.findtext("a:summary", default="", namespaces=ATOM) or "").strip() or None,
                entry.findtext("a:doi", namespaces=ATOM),
                pdf,
                entry.findtext("a:id", namespaces=ATOM),
            ))
        return out

    async def _crossref(self, query: str, qi: int) -> list[Candidate]:
        payload = (await self._get(
            "https://api.crossref.org/works",
            **{"query.bibliographic": query, "rows": self.per_source, "mailto": self.email},
        )).json()

        out = []
        for item in payload.get("message", {}).get("items", []):
            pdf = next((l["URL"] for l in item.get("link", [])
                        if l.get("content-type") == "application/pdf"), None)
            out.append(Candidate(
                "crossref", qi,
                (item.get("title") or [None])[0],
                strip_markup(item.get("abstract")),  # JATS XML when present at all
                item.get("DOI"),
                pdf,  # usually publisher-gated; PDF validation will catch it
                item.get("URL"),
            ))
        return out

    # -- dedup, rank, verify ------------------------------------------------- #

    def _dedup(self, candidates: list[Candidate]) -> list[Paper]:
        """Merge candidates into works, BEFORE anything expensive runs.

        Five overlapping indexes return the same paper repeatedly; without this
        the top-k comes back as five copies of one work, and every duplicate
        costs an embedding.
        """
        papers: dict[str, Paper] = {}
        by_title: dict[str, str] = {}  # normalized title -> key, for DOI-less merges

        for cand in candidates:
            norm = normalize_title(cand.title or "")
            if not norm:
                continue  # nothing to embed and nothing to dedup on
            doi = (cand.doi or "").strip().lower() or None

            # Title takes precedence over DOI. A preprint and its published
            # version carry *different* DOIs for the same work, so keying on DOI
            # first leaves both in the pool and lets one work occupy two of the
            # k slots. Someone asking "find me papers about X" wants the work,
            # not each of its versions.
            key = by_title.get(norm) or doi or f"title:{norm}"

            paper = papers.get(key)
            if paper is None:
                paper = papers[key] = Paper(key=key, title=cand.title, doi=doi)
            by_title.setdefault(norm, key)

            # Keep the longest abstract seen — sources truncate differently.
            if cand.abstract and len(cand.abstract) > len(paper.abstract or ""):
                paper.abstract = cand.abstract
            paper.doi = paper.doi or doi
            paper.landing_url = paper.landing_url or cand.landing_url
            if cand.pdf_url and cand.pdf_url not in paper.pdf_urls:
                paper.pdf_urls.append(cand.pdf_url)
            if cand.source not in paper.retrieval_sources:
                paper.retrieval_sources.append(cand.source)
            if cand.query_index not in paper.matched_queries:
                paper.matched_queries.append(cand.query_index)

        for paper in papers.values():
            paper.matched_queries.sort()
        return list(papers.values())

    def _rank(self, description: str, papers: list[Paper]) -> list[Paper]:
        """Score every paper against the ORIGINAL description.

        The expanded variants are retrieval aids — they widen the pool. The
        description is what the user actually meant, so it alone defines
        relevance. Multi-source agreement is recorded as provenance but
        deliberately does not boost the score: rank stays purely semantic.
        """
        if not papers:
            return []
        texts = [f"{p.title}\n{p.abstract}" if p.abstract else (p.title or "")
                 for p in papers]
        vectors = self.embedder.encode([description, *texts])
        query_vec, paper_vecs = vectors[0], vectors[1:]

        for paper, vec in zip(papers, paper_vecs):
            score = cosine(query_vec, vec)
            paper.title_only = not paper.abstract
            paper.similarity = score * (TITLE_ONLY_PENALTY if paper.title_only else 1.0)

        papers.sort(key=lambda p: p.similarity, reverse=True)
        return papers

    async def _verify_pdf(self, url: str) -> bool:
        """Confirm the URL really serves a PDF — publishers routinely return a
        200 HTML paywall from a .pdf path.

        Uses a short timeout rather than the client's: this now runs on a
        widening window of candidates, several of them per query, so one host
        that accepts a connection and then stalls would otherwise hold up the
        whole selection for the full 60 s (measured: a batch run exceeded ten
        minutes before this cap). A server too slow to answer a HEAD is a poor
        download bet anyway, and a false negative only costs that paper its
        preference, never its place in the results.

        Paced per host. These are PDF hosts, not the search APIs, so the
        `_limiters` table `_get` uses mostly does not cover them — and a window
        of candidates from one query lands on very few hosts. Unbounded, a single
        run could open dozens of simultaneous connections to arxiv.org, which is
        how a tool earns an IP block.
        """
        host = httpx.URL(url).host
        limiter = self._limiters.get(host)
        gate = self._probe_gates.get(host)
        if gate is None:
            gate = self._probe_gates[host] = asyncio.Semaphore(PROBE_CONCURRENCY)
        try:
            async with gate:
                if limiter:
                    async with limiter:
                        head = await self._client.head(url, timeout=VERIFY_TIMEOUT)
                else:
                    head = await self._client.head(url, timeout=VERIFY_TIMEOUT)
                if (head.status_code < 400
                        and "pdf" in head.headers.get("content-type", "").lower()):
                    return True
                async with self._client.stream(  # some reject HEAD
                    "GET", url, timeout=VERIFY_TIMEOUT
                ) as stream:
                    if stream.status_code >= 400:
                        return False
                    if "pdf" in stream.headers.get("content-type", "").lower():
                        return True
                    async for chunk in stream.aiter_bytes(5):
                        return chunk.startswith(b"%PDF-")
            return False
        except httpx.HTTPError:
            return False

    # -- orchestration ------------------------------------------------------- #

    async def resolve(
        self,
        description: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        min_similarity: float | None = None,
    ) -> Resolution:
        description = (description or "").strip()
        if not description:
            raise ValueError("resolve() requires a non-empty description")
        floor = MIN_SIMILARITY if min_similarity is None else min_similarity

        queries, expansion_error = ([description], None)
        if self.expand:
            queries, expansion_error = await expand_queries(
                description, variants=self.variants, client=self._client,
                **self._expander_kwargs,
            )

        resolution = Resolution(
            query_description=description,
            expanded_queries=queries,
            expansion_error=expansion_error,
            embedding_model=self.embedder.model_name,
            min_similarity=floor,
        )

        sources = {
            "openalex": self._openalex,
            "semantic_scholar": self._semantic_scholar,
            "europepmc": self._europepmc,
            "arxiv": self._arxiv,
            "crossref": self._crossref,
        }
        jobs = [(name, fn, qi, q)
                for name, fn in sources.items()
                for qi, q in enumerate(queries)]
        results = await asyncio.gather(
            *(fn(q, qi) for _, fn, qi, q in jobs), return_exceptions=True
        )

        candidates: list[Candidate] = []
        truncated: set[str] = set()
        for (name, _, _, _), result in zip(jobs, results):
            if isinstance(result, Exception):
                # A source that fails is recorded and the run continues on the
                # others. A run with errors is not a failed run.
                resolution.errors.setdefault(name, f"{type(result).__name__}: {result}")
                continue
            if len(result) >= self.per_source:
                truncated.add(name)  # never cap silently — report it
            candidates.extend(result)

        resolution.truncated_sources = sorted(truncated)

        papers = self._rank(description, self._dedup(candidates))
        resolution.pool_size = len(papers)
        resolution.scored = [round(p.similarity, 4) for p in papers]

        above = [p for p in papers if p.similarity >= floor]
        for i, paper in enumerate(above, 1):
            paper.similarity_rank = i

        kept = await self._select(above, top_k, resolution)
        for i, paper in enumerate(kept, 1):
            paper.rank = i
        resolution.papers = kept
        return resolution

    async def _select(
        self, above: list[Paper], top_k: int, resolution: Resolution
    ) -> list[Paper]:
        """Choose which `top_k` of the above-floor candidates to return.

        Similarity alone is the wrong criterion for a tool whose output is a
        directory of PDFs. Measured on "best time for playing instruments"
        (2026-08-07): 22 candidates cleared the floor and 12 had a PDF, but only
        2 of the top 5 did — Crossref, which contributes DOIs rather than open
        access (limitation #5), held ranks 1/3/4 and crowded out fetchable arXiv
        and Europe PMC hits at ranks 10-22. The run downloaded nothing.

        So membership prefers papers we can actually fetch, while *ordering*
        stays by similarity — a reader expects the printed scores to descend.
        `similarity_rank` records the pure-similarity position of every kept
        paper, so the reordering is visible rather than implied.

        Unfetchable papers still backfill any slots left over: a `MISS` line
        saying a relevant paper exists behind a paywall is useful information,
        it just should not consume the whole result set.
        """
        if self.rank_only:
            resolution.selection = "rank_only"
            kept = above[:top_k]
            if self.verify_pdfs and kept:
                await self._mark_verified(kept)
                # Report the probing that actually happened — the manifest field
                # is provenance, and 0 here would misdescribe the run.
                resolution.verified_window = len(kept)
            return kept

        resolution.selection = "pdf_preferring"

        if not self.verify_pdfs:
            # --dry-run / verify_pdfs=False must issue no HEAD requests, so fall
            # back to the weaker proxy: the index claims an acceptable URL exists.
            fetchable = [p for p in above if self._fetch_target(p)]
        else:
            # Verify in widening batches rather than one fixed window. A fixed 3x
            # window is enough when the head of the ranking is fetchable, but on
            # a query where a PDF-less source owns the top it is not: measured
            # 2026-08-07 on "best time for playing instruments", the top 15 held
            # only 3 fetchable papers, so PDF-less hits still backfilled. Widening
            # costs nothing on the easy case (one batch and stop) and is what
            # rescues the hard one.
            #
            # These are HEAD requests, concurrent within a batch — the same call
            # the old code made after selecting, moved earlier so its answer can
            # inform the choice. It also drops a publisher that 403s or 200s a
            # paywall (Wiley did exactly that) before it costs a slot.
            batch = max(top_k, VERIFY_WINDOW_FACTOR * top_k)
            probed = 0
            while probed < min(len(above), VERIFY_WINDOW_CAP):
                nxt = min(probed + batch, len(above), VERIFY_WINDOW_CAP)
                await self._mark_verified(above[probed:nxt])
                probed = nxt
                if sum(1 for p in above[:probed] if p.pdf_verified) >= top_k:
                    break
            resolution.verified_window = probed
            fetchable = [p for p in above[:probed] if p.pdf_verified]

        chosen = fetchable[:top_k]
        if len(chosen) < top_k:
            picked = {id(p) for p in chosen}
            chosen += [p for p in above if id(p) not in picked][:top_k - len(chosen)]

        chosen.sort(key=lambda p: p.similarity, reverse=True)
        return chosen

    def _fetch_target(self, paper: Paper) -> str | None:
        """The URL the runner would actually try first for this paper.

        With a url_filter set this is the first *acceptable* URL rather than the
        top-ranked one, so verification answers the question that matters — "can
        we get this paper?" — instead of "does its first URL happen to serve?"
        """
        if self.url_filter is None:
            return paper.pdf_url
        return next((u for u in paper.pdf_urls if self.url_filter(u)), None)

    async def _mark_verified(self, papers: list[Paper]) -> None:
        """Probe each paper's fetch target concurrently and record the result."""
        targets = [self._fetch_target(p) for p in papers]
        results = await asyncio.gather(
            *(self._verify_pdf(u) if u else _false() for u in targets)
        )
        for paper, ok in zip(papers, results):
            paper.pdf_verified = ok


async def _false() -> bool:
    return False


# --------------------------------------------------------------------------- #
# Agent tool surface
# --------------------------------------------------------------------------- #

FIND_PAPERS_BY_DESCRIPTION_TOOL = {
    "name": "find_papers_by_description",
    "description": (
        "Find open-access research papers semantically similar to a free-text description "
        "of a topic, for when no title or DOI is available. Queries OpenAlex, Semantic "
        "Scholar, Europe PMC, arXiv and Crossref, then ranks candidates by embedding "
        "similarity and returns up to top_k with their scores and abstracts. "
        "Results are the nearest matches among what retrieval returned — they are NOT "
        "verified to be any specific paper, so read the abstract before relying on one. "
        "An empty list means nothing cleared the similarity floor, NOT that no such "
        "research exists. If you have a title or DOI, use find_paper_pdf instead: it "
        "verifies identity, which this tool cannot."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "What the papers should be about, in plain language",
            },
            "top_k": {"type": "integer", "description": "Max results (default 5)"},
            "min_similarity": {
                "type": "number",
                "description": "Cosine floor; below it results are dropped. Defaults to "
                               "the tool's calibrated value.",
            },
        },
        "required": ["description"],
    },
}


async def find_papers_by_description(
    description: str,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float | None = None,
) -> dict:
    """Tool entrypoint. Opens a client per call — for batches, hold one
    SemanticResolver open and call .resolve() directly to reuse connections and
    the loaded embedding model."""
    async with SemanticResolver() as resolver:
        resolution = await resolver.resolve(
            description, top_k=top_k, min_similarity=min_similarity
        )
        return resolution.as_dict()


if __name__ == "__main__":
    import json

    async def main():
        async with SemanticResolver() as resolver:
            for description in (
                "attention mechanisms for neural machine translation",
                "deep learning methods for predicting protein structure",
                "quantum tunneling effects in medieval manuscript bookbinding adhesives",
            ):
                result = await resolver.resolve(description, top_k=3)
                print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
                print(f"  pool={result.pool_size}  top scores={result.scored[:8]}\n")

    asyncio.run(main())
