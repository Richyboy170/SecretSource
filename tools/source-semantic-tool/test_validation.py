"""Offline validation for the parts test_selection.py does not reach.

    py test_validation.py

`test_selection.py` covers top-k selection, Retry-After backoff and probe
pacing. This file covers what limitation #7 in the README names as untested:
**retrieval parsing, dedup, the download guard chain, and the allowlist** — plus
the manifest contract and the query expander's failure paths.

No network and no torch. Source responses are fixtures, `download()` runs against
an `httpx.MockTransport`, and every ranking assertion here is arithmetic over the
constants rather than a real embedding — so this suite is deterministic and safe
to run against throttled APIs. Claims that need live APIs or the embedding model
are recorded in VALIDATION.md instead, with the command that reproduces them.

Each test names the documented behaviour it guards, so a failure says which
README claim stopped being true.
"""

from __future__ import annotations

import asyncio
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

import query_expander
from embedder import Embedder, cosine
from fetch_semantic import (
    MIN_PDF_BYTES,
    download,
    is_allowed,
    load_sources,
    run_metadata,
    slugify,
    source_tool_resolution,
)
from semantic_resolver import (
    MIN_SIMILARITY,
    TITLE_ONLY_PENALTY,
    Candidate,
    Paper,
    RateLimiter,
    Resolution,
    SemanticResolver,
    clean_query,
    normalize_title,
    reconstruct_abstract,
    strip_markup,
    title_similarity,
)


def check(name: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}"
          f"{'' if condition else '  <- ' + detail}")
    return bool(condition)


def bare_resolver(**attrs) -> SemanticResolver:
    """A resolver with no client, no limiter and no API key.

    Same `__new__` trick test_selection.py and bench_ranking.py use: the source
    parsers and `_dedup` touch nothing but `self.per_source`, so constructing the
    real object (which opens an HTTP client) would only add a resource to close.
    """
    r = SemanticResolver.__new__(SemanticResolver)
    r.per_source = 20
    r.url_filter = None
    r.email = "validation@example.test"
    for k, v in attrs.items():
        setattr(r, k, v)
    return r


def stub_get(payload: dict | None = None, text: str = ""):
    """Replace `_get` with one that answers from a fixture."""
    class Fixture:
        def json(self):
            return payload
        @property
        def text(self):
            return text

    async def _get(url, **params):
        return Fixture()
    return _get


# --------------------------------------------------------------------------- #
# Allowlist — the guard that decides what reaches disk
# --------------------------------------------------------------------------- #

def allowlist_tests() -> list[bool]:
    """README: "Suffix-anchored match so a listed domain covers its subdomains
    without also matching lookalikes."

    This is the tool's only host-level security boundary, and a regression is
    silent — a lookalike host downloads successfully and looks like any other OK.
    """
    print("\nallowlist parsing (load_sources)")
    results = []
    with TemporaryDirectory() as tmp:
        src = Path(tmp) / "sources.txt"
        src.write_text(
            "# a whole-line comment\n"
            "\n"
            "https://arxiv.org/\n"
            "www.nature.com\n"
            "europepmc.org\n"
            "https://www.aclweb.org/          # an inline comment\n",
            encoding="utf-8")
        hosts = load_sources(src)

    results.append(check("full URL, bare domain and www- form normalize alike",
                         hosts == {"arxiv.org", "nature.com", "europepmc.org",
                                   "aclweb.org"},
                         f"got {sorted(hosts)}"))
    results.append(check("inline # comment is stripped from the host",
                         "aclweb.org" in hosts and not any("#" in h for h in hosts),
                         f"got {sorted(hosts)}"))

    print("\nallowlist matching (is_allowed)")
    hosts = {"arxiv.org", "europepmc.org"}
    cases = [
        ("https://arxiv.org/pdf/1706.03762", True, "exact host"),
        ("https://export.arxiv.org/pdf/1706.03762", True, "subdomain is covered"),
        ("https://www.arxiv.org/pdf/x", True, "www- prefix is stripped before matching"),
        ("https://arxiv.org.example.com/x.pdf", False,
         "LOOKALIKE: listed domain as a subdomain of an attacker host"),
        ("https://notarxiv.org/x.pdf", False, "suffix without a dot boundary"),
        ("https://sci-hub.se/x.pdf", False, "unlisted host"),
        ("https://EUROPEPMC.ORG/x.pdf", True, "host match is case-insensitive"),
    ]
    for url, expected, why in cases:
        results.append(check(f"{why}: {url[:52]}",
                             is_allowed(url, hosts) is expected,
                             f"expected {expected}"))
    return results


# --------------------------------------------------------------------------- #
# Download guard chain — README "Where each guard sits", rows 7-9
# --------------------------------------------------------------------------- #

async def download_tests() -> list[bool]:
    """Every guard in `download()`: post-redirect allowlist re-check, %PDF- magic
    bytes, and the MIN_PDF_BYTES floor. Plus the invariant that a failed attempt
    leaves neither a .part nor a partial .pdf behind — otherwise the next run
    reports HAVE on a truncated file.
    """
    print("\ndownload guard chain")
    results = []
    hosts = {"good.test"}
    real_pdf = b"%PDF-1.7\n" + b"x" * (MIN_PDF_BYTES * 2)

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/ok.pdf":
            return httpx.Response(200, content=real_pdf)
        if path == "/paywall.pdf":  # 200 OK, but it is a login page
            return httpx.Response(200, content=b"<html><body>Sign in</body></html>")
        if path == "/tiny.pdf":
            return httpx.Response(200, content=b"%PDF-1.4\nshort")
        if path == "/redirect.pdf":
            return httpx.Response(302, headers={"Location": "https://evil.test/x.pdf"})
        if path == "/x.pdf":  # the off-allowlist landing spot, serving a real PDF
            return httpx.Response(200, content=real_pdf)
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler),
                               follow_redirects=True)
    async with client:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)

            dest = out / "ok.pdf"
            await download(client, "https://good.test/ok.pdf", dest, hosts)
            results.append(check("a real PDF on an allowed host is committed",
                                 dest.exists() and dest.read_bytes().startswith(b"%PDF-")))
            results.append(check("no .part survives a success",
                                 not dest.with_suffix(".part").exists()))

            async def expect_failure(name: str, url: str, fragment: str):
                d = out / "bad.pdf"
                try:
                    await download(client, url, d, hosts)
                except Exception as exc:
                    ok = fragment in str(exc)
                    results.append(check(name, ok, f"raised {type(exc).__name__}: {exc}"))
                else:
                    results.append(check(name, False, "no exception raised"))
                results.append(check(f"  ...and writes nothing to disk",
                                     not d.exists() and not d.with_suffix(".part").exists()))

            await expect_failure("HTML paywall served from a .pdf URL is rejected",
                                 "https://good.test/paywall.pdf", "not a PDF")
            await expect_failure(f"a PDF under {MIN_PDF_BYTES} bytes is rejected",
                                 "https://good.test/tiny.pdf", "too small")
            await expect_failure("a redirect off the allowlist is rejected",
                                 "https://good.test/redirect.pdf", "off-allowlist")

            # The redirect guard must key on where we LANDED, not where we asked.
            # Without hosts there is nothing to re-check, so the same URL succeeds
            # — which is what proves the previous failure came from the guard.
            dest = out / "any.pdf"
            await download(client, "https://good.test/redirect.pdf", dest, None)
            results.append(check("the same redirect succeeds with no allowlist "
                                 "(proves the guard, not the transport, blocked it)",
                                 dest.exists()))
    return results


# --------------------------------------------------------------------------- #
# Source parsers — README "Data flow", five-source fan-out
# --------------------------------------------------------------------------- #

async def parser_tests() -> list[bool]:
    """Each source's response shape, from a fixture.

    These are the quiet failures: a field renamed upstream makes a source return
    zero candidates, the fan-out silently narrows, and nothing errors.
    """
    print("\nOpenAlex parsing")
    results = []
    r = bare_resolver()
    r._get = stub_get({"results": [
        {
            "display_name": "Attention Is All You Need",
            "doi": "https://doi.org/10.5555/3295222",
            # OpenAlex ships abstracts only as {word: [positions]}
            "abstract_inverted_index": {"The": [0], "dominant": [1], "models": [2]},
            "best_oa_location": {"pdf_url": "https://arxiv.org/pdf/1706.03762",
                                 "landing_page_url": "https://arxiv.org/abs/1706.03762"},
            "primary_location": {"pdf_url": "https://mirror.test/a.pdf"},
            "locations": [{"pdf_url": "https://arxiv.org/pdf/1706.03762"}],  # duplicate
        },
        {"display_name": "No Location Paper", "abstract_inverted_index": None},
    ]})
    cands = await r._openalex("q", 0)
    results.append(check("reconstructs the inverted-index abstract in word order",
                         cands[0].abstract == "The dominant models",
                         f"got {cands[0].abstract!r}"))
    results.append(check("strips the doi.org prefix to a bare DOI",
                         cands[0].doi == "10.5555/3295222", f"got {cands[0].doi}"))
    results.append(check("emits every distinct OA location, deduping repeats",
                         [c.pdf_url for c in cands[:2]] ==
                         ["https://arxiv.org/pdf/1706.03762", "https://mirror.test/a.pdf"],
                         f"got {[c.pdf_url for c in cands]}"))
    results.append(check("a work with no PDF still contributes its title/abstract",
                         cands[-1].title == "No Location Paper" and cands[-1].pdf_url is None,
                         f"got {cands[-1]}"))

    print("\nEurope PMC parsing")
    r._get = stub_get({"resultList": {"result": [
        {
            "title": "Dyad practice facilitates motor learning",
            "abstractText": "We tested pairs of learners.",
            "doi": "10.1000/dyad",
            "fullTextUrlList": {"fullTextUrl": [
                {"documentStyle": "html", "url": "https://europepmc.org/article/MED/1"},
                {"documentStyle": "pdf", "url": "https://europepmc.org/api/x.pdf"},
            ]},
        },
        {  # landing pages only — must contribute no pdf_url at all
            "title": "HTML Only Paper", "abstractText": "Abstract here.",
            "fullTextUrlList": {"fullTextUrl": [
                {"documentStyle": "html", "url": "https://europepmc.org/article/MED/2"}]},
        },
    ]}})
    cands = await r._europepmc("q", 0)
    results.append(check('only documentStyle == "pdf" becomes a pdf_url',
                         cands[0].pdf_url == "https://europepmc.org/api/x.pdf",
                         f"got {cands[0].pdf_url}"))
    results.append(check("the first URL of any style becomes the landing page",
                         cands[0].landing_url == "https://europepmc.org/article/MED/1",
                         f"got {cands[0].landing_url}"))
    results.append(check("an HTML-only record yields a candidate with no pdf_url "
                         "(honest MISS, not a misleading FAIL)",
                         cands[1].pdf_url is None and cands[1].abstract == "Abstract here.",
                         f"got {cands[1]}"))

    print("\nSemantic Scholar parsing")
    r._get = stub_get({"data": [
        {"title": "DOI Redirector", "abstract": "a",
         "openAccessPdf": {"url": "https://doi.org/10.1/x"}, "externalIds": {}},
        {"title": "ArXiv Fallback", "abstract": "b",
         "openAccessPdf": None, "externalIds": {"ArXiv": "2101.00001"}},
        {"title": "Real PDF", "abstract": "c",
         "openAccessPdf": {"url": "https://cdn.test/p.pdf"},
         "externalIds": {"DOI": "10.2/y"}},
        None,  # S2 returns nulls in `data`; a naive loop would crash here
    ]})
    cands = await r._semantic_scholar("q", 0)
    results.append(check("a doi.org link is demoted to landing_url, not treated "
                         "as a PDF",
                         cands[0].pdf_url is None
                         and cands[0].landing_url == "https://doi.org/10.1/x",
                         f"got pdf={cands[0].pdf_url} landing={cands[0].landing_url}"))
    results.append(check("an arXiv id fills in for a missing openAccessPdf",
                         cands[1].pdf_url == "https://arxiv.org/pdf/2101.00001",
                         f"got {cands[1].pdf_url}"))
    results.append(check("a real openAccessPdf URL passes through with its DOI",
                         cands[2].pdf_url == "https://cdn.test/p.pdf"
                         and cands[2].doi == "10.2/y"))
    results.append(check("a null entry in `data` is skipped, not crashed on",
                         len(cands) == 3, f"got {len(cands)}"))

    print("\nCrossref parsing")
    r._get = stub_get({"message": {"items": [
        {"title": ["A JATS Abstract"],
         "abstract": "<jats:p>Real <jats:italic>text</jats:italic> here.</jats:p>",
         "DOI": "10.3/z", "URL": "https://doi.org/10.3/z",
         "link": [{"content-type": "text/html", "URL": "https://pub.test/html"},
                  {"content-type": "application/pdf", "URL": "https://pub.test/p.pdf"}]},
        {"title": ["No Abstract"], "DOI": "10.4/w"},
    ]}})
    cands = await r._crossref("q", 0)
    results.append(check("JATS markup is stripped before it pollutes the embedding",
                         cands[0].abstract == "Real text here.",
                         f"got {cands[0].abstract!r}"))
    results.append(check("only the application/pdf link is taken as a pdf_url",
                         cands[0].pdf_url == "https://pub.test/p.pdf",
                         f"got {cands[0].pdf_url}"))
    results.append(check("an abstract-less item still yields a candidate",
                         cands[1].title == "No Abstract" and cands[1].abstract is None))

    print("\narXiv parsing")
    atom = """<feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/1706.03762v5</id>
        <title>  Attention Is All You Need  </title>
        <summary>  The dominant sequence transduction models.  </summary>
        <link href="http://arxiv.org/abs/1706.03762v5" rel="alternate"/>
        <link title="pdf" href="http://arxiv.org/pdf/1706.03762v5" rel="related"/>
      </entry>
    </feed>"""
    r._get = stub_get(text=atom)
    cands = await r._arxiv("attention", 0)
    results.append(check("title and summary are parsed and whitespace-trimmed",
                         cands[0].title == "Attention Is All You Need"
                         and cands[0].abstract == "The dominant sequence transduction models.",
                         f"got {cands[0].title!r} / {cands[0].abstract!r}"))
    results.append(check('the pdf link is chosen by title="pdf", not by order',
                         cands[0].pdf_url == "http://arxiv.org/pdf/1706.03762v5",
                         f"got {cands[0].pdf_url}"))
    results.append(check("a query that cleans to nothing makes no request",
                         await r._arxiv('"()[]:+^~', 0) == []))
    return results


# --------------------------------------------------------------------------- #
# Dedup — README "Where each guard sits", row 1
# --------------------------------------------------------------------------- #

def dedup_tests() -> list[bool]:
    """README: "Title takes precedence over DOI. A preprint and its published
    version carry *different* DOIs for the same work."

    Keying on DOI first leaves both in the pool and lets one work occupy two of
    the k slots — a failure that looks like a normal result set.
    """
    print("\ndedup")
    results = []
    r = bare_resolver()

    papers = r._dedup([
        Candidate("arxiv", 0, "Attention Is All You Need", "short abstract",
                  "10.48550/arxiv.1706.03762", "https://arxiv.org/pdf/1706.03762"),
        Candidate("openalex", 1, "Attention is all you need!", "a much longer abstract "
                  "with substantially more detail in it", "10.5555/3295222",
                  "https://papers.nips.cc/p.pdf"),
    ])
    results.append(check("preprint and published version merge into ONE work "
                         "despite different DOIs",
                         len(papers) == 1, f"got {len(papers)} papers"))
    p = papers[0]
    results.append(check("the longest abstract seen wins",
                         p.abstract.startswith("a much longer"), f"got {p.abstract!r}"))
    results.append(check("every source that found it is recorded",
                         p.retrieval_sources == ["arxiv", "openalex"],
                         f"got {p.retrieval_sources}"))
    results.append(check("every query variant that surfaced it is recorded, sorted",
                         p.matched_queries == [0, 1], f"got {p.matched_queries}"))
    results.append(check("both hosts are kept, so a failed download can fall through",
                         len(p.pdf_urls) == 2, f"got {p.pdf_urls}"))

    papers = r._dedup([
        Candidate("crossref", 0, None, "no title", "10.1/x"),
        Candidate("crossref", 0, "   ", "whitespace title", "10.2/y"),
    ])
    results.append(check("a title-less candidate is dropped (nothing to embed)",
                         papers == [], f"got {len(papers)}"))

    papers = r._dedup([
        Candidate("openalex", 0, "Deep Learning", "x", None, "https://a.test/1.pdf"),
        Candidate("arxiv", 0, "Deep Learning", "y", None, "https://a.test/1.pdf"),
    ])
    results.append(check("DOI-less candidates still merge on normalized title",
                         len(papers) == 1 and papers[0].pdf_urls == ["https://a.test/1.pdf"],
                         f"got {len(papers)} / {papers[0].pdf_urls}"))

    papers = r._dedup([
        Candidate("openalex", 0, "Distinct Paper One", "x"),
        Candidate("openalex", 0, "Distinct Paper Two", "y"),
    ])
    results.append(check("genuinely different titles are NOT merged",
                         len(papers) == 2, f"got {len(papers)}"))
    return results


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #

def text_tests() -> list[bool]:
    print("\ntext normalization")
    results = []
    # Accents fold to their base letter; every other non-alphanumeric becomes a
    # space (so "Bü-chner" -> "bu chner", not "buchner"). Both halves matter for
    # dedup: accent folding merges the same title from two indexes, and the
    # space rule is what makes punctuation differences invisible.
    folded = normalize_title("Bü-chner's  Éxample: A Study!")
    results.append(check("normalize_title folds accents and flattens punctuation",
                         folded == "bu chner s example a study", f"got {folded!r}"))
    results.append(check("titles differing only by case/punctuation are identical",
                         title_similarity("Attention Is All You Need",
                                          "attention is all you need.") == 1.0))
    results.append(check("unrelated titles score low",
                         title_similarity("Attention Is All You Need",
                                          "Molecular Property Prediction") < 0.5))
    results.append(check("a missing title is 0.0, never an exception",
                         title_similarity(None, "x") == 0.0))
    results.append(check("clean_query strips Lucene operators that break arXiv",
                         clean_query('deep "learning" (AI) [x] a:b +c ^d ~e')
                         == "deep learning AI x a b c d e",
                         f"got {clean_query(chr(34)+'x'+chr(34))!r}"))
    results.append(check("strip_markup removes JATS tags and collapses whitespace",
                         strip_markup("<jats:p>a   <b>b</b></jats:p>") == "a b"))
    results.append(check("strip_markup of an empty/None value is None",
                         strip_markup(None) is None and strip_markup("<p></p>") is None))
    results.append(check("reconstruct_abstract of a non-dict is None",
                         reconstruct_abstract(None) is None
                         and reconstruct_abstract({}) is None))
    results.append(check("reconstruct_abstract orders by position, not dict order",
                         reconstruct_abstract({"world": [1], "hello": [0]}) == "hello world"))

    print("\ncosine")
    results.append(check("identical vectors -> 1.0",
                         abs(cosine([1.0, 2.0], [1.0, 2.0]) - 1.0) < 1e-9))
    results.append(check("orthogonal vectors -> 0.0",
                         abs(cosine([1.0, 0.0], [0.0, 1.0])) < 1e-9))
    results.append(check("an empty or zero vector is 0.0, never a ZeroDivisionError",
                         cosine([], [1.0]) == 0.0 and cosine([0.0, 0.0], [1.0, 1.0]) == 0.0))

    print("\njoin_paper_text (model-specific pairing)")
    # A non-SPECTER name resolves its separator without loading any weights, so
    # this stays offline. The SPECTER [SEP] path needs the model — see
    # VALIDATION.md, "Model-backed checks".
    e = Embedder("sentence-transformers/all-MiniLM-L6-v2")
    results.append(check("non-SPECTER models join with a newline",
                         e.join_paper_text("T", "A") == "T\nA",
                         f"got {e.join_paper_text('T', 'A')!r}"))
    results.append(check("an abstract-less paper is its bare title (no dangling sep)",
                         e.join_paper_text("T", None) == "T"
                         and e.join_paper_text("T", "   ") == "T"))
    results.append(check("a title-less paper is its bare abstract",
                         e.join_paper_text(None, "A") == "A"))
    return results


# --------------------------------------------------------------------------- #
# Scoring arithmetic — README "Calibrating the floor"
# --------------------------------------------------------------------------- #

def scoring_tests() -> list[bool]:
    """README: "MIN_SIMILARITY and TITLE_ONLY_PENALTY multiply — calibrate the
    product, not the distribution. A bare title survives only at
    raw >= MIN_SIMILARITY / 0.85."

    Asserted here so that changing either constant trips a test rather than
    silently excluding every abstract-less paper (which is what a floor of 0.75
    would do — the bar lands at 0.882, above two of the five measured pools).
    """
    print("\nfloor x title-only penalty coupling")
    results = []
    bar = MIN_SIMILARITY / TITLE_ONLY_PENALTY
    results.append(check(f"a bare title needs raw >= {bar:.3f} to clear the floor "
                         f"({MIN_SIMILARITY} / {TITLE_ONLY_PENALTY})",
                         abs(bar - 0.8235) < 0.001, f"bar={bar:.4f}"))
    results.append(check("that bar sits under the 0.880-0.912 measured pool ceilings, "
                         "so abstract-less papers stay reachable",
                         bar < 0.880, f"bar={bar:.4f}"))
    results.append(check("a floor of 0.75 would push the bar above two measured "
                         "pool ceilings (0.793, 0.848) — documented, not shipped",
                         0.75 / TITLE_ONLY_PENALTY > 0.848,
                         f"would be {0.75 / TITLE_ONLY_PENALTY:.4f}"))

    print("\nconfidence discount")
    p = Paper(key="k", similarity=0.800, pdf_verified=True)
    results.append(check("a verified PDF keeps the raw similarity as confidence",
                         p.confidence == 0.8, f"got {p.confidence}"))
    p.pdf_verified = False
    results.append(check("an unverified PDF is discounted to 0.7x",
                         p.confidence == 0.56, f"got {p.confidence}"))
    results.append(check("`similarity` itself is never mutated by the discount",
                         p.similarity == 0.800, f"got {p.similarity}"))

    print("\nfetch target selection")
    r = bare_resolver(url_filter=None)
    p = Paper(key="k", pdf_urls=["https://blocked.test/a.pdf", "https://arxiv.org/b.pdf"])
    results.append(check("with no filter, the top-ranked URL is the target",
                         r._fetch_target(p) == "https://blocked.test/a.pdf"))
    r.url_filter = lambda u: "arxiv.org" in u
    results.append(check("with an allowlist, the first ACCEPTABLE URL is the target "
                         "(stops selection spending slots on future BLOCKs)",
                         r._fetch_target(p) == "https://arxiv.org/b.pdf"))
    r.url_filter = lambda u: False
    results.append(check("a paper with no acceptable URL has no target",
                         r._fetch_target(p) is None))
    return results


# --------------------------------------------------------------------------- #
# Manifest contract — README "Manifest schema"
# --------------------------------------------------------------------------- #

def manifest_tests() -> list[bool]:
    """README: the manifest is "a superset of source-tool's, so one consumer can
    read both. Every record carries source-tool's five keys with the same
    meanings."

    A downstream reader doing `record["resolution"]["pdf_url"]` is the documented
    contract, so the block must exist on EVERY status — including the unresolved
    path, where there is no paper at all.
    """
    print("\nmanifest schema")
    results = []
    resolution = Resolution(query_description="a description",
                            expanded_queries=["a description", "variant"],
                            errors={"semantic_scholar": "429"},
                            pool_size=75, truncated_sources=["arxiv"])
    paper = Paper(key="k", title="A Title", doi="10.1/x", similarity=0.9,
                  pdf_verified=True, rank=1, similarity_rank=3,
                  pdf_urls=["https://a.test/1.pdf", "https://b.test/2.pdf"],
                  retrieval_sources=["arxiv", "openalex"])

    st_keys = {"pdf_url", "doi", "matched_title", "confidence", "source",
               "alternatives", "errors"}
    block = source_tool_resolution(paper, resolution)
    results.append(check("the source-tool resolution block carries all its keys",
                         st_keys <= block.keys(), f"missing {st_keys - block.keys()}"))
    results.append(check("`source` joins every retrieval source",
                         block["source"] == "arxiv+openalex", f"got {block['source']}"))
    results.append(check("other hosts for the same work land in `alternatives`",
                         block["alternatives"] == [{"pdf_url": "https://b.test/2.pdf"}],
                         f"got {block['alternatives']}"))

    empty = source_tool_resolution(None, resolution)
    results.append(check("the block exists with the same keys when nothing resolved "
                         "(a MISS record must not KeyError downstream)",
                         st_keys <= empty.keys() and empty["pdf_url"] is None
                         and empty["confidence"] == 0.0,
                         f"got {empty}"))
    results.append(check("run errors are carried on both the hit and miss paths",
                         block["errors"] == empty["errors"] == {"semantic_scholar": "429"}))

    meta = run_metadata(resolution)
    run_keys = {"query_description", "expanded_queries", "expansion_error",
                "embedding_model", "min_similarity", "selection", "verified_window",
                "pool_size", "truncated_sources", "errors"}
    results.append(check("the run block carries all documented keys",
                         run_keys <= meta.keys(), f"missing {run_keys - meta.keys()}"))

    paper_keys = {"rank", "similarity_rank", "similarity", "confidence", "title",
                  "doi", "abstract", "title_only", "pdf_url", "pdf_urls",
                  "landing_url", "pdf_verified", "retrieval_sources", "matched_queries"}
    as_dict = paper.as_dict()
    results.append(check("the paper block carries all documented keys",
                         paper_keys <= as_dict.keys(),
                         f"missing {paper_keys - as_dict.keys()}"))
    results.append(check("rank and similarity_rank are BOTH reported, so a "
                         "reordering is visible rather than implied",
                         as_dict["rank"] == 1 and as_dict["similarity_rank"] == 3))
    results.append(check("the whole record is JSON-serializable",
                         bool(json.dumps({"resolution": block, "run": meta,
                                          "paper": as_dict}))))

    print("\nfilename slugs")
    results.append(check("the slug is derived from the title and hashed on the URL",
                         slugify(paper).startswith("a-title_")
                         and slugify(paper).endswith(".pdf"), f"got {slugify(paper)}"))
    results.append(check("the same paper always produces the same filename "
                         "(this is what makes HAVE work)",
                         slugify(paper) == slugify(paper)))
    other = Paper(key="k", title="A Title", pdf_urls=["https://elsewhere.test/9.pdf"])
    results.append(check("a different host for the same title gets a different name "
                         "(no silent overwrite)",
                         slugify(paper) != slugify(other)))
    long_title = Paper(key="k", title="word " * 60, pdf_urls=["https://a.test/x.pdf"])
    stem = slugify(long_title).rsplit("_", 1)[0]
    results.append(check("an overlong title is truncated to a usable filename",
                         len(stem) <= 80, f"stem was {len(stem)} chars"))
    results.append(check("a title of pure punctuation still yields a valid name",
                         slugify(Paper(key="k", title="!!! ???")).startswith("paper_")))
    return results


# --------------------------------------------------------------------------- #
# Query expander — README: "deliberately unable to break a run"
# --------------------------------------------------------------------------- #

async def expander_tests() -> list[bool]:
    """README: "Every failure path (Ollama not running, model not pulled, timeout,
    invalid JSON, valid JSON of the wrong shape) returns just the original
    description plus a reason string."

    Expansion is optional, so its only hard requirement is that it never raises.
    """
    print("\nexpander response parsing")
    results = []
    parse = query_expander._parse

    queries, err = parse('{"queries": ["a b c", "d e f"]}', 4)
    results.append(check("the documented shape parses", queries == ["a b c", "d e f"]
                         and err is None, f"got {queries} / {err}"))
    queries, err = parse('["a b", "c d"]', 4)
    results.append(check("a bare list is accepted as a lenient alias",
                         queries == ["a b", "c d"], f"got {queries}"))
    queries, err = parse('{"search_queries": ["a b"]}', 4)
    results.append(check("a single list under an unexpected key is accepted",
                         queries == ["a b"], f"got {queries}"))
    queries, err = parse('{"a": [1], "b": [2]}', 4)
    results.append(check("two candidate lists are ambiguous -> reported, not guessed",
                         queries == [] and err is not None, f"got {queries} / {err}"))
    queries, err = parse("not json at all", 4)
    results.append(check("unparseable output is an error string, not an exception",
                         queries == [] and "unparseable" in err, f"got {err}"))
    queries, err = parse('{"queries": []}', 4)
    results.append(check("an empty query list is reported as unusable",
                         queries == [] and err is not None, f"got {err}"))
    queries, err = parse('{"queries": ["a", "b", "c", "d", "e", "f"]}', 2)
    results.append(check("more variants than requested are truncated",
                         len(queries) == 2, f"got {queries}"))

    print("\nexpander query normalization")
    norm = query_expander._normalize
    results.append(check("snake_case is split into words",
                         norm("neural_networks_focus_selection") == "neural networks focus selection"))
    results.append(check("camelCase is split into words",
                         norm("AttentionMechanismSeq") == "Attention Mechanism Seq",
                         f"got {norm('AttentionMechanismSeq')!r}"))
    results.append(check("quotes and hyphens are flattened",
                         norm('"seq-to-seq"  models') == "seq to seq models",
                         f"got {norm(chr(34)+'seq-to-seq'+chr(34)+'  models')!r}"))

    print("\nexpander failure paths (never breaks a run)")
    # Port 1 is closed everywhere; this is a local socket, not a network call.
    queries, err = await query_expander.expand("a topic", url="http://127.0.0.1:1",
                                               timeout=2.0)
    results.append(check("an unreachable Ollama returns the description + a reason",
                         queries == ["a topic"] and err and "unreachable" in err,
                         f"got {queries} / {err}"))
    queries, err = await query_expander.expand("a topic", variants=0)
    results.append(check("variants=0 short-circuits with no request at all",
                         queries == ["a topic"] and err is None))
    queries, err = await query_expander.expand("", variants=4)
    results.append(check("an empty description does not call the model",
                         queries == [""] and err is None, f"got {queries} / {err}"))
    return results


def expander_dedup_tests() -> list[bool]:
    """queries[0] must always be the original description — `_rank` scores against
    it, and the manifest reports it as `query`."""
    print("\nexpander output contract")
    results = []

    async def fake_expand(dupe: str):
        original = query_expander._parse
        try:
            query_expander._parse = lambda payload, limit: ([dupe, "a distinct query"], None)

            class Client:
                async def post(self, *a, **kw):
                    return httpx.Response(200, json={"response": "{}"},
                                          request=httpx.Request("POST", "http://x"))
                async def aclose(self):
                    pass
            return await query_expander.expand("Attention In NMT", client=Client())
        finally:
            query_expander._parse = original

    queries, _ = asyncio.run(fake_expand("attention in nmt"))
    results.append(check("the description always leads the query list",
                         queries[0] == "Attention In NMT", f"got {queries}"))
    results.append(check("a variant that only restates the description is dropped "
                         "(case-insensitively), saving 5 wasted API calls",
                         queries == ["Attention In NMT", "a distinct query"],
                         f"got {queries}"))
    return results


# --------------------------------------------------------------------------- #
# Rate limiting — README: "these are the APIs' own published limits"
# --------------------------------------------------------------------------- #

async def rate_limit_tests() -> list[bool]:
    """arXiv publishes 1-per-3s and NCBI/EBI 3-per-s. The limiter is what keeps
    the fan-out inside those, and exceeding them is how the IP gets throttled."""
    print("\nrate limiter")
    results = []
    limiter = RateLimiter(0.15)
    start = time.monotonic()
    for _ in range(3):
        async with limiter:
            pass
    elapsed = time.monotonic() - start
    results.append(check("three acquisitions of a 0.15s gate take >= 0.30s",
                         elapsed >= 0.30, f"took {elapsed:.3f}s"))

    limiter = RateLimiter(0.0)
    start = time.monotonic()
    for _ in range(5):
        async with limiter:
            pass
    results.append(check("a zero interval does not sleep",
                         time.monotonic() - start < 0.05))

    # The real limiter table, built the way __init__ does it.
    real = SemanticResolver(timeout=1.0)
    try:
        paced = set(real._limiters)
        results.append(check("arXiv, S2, EBI and both PMC PDF hosts are paced",
                             {"export.arxiv.org", "api.semanticscholar.org",
                              "www.ebi.ac.uk", "www.ncbi.nlm.nih.gov",
                              "europepmc.org"} <= paced,
                             f"got {sorted(paced)}"))
        results.append(check("arXiv keeps its published 1-per-3s interval",
                             real._limiters["export.arxiv.org"]._min_interval == 3.0))
    finally:
        await real._client.aclose()
    return results


# --------------------------------------------------------------------------- #

async def main() -> int:
    results: list[bool] = []
    results += allowlist_tests()
    results += await download_tests()
    results += await parser_tests()
    results += dedup_tests()
    results += text_tests()
    results += scoring_tests()
    results += manifest_tests()
    results += await expander_tests()
    results += await rate_limit_tests()

    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    code = asyncio.run(main())
    # Sits outside the loop above: it runs its own asyncio.run() to exercise
    # expand() with a stubbed client.
    extra = expander_dedup_tests()
    print(f"{sum(extra)}/{len(extra)} passed (expander output contract)")
    raise SystemExit(code or (0 if all(extra) else 1))
