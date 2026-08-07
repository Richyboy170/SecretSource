"""Offline tests for top-k selection.

    py test_selection.py

No network: `_verify_pdf` is stubbed so a URL's fetchability is whatever the
test says it is. That is the point — selection is the piece where a mistake is
silent (you get plausible results, just not the ones you could have had), and
live APIs make it untestable. Verified live behaviour is in the README; this
file covers the branches, including the ones a live run rarely reaches.
"""

from __future__ import annotations

import asyncio

from semantic_resolver import (
    VERIFY_WINDOW_CAP,
    Paper,
    Resolution,
    SemanticResolver,
)


def make_papers(spec: list[tuple[float, str | None]]) -> list[Paper]:
    """Build ranked papers from (similarity, pdf_url) pairs, best first."""
    papers = []
    for i, (sim, url) in enumerate(spec, 1):
        p = Paper(key=f"k{i}", title=f"paper {i}", similarity=sim,
                  pdf_urls=[url] if url else [])
        p.similarity_rank = i
        papers.append(p)
    return papers


async def select(papers, top_k, *, fetchable=(), **kwargs):
    """Run _select with `_verify_pdf` answering True only for `fetchable` URLs."""
    resolver = SemanticResolver.__new__(SemanticResolver)  # no client, no network
    resolver.verify_pdfs = kwargs.pop("verify_pdfs", True)
    resolver.rank_only = kwargs.pop("rank_only", False)
    resolver.url_filter = kwargs.pop("url_filter", None)
    calls: list[str] = []

    async def fake_verify(url: str) -> bool:
        calls.append(url)
        return url in fetchable

    resolver._verify_pdf = fake_verify
    resolution = Resolution(query_description="q")
    kept = await resolver._select(papers, top_k, resolution)
    return kept, calls, resolution


def check(name: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{'' if condition else '  <- ' + detail}")
    return condition


async def main() -> int:
    results = []

    # The regression this feature exists for: the three highest-scoring papers
    # have no PDF, so similarity-only selection returns nothing downloadable.
    print("\npdf-preferring selection")
    papers = make_papers([
        (0.50, None), (0.49, None), (0.48, None),
        (0.40, "a"), (0.39, "b"), (0.38, "c"),
    ])
    kept, _, res = await select(papers, 3, fetchable={"a", "b", "c"})
    results.append(check("picks fetchable over higher-scoring unfetchable",
                         [p.pdf_url for p in kept] == ["a", "b", "c"],
                         f"got {[p.pdf_url for p in kept]}"))
    results.append(check("similarity_rank records the pure-score position",
                         [p.similarity_rank for p in kept] == [4, 5, 6],
                         f"got {[p.similarity_rank for p in kept]}"))
    results.append(check("run records the mode", res.selection == "pdf_preferring"))

    # Ordering must stay monotonic — only membership is PDF-aware.
    sims = [p.similarity for p in kept]
    results.append(check("kept papers stay in descending similarity",
                         all(a >= b for a, b in zip(sims, sims[1:])), f"got {sims}"))

    # A URL that exists but does not serve a PDF (paywall, 403) must not win a
    # slot — the failure mode a "has a pdf_url" check cannot see.
    print("\nverification, not just presence of a url")
    papers = make_papers([(0.50, "paywall"), (0.40, "real")])
    kept, _, _ = await select(papers, 1, fetchable={"real"})
    results.append(check("unverified url loses to a verified one",
                         [p.pdf_url for p in kept] == ["real"],
                         f"got {[p.pdf_url for p in kept]}"))

    # Fewer fetchable than k: the rest still surface, so a MISS line can report
    # that a relevant paper exists behind a paywall.
    print("\nbackfill")
    papers = make_papers([(0.50, None), (0.40, "a"), (0.30, None)])
    kept, _, _ = await select(papers, 3, fetchable={"a"})
    results.append(check("unfetchable papers backfill the empty slots",
                         len(kept) == 3, f"got {len(kept)}"))
    results.append(check("backfilled set is still sorted by similarity",
                         [round(p.similarity, 2) for p in kept] == [0.50, 0.40, 0.30],
                         f"got {[round(p.similarity, 2) for p in kept]}"))
    results.append(check("no paper is selected twice",
                         len({id(p) for p in kept}) == 3))

    print("\n--rank-only")
    papers = make_papers([(0.50, None), (0.40, "a")])
    kept, _, res = await select(papers, 2, fetchable={"a"}, rank_only=True)
    results.append(check("keeps pure similarity order",
                         [p.similarity_rank for p in kept] == [1, 2],
                         f"got {[p.similarity_rank for p in kept]}"))
    results.append(check("run records the mode", res.selection == "rank_only"))

    print("\n--dry-run / verify_pdfs=False")
    papers = make_papers([(0.50, None), (0.40, "a")])
    kept, calls, _ = await select(papers, 1, fetchable={"a"}, verify_pdfs=False)
    results.append(check("issues no probe requests", calls == [], f"probed {calls}"))
    results.append(check("still prefers a paper that claims a url",
                         [p.pdf_url for p in kept] == ["a"],
                         f"got {[p.pdf_url for p in kept]}"))

    # A paper the runner would refuse to download is not a useful pick, however
    # fetchable it is — this is what stops selection filling k with BLOCKs.
    print("\nurl_filter (allowlist awareness)")
    papers = make_papers([(0.50, "https://blocked.test/x.pdf"),
                          (0.40, "https://arxiv.org/y.pdf")])
    kept, calls, _ = await select(
        papers, 1,
        fetchable={"https://blocked.test/x.pdf", "https://arxiv.org/y.pdf"},
        url_filter=lambda u: "arxiv.org" in u,
    )
    results.append(check("prefers a paper on an acceptable host",
                         [p.pdf_url for p in kept] == ["https://arxiv.org/y.pdf"],
                         f"got {[p.pdf_url for p in kept]}"))
    results.append(check("never probes a url the caller would refuse",
                         "https://blocked.test/x.pdf" not in calls, f"probed {calls}"))

    # A fixed 3x window missed these; widening is what finds them.
    print("\nadaptive widening")
    papers = make_papers([(1.0 - i / 100, None) for i in range(20)]
                         + [(0.5, "deep1"), (0.49, "deep2")])
    kept, calls, res = await select(papers, 2, fetchable={"deep1", "deep2"})
    results.append(check("finds fetchable papers past the first batch",
                         [p.pdf_url for p in kept] == ["deep1", "deep2"],
                         f"got {[p.pdf_url for p in kept]}"))
    results.append(check("probes beyond one batch to do it",
                         res.verified_window > 6, f"window={res.verified_window}"))

    # An easy query must not pay for the widening it does not need.
    papers = make_papers([(0.9, "a"), (0.8, "b")] + [(0.5, None)] * 40)
    kept, calls, res = await select(papers, 2, fetchable={"a", "b"})
    results.append(check("stops early when the head is fetchable",
                         res.verified_window <= 6, f"window={res.verified_window}"))

    # Without a cap a hopeless query would probe the whole pool.
    papers = make_papers([(0.5, f"u{i}") for i in range(200)])
    kept, calls, res = await select(papers, 5, fetchable=set())
    results.append(check("caps probing on a pool where nothing is fetchable",
                         res.verified_window <= VERIFY_WINDOW_CAP,
                         f"window={res.verified_window}"))
    results.append(check("still returns k papers when none are fetchable",
                         len(kept) == 5, f"got {len(kept)}"))

    # Fewer candidates than requested must not crash or pad.
    print("\nedge cases")
    kept, _, _ = await select(make_papers([(0.5, "a")]), 5, fetchable={"a"})
    results.append(check("returns what exists when the pool is smaller than k",
                         len(kept) == 1, f"got {len(kept)}"))
    kept, calls, _ = await select([], 5)
    results.append(check("handles an empty pool", kept == [] and calls == []))

    print("\n--rank-only provenance")
    papers = make_papers([(0.50, "a"), (0.40, "b")])
    _, _, res = await select(papers, 2, fetchable={"a"}, rank_only=True)
    results.append(check("reports the probing it actually did",
                         res.verified_window == 2, f"got {res.verified_window}"))

    results.extend(await retry_after_tests())
    results.extend(await probe_pacing_tests())

    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


async def probe_pacing_tests() -> list[bool]:
    """PDF probing must not burst at one host.

    Selection probes a widening window and those candidates cluster on very few
    hosts — 11 of 12 results in one measured batch were arxiv.org. Unbounded
    that is dozens of simultaneous connections to one server, which is how a
    tool earns an IP block. Guards the cap, since nothing in a passing live run
    would reveal its absence.
    """
    import httpx

    from semantic_resolver import PROBE_CONCURRENCY

    print("\nprobe pacing")
    results = []
    live = 0
    peak = {}

    class Client:
        async def head(self, url, **kw):
            nonlocal live
            live += 1
            host = httpx.URL(url).host
            peak[host] = max(peak.get(host, 0), live)
            await asyncio.sleep(0)  # let every queued probe pile up if it can
            live -= 1
            return httpx.Response(
                200, headers={"content-type": "application/pdf"},
                request=httpx.Request("HEAD", url))

    r = SemanticResolver.__new__(SemanticResolver)
    r._client = Client()
    r._limiters = {}
    r._probe_gates = {}
    urls = [f"https://arxiv.org/pdf/{i}" for i in range(20)]
    await asyncio.gather(*(r._verify_pdf(u) for u in urls))
    results.append(check(f"caps concurrent probes per host at {PROBE_CONCURRENCY}",
                         peak.get("arxiv.org", 0) <= PROBE_CONCURRENCY,
                         f"peaked at {peak.get('arxiv.org')}"))

    # A shared cap would throttle unrelated hosts; the gate is per host.
    r2 = SemanticResolver.__new__(SemanticResolver)
    r2._client = Client()
    r2._limiters = {}
    r2._probe_gates = {}
    await asyncio.gather(*(r2._verify_pdf(f"https://h{i}.test/x.pdf") for i in range(8)))
    results.append(check("does not serialise across different hosts",
                         len(r2._probe_gates) == 8, f"gates={len(r2._probe_gates)}"))
    return results


async def retry_after_tests() -> list[bool]:
    """A long Retry-After must abandon the source, not sleep on it.

    OpenAlex answers an exhausted quota with `Retry-After: 36494`. Obeying that
    literally hung the whole tool with no output, so this is a regression guard:
    the failure is invisible (it looks like a slow network) and the fix is one
    comparison that is easy to drop.
    """
    import httpx

    from semantic_resolver import RETRY_AFTER_CAP

    print("\nRetry-After handling")
    results = []
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    def resolver_with(retry_after: str):
        r = SemanticResolver.__new__(SemanticResolver)
        r.max_retries = 3
        r._limiters = {}
        r._auth_headers = {}
        r._s2_key = None
        request = httpx.Request("GET", "https://api.example.test/works")
        response = httpx.Response(429, headers={"Retry-After": retry_after},
                                  request=request)

        class Client:
            async def get(self, *a, **kw):
                return response

        r._client = Client()
        return r

    original_sleep = asyncio.sleep
    asyncio.sleep = fake_sleep
    try:
        slept.clear()
        try:
            await resolver_with(str(int(RETRY_AFTER_CAP * 100)))._get(
                "https://api.example.test/works")
            results.append(check("long Retry-After raises instead of waiting", False,
                                 "no exception"))
        except httpx.HTTPStatusError:
            results.append(check("long Retry-After raises instead of waiting", True))
        results.append(check("and never sleeps", slept == [], f"slept {slept}"))

        slept.clear()
        short = str(int(RETRY_AFTER_CAP) - 1)
        try:
            await resolver_with(short)._get("https://api.example.test/works")
        except httpx.HTTPStatusError:
            pass
        results.append(check("short Retry-After is still honoured",
                             slept and all(s == float(short) for s in slept),
                             f"slept {slept}"))
    finally:
        asyncio.sleep = original_sleep
    return results


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
