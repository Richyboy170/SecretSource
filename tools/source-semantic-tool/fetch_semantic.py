"""Find papers by description and download the top matches from allowed sites.

    py fetch_semantic.py "transformer models for protein structure prediction"
    py fetch_semantic.py "..." -k 10 --min-similarity 0.45
    py fetch_semantic.py "..." --min-similarity 0.0 --dry-run   # see raw scores
    py fetch_semantic.py --from-file descriptions.txt

The allowlist lives in sources.txt — edit that file to declare which sites you
trust. A PDF hosted anywhere else is reported as "blocked" and never written to
disk.

This tool is self-contained: the allowlist and download helpers below are
deliberate copies of source-tool's rather than imports, so the folder can be
moved on its own. The tradeoff is that a fix to either tool's download path has
to be applied twice.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

from embedder import DEFAULT_MODEL
from semantic_resolver import (
    DEFAULT_PER_SOURCE,
    DEFAULT_TOP_K,
    MIN_SIMILARITY,
    Paper,
    Resolution,
    SemanticResolver,
)

MIN_PDF_BYTES = 1024  # anything smaller is an error page, not a paper


@dataclass
class Outcome:
    # `query` is ALWAYS the input description, never a result title — same
    # meaning the field has in source-tool's manifest. The rank+title string
    # used for console display lives in `label`, so one field never carries
    # both an input and an output depending on which branch produced it.
    query: str
    status: str  # downloaded | skipped | unresolved | blocked | failed
    path: str | None = None
    detail: str | None = None
    label: str | None = None
    resolution: dict | None = None  # source-tool-shaped, for shared consumers
    paper: dict | None = None       # the semantic-specific payload
    run: dict | None = None


# --------------------------------------------------------------------------- #
# Allowlist
# --------------------------------------------------------------------------- #

def load_sources(path: Path) -> set[str]:
    """Read sources.txt into a set of bare hosts. Accepts full URLs, bare domains,
    or www-prefixed forms and normalizes all three to the same key."""
    hosts = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        # Strip trailing comments too, not just whole-line ones: an inline "# note"
        # would otherwise be parsed as part of the host and silently never match.
        line = raw.split("#")[0].strip()
        if not line:
            continue
        if "//" not in line:
            line = "https://" + line
        host = (httpx.URL(line).host or "").lower().removeprefix("www.")
        if host:
            hosts.add(host)
    return hosts


def is_allowed(url: str, hosts: set[str]) -> bool:
    """Suffix-anchored match so a listed domain covers its subdomains without
    also matching lookalikes — arxiv.org allows export.arxiv.org but not
    arxiv.org.example.com."""
    host = (httpx.URL(url).host or "").lower().removeprefix("www.")
    return any(host == h or host.endswith("." + h) for h in hosts)


# --------------------------------------------------------------------------- #
# Input and naming
# --------------------------------------------------------------------------- #

def parse_descriptions(path: Path) -> list[str]:
    """Batch input: one description per line. Blank lines and # comments ignored."""
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def slugify(paper: Paper) -> str:
    """Stable filename from the matched title.

    Suffixed with a hash of the PDF URL rather than the DOI. source-tool uses the
    work-level DOI and its own README documents the consequence: when an index
    has merged a predatory mirror into a work record, the right file lands under
    a junk DOI, and that wrong identity reaches the filename. A URL hash gives
    the same collision-avoidance without asserting an identity we never verified.
    """
    basis = paper.title or "paper"
    slug = re.sub(r"[^a-z0-9]+", "-", basis.lower()).strip("-")[:80] or "paper"
    digest = hashlib.sha256((paper.pdf_url or basis).encode("utf-8")).hexdigest()[:10]
    return f"{slug}_{digest}.pdf"


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #

async def download(client: httpx.AsyncClient, url: str, dest: Path,
                   hosts: set[str] | None) -> None:
    """Stream to a temp file, validating the PDF header before committing so a
    paywall page or truncated transfer never lands in the output directory."""
    tmp = dest.with_suffix(".part")
    try:
        async with client.stream("GET", url) as response:
            # Redirects can leave the allowlist — re-check where we actually landed.
            if hosts and not is_allowed(str(response.url), hosts):
                raise ValueError(f"redirected off-allowlist to {response.url.host}")
            response.raise_for_status()
            first = True
            with tmp.open("wb") as fh:
                async for chunk in response.aiter_bytes(65_536):
                    if first:
                        if not chunk.startswith(b"%PDF-"):
                            raise ValueError("response is not a PDF")
                        first = False
                    fh.write(chunk)
        if tmp.stat().st_size < MIN_PDF_BYTES:
            raise ValueError(f"file too small ({tmp.stat().st_size} bytes)")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #

def source_tool_resolution(paper: Paper | None, resolution: Resolution) -> dict:
    """The record shaped like source-tool's `resolution` block.

    source-tool's README tells downstream code to read provenance out of
    `record["resolution"]`. Emitting the same block here is what makes a single
    consumer able to read both manifests — without it, `resolution["pdf_url"]`
    would KeyError on every record this tool writes.

    `confidence` carries source-tool's meaning (score discounted when the PDF was
    not confirmed fetchable), but the underlying score is a cosine against a
    description, NOT evidence of identity. See README limitation #1.
    """
    if paper is None:
        return {
            "pdf_url": None, "doi": None, "matched_title": None,
            "confidence": 0.0, "source": None, "alternatives": [],
            "errors": resolution.errors,
        }
    return {
        "pdf_url": paper.pdf_url,
        "doi": paper.doi,
        "matched_title": paper.title,
        "confidence": paper.confidence,
        "source": "+".join(paper.retrieval_sources) or None,
        # Other hosts serving this same work — the analogue of source-tool's
        # alternative candidates, and what you would reach for if the chosen
        # host is blocked or 403s.
        "alternatives": [{"pdf_url": u} for u in paper.pdf_urls[1:4]],
        "errors": resolution.errors,
    }


def run_metadata(resolution: Resolution) -> dict:
    """Run-level context repeated on every manifest record, so a single line is
    self-describing without needing the rest of the file."""
    return {
        "query_description": resolution.query_description,
        "expanded_queries": resolution.expanded_queries,
        "expansion_error": resolution.expansion_error,
        "embedding_model": resolution.embedding_model,
        "min_similarity": resolution.min_similarity,
        "selection": resolution.selection,
        "verified_window": resolution.verified_window,
        "pool_size": resolution.pool_size,
        "truncated_sources": resolution.truncated_sources,
        "errors": resolution.errors,
    }


async def process(description: str, resolver: SemanticResolver,
                  client: httpx.AsyncClient, out_dir: Path, hosts: set[str] | None,
                  top_k: int, min_similarity: float | None,
                  semaphore: asyncio.Semaphore) -> tuple[Resolution | None, list[Outcome]]:
    async with semaphore:
        try:
            resolution = await resolver.resolve(
                description, top_k=top_k, min_similarity=min_similarity
            )
        except Exception as exc:
            return None, [Outcome(description, "failed",
                                  detail=f"{type(exc).__name__}: {exc}")]

        meta = run_metadata(resolution)

        if not resolution.papers:
            detail = (f"nothing above the similarity floor "
                      f"({resolution.min_similarity}) among {resolution.pool_size} candidates")
            if resolution.errors and not resolution.pool_size:
                detail += "; " + "; ".join(f"{k}={v}" for k, v in resolution.errors.items())
            return resolution, [Outcome(description, "unresolved", detail=detail,
                                        resolution=source_tool_resolution(None, resolution),
                                        run=meta)]

        outcomes = []

        def record(paper: Paper, status: str, **kw) -> Outcome:
            return Outcome(description, status,
                           label=f"[{paper.rank}] {paper.title or '(untitled)'}",
                           resolution=source_tool_resolution(paper, resolution),
                           paper=paper.as_dict(), run=meta, **kw)

        for paper in resolution.papers:
            allowed = [u for u in paper.pdf_urls if not hosts or is_allowed(u, hosts)]
            if not allowed:
                if paper.pdf_urls:
                    seen = sorted({httpx.URL(u).host for u in paper.pdf_urls if u})
                    outcomes.append(record(
                        paper, "blocked",
                        detail=f"no allowed host among: {', '.join(h for h in seen if h)}"))
                else:
                    outcomes.append(record(
                        paper, "unresolved",
                        detail="no open-access PDF found for this paper"))
                continue

            # Kept only so the manifest can still show where else this work lives;
            # nothing off the allowlist is ever fetched.
            blocked_urls = [u for u in paper.pdf_urls if u not in allowed]

            # Try every allowed host before giving up. One work is routinely
            # listed on several — a publisher that 403s, a repository mirror, a
            # PMC copy — and the first is not the likeliest to serve. Measured
            # 2026-08-07: a EuropePMC hit failed on its chosen URL while an MDPI
            # copy of the same paper sat unused in `alternatives`. Stopping at the
            # first error throws away a PDF we already know how to reach.
            #
            # Each attempt re-runs the full guard chain inside download() — host
            # allowlist after redirect, %PDF- sniff, size floor — so falling
            # through relaxes persistence, never safety.
            def promote(url: str) -> Path:
                """Move `url` to the front and return where it would be saved.

                pdf_url is pdf_urls[0], and it feeds both the filename hash and
                the manifest, so promoting the URL under consideration is what
                makes the record name the host that actually served.
                """
                paper.pdf_urls = [url] + [u for u in allowed if u != url] + blocked_urls
                return out_dir / slugify(paper)

            # A file on disk for *any* allowed host means we already have this
            # paper. Check them all before touching the network: the filename is
            # a hash of the URL, so testing only the first host would re-issue
            # last run's failed requests on every re-run and still end in HAVE —
            # the wasted calls would never show up in the output.
            have = next((d for u in allowed if (d := promote(u)).exists()), None)
            if have is not None:
                outcomes.append(record(paper, "skipped", path=str(have)))
                continue

            attempts: list[str] = []
            for url in allowed:
                dest = promote(url)
                try:
                    await download(client, url, dest, hosts)
                except Exception as exc:
                    # First line only — httpx appends a multi-line "For more
                    # information check: <mdn url>" footer to status errors, and
                    # with several hosts joined that footer is most of the detail.
                    first = str(exc).splitlines()[0] if str(exc) else ""
                    attempts.append(f"{httpx.URL(url).host}: {type(exc).__name__}: {first}")
                    continue
                outcomes.append(record(paper, "downloaded", path=str(dest)))
                break
            else:
                # Every allowed host failed. Restore preference order first, so
                # the manifest's pdf_url is the top-ranked allowed URL rather than
                # whichever one happened to be tried last.
                paper.pdf_urls = allowed + blocked_urls
                # Report every host that failed, not just the last one.
                outcomes.append(record(paper, "failed", detail="; ".join(attempts)))
        return resolution, outcomes


def print_ranking(resolution: Resolution) -> None:
    """Dry-run view. Prints the score distribution over the whole pool, which is
    what calibrating --min-similarity actually needs."""
    print(f'\n"{resolution.query_description}"')
    if resolution.expansion_error:
        print(f"  expansion unavailable: {resolution.expansion_error}")
    for i, q in enumerate(resolution.expanded_queries[1:], 1):
        print(f"  query[{i}]: {q}")
    print(f"  pool={resolution.pool_size}  floor={resolution.min_similarity}"
          f"  model={resolution.embedding_model}")
    if resolution.truncated_sources:
        print(f"  truncated (hit --per-source cap): {', '.join(resolution.truncated_sources)}")
    for name, err in resolution.errors.items():
        print(f"  ! {name}: {err[:110]}")

    scores = resolution.scored
    if scores:
        n = len(scores)
        # scores arrive sorted descending, so index n*p is p of the way DOWN
        # from the best — labelled "top10%" rather than "p10" to avoid reading
        # like a 10th-percentile (i.e. near-worst) figure.
        at = lambda p: scores[min(n - 1, int(n * p))]
        print(f"  scores: max={scores[0]:.3f}  top10%={at(0.10):.3f}  "
              f"median={at(0.50):.3f}  min={scores[-1]:.3f}")

    if not resolution.papers:
        print("  MISS — nothing cleared the floor")
    for paper in resolution.papers:
        flag = " (title only)" if paper.title_only else ""
        print(f"  {paper.similarity:.3f}  {(paper.title or '(untitled)')[:88]}{flag}")
        print(f"         {'+'.join(paper.retrieval_sources)}"
              f"  q{paper.matched_queries}"
              f"  {paper.pdf_url or 'no PDF'}")


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find and download open-access papers matching a description.")
    parser.add_argument("description", nargs="*", help="what the papers should be about")
    parser.add_argument("--from-file", type=Path, help="batch input: one description per line")
    parser.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--min-similarity", type=float, default=None,
                        help=f"cosine floor (default {MIN_SIMILARITY}); 0.0 shows everything")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model")
    parser.add_argument("--per-source", type=int, default=DEFAULT_PER_SOURCE,
                        help="max results per source per query")
    parser.add_argument("--no-expand", action="store_true", help="skip LLM query expansion")
    parser.add_argument("--rank-only", action="store_true",
                        help="fill the top-k strictly by similarity, even with candidates "
                             "that have no fetchable PDF (default prefers fetchable ones)")
    parser.add_argument("--variants", type=int, default=4, help="query variants to generate")
    parser.add_argument("--ollama-model", default=None)
    parser.add_argument("--ollama-url", default=None)
    parser.add_argument("--sources", default="sources.txt", type=Path, help="allowed-site list")
    parser.add_argument("--out", default="output_pdf", type=Path)
    parser.add_argument("--workers", type=int, default=2, help="concurrent descriptions")
    parser.add_argument("--any-host", action="store_true", help="ignore sources.txt")
    parser.add_argument("--dry-run", action="store_true",
                        help="rank and print scores, download nothing")
    args = parser.parse_args()

    descriptions: list[str] = []
    if args.from_file:
        if not args.from_file.exists():
            print(f"input file not found: {args.from_file}", file=sys.stderr)
            return 1
        descriptions += parse_descriptions(args.from_file)
    if args.description:
        descriptions.append(" ".join(args.description))

    if not descriptions:
        parser.print_usage(sys.stderr)
        print("\ngive a description or --from-file", file=sys.stderr)
        return 1

    hosts = None
    if not args.any_host and not args.dry_run:
        if not args.sources.exists():
            print(f"allowlist not found: {args.sources} (use --any-host to skip)", file=sys.stderr)
            return 1
        hosts = load_sources(args.sources)
        if not hosts:
            print(f"{args.sources} lists no sites", file=sys.stderr)
            return 1

    if not args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)
    scope = f"{len(hosts)} allowed sites" if hosts else "any host"
    where = "ranking only (dry run)" if args.dry_run else f"{scope} -> {args.out}/"
    mode = "similarity only" if args.rank_only else "fetchable-first"
    print(f"searching {len(descriptions)} description(s); {where}; {mode}")

    semaphore = asyncio.Semaphore(max(1, args.workers))
    async with SemanticResolver(
        model=args.model,
        per_source=args.per_source,
        expand=not args.no_expand,
        rank_only=args.rank_only,
        # Let selection prefer papers this run could actually keep. Without it a
        # restrictive sources.txt makes the resolver spend its slots on papers
        # that verify fine and are then reported BLOCK.
        url_filter=(lambda u: is_allowed(u, hosts)) if hosts else None,
        variants=args.variants,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
        verify_pdfs=not args.dry_run,
    ) as resolver, httpx.AsyncClient(
        timeout=60.0, follow_redirects=True, headers=resolver._client.headers
    ) as client:
        if args.dry_run:
            for description in descriptions:
                try:
                    print_ranking(await resolver.resolve(
                        description, top_k=args.top_k, min_similarity=args.min_similarity))
                except Exception as exc:
                    print(f'\n"{description}"\n  FAILED {type(exc).__name__}: {exc}')
            return 0

        pairs = await asyncio.gather(*(
            process(d, resolver, client, args.out, hosts,
                    args.top_k, args.min_similarity, semaphore)
            for d in descriptions
        ))

    outcomes = [o for _, group in pairs for o in group]

    manifest = args.out / "manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as fh:
        for o in outcomes:
            fh.write(json.dumps(o.__dict__, ensure_ascii=False) + "\n")

    marks = {"downloaded": "OK", "skipped": "HAVE", "unresolved": "MISS",
             "blocked": "BLOCK", "failed": "FAIL"}
    for o in outcomes:
        print(f"{marks[o.status]:<5} {(o.label or o.query)[:78]}")
        if o.paper:
            # Show the pure-similarity position when it differs from the printed
            # rank, so a skipped number is explained rather than looking like a
            # missing result.
            srank = o.paper.get("similarity_rank") or 0
            by_score = f"  (#{srank} by score)" if srank and srank != o.paper["rank"] else ""
            print(f"      sim={o.paper['similarity']:.3f}"
                  f"  {'+'.join(o.paper['retrieval_sources'])}{by_score}")
        if o.detail:
            print(f"      {o.detail[:150]}")

    counts = {s: sum(1 for o in outcomes if o.status == s) for s in marks}
    print("\n" + "  ".join(f"{k}={v}" for k, v in counts.items() if v))
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
