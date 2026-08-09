"""Resolve paper titles to PDFs, accepting downloads only from allowed sites.

    py fetch_papers.py "Attention Is All You Need"
    py fetch_papers.py "Deep learning" --doi 10.1038/nature14539
    py fetch_papers.py --from-file titles.txt --workers 6
    py fetch_papers.py "..." --sources my_sources.txt --out pdfs

The allowlist lives in sources.txt — edit that file to declare which sites you
trust. A resolved PDF hosted anywhere else is reported as "blocked" and never
written to disk.
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

from paper_resolver import PaperResolver, Resolution, pick_best

MIN_PDF_BYTES = 1024  # anything smaller is an error page, not a paper


@dataclass
class Outcome:
    query: str
    status: str  # downloaded | skipped | unresolved | blocked | failed
    path: str | None = None
    detail: str | None = None
    resolution: dict | None = None


# --------------------------------------------------------------------------- #
# Allowlist
# --------------------------------------------------------------------------- #

def load_sources(path: Path) -> set[str]:
    """Read sources.txt into a set of bare hosts. Accepts full URLs, bare
    domains, or www-prefixed forms and normalizes all three to the same key."""
    hosts = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
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

def parse_titles(path: Path) -> list[tuple[str | None, str | None]]:
    """Optional batch input: one paper per line, as `Title`, `doi:10.x/y`, or
    `Title | 10.x/y`."""
    entries = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            title, _, doi = line.partition("|")
            entries.append((title.strip() or None, doi.strip() or None))
        elif line.lower().startswith("doi:") or re.fullmatch(r"10\.\d{4,9}/\S+", line):
            entries.append((None, re.sub(r"(?i)^doi:", "", line).strip()))
        else:
            entries.append((line, None))
    return entries


def slugify(resolution: Resolution) -> str:
    """Stable filename from the matched title, suffixed with the matched copy's own
    identity so two papers with similar truncated titles cannot overwrite each other."""
    best = resolution.best
    basis = (best.title if best else None) or resolution.query_title or "paper"
    slug = re.sub(r"[^a-z0-9]+", "-", basis.lower()).strip("-")[:80] or "paper"

    # Only what the match actually proved may name the file. Falling back to the
    # *requested* DOI stamped a borrowed identity onto content that never matched
    # it, and the filename is the surface a human reads.
    identity = best.identity if best else None
    if identity:
        # Keep the head, not the tail: a DOI is distinctive from its registrant
        # prefix onwards, and slicing from the end cuts the "10." off the front.
        slug += "_" + re.sub(r"[^a-z0-9]+", "-", identity.lower()).strip("-")[:40]
    elif best and best.pdf_url:
        # Nothing proved an identity, so disambiguate on the URL we fetched rather
        # than asserting a DOI this copy never established.
        slug += "_" + hashlib.sha1(best.pdf_url.encode("utf-8")).hexdigest()[:8]
    return slug + ".pdf"


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #

async def download(client: httpx.AsyncClient, url: str, dest: Path, hosts: set[str] | None) -> None:
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


async def process(entry: tuple[str | None, str | None], resolver: PaperResolver,
                  client: httpx.AsyncClient, out_dir: Path, hosts: set[str] | None,
                  semaphore: asyncio.Semaphore) -> Outcome:
    title, doi = entry
    label = title or f"doi:{doi}"
    async with semaphore:
        try:
            resolution = await resolver.resolve(title=title, doi=doi)
        except Exception as exc:
            return Outcome(label, "failed", detail=f"{type(exc).__name__}: {exc}")

        # Prefer any candidate hosted on an allowed site over the top-ranked one.
        if hosts:
            allowed = [c for c in resolution.candidates if c.pdf_url and is_allowed(c.pdf_url, hosts)]
            if not allowed:
                seen = {httpx.URL(c.pdf_url).host for c in resolution.candidates if c.pdf_url}
                detail = f"no allowed host among: {', '.join(sorted(h for h in seen if h))}" if seen \
                    else "no open-access PDF found"
                return Outcome(label, "blocked" if seen else "unresolved",
                               detail=detail, resolution=resolution.as_dict())
            # Pick by the same rule the resolver uses — identity first, fetchability
            # only as a tie-break — rather than taking allowed[0] and letting a
            # verified weaker match outrank a stronger one.
            chosen = pick_best(allowed)
            # Re-rank rather than just reassigning best, or as_dict() reports best
            # twice and hides the top-ranked blocked candidate. Blocked hits stay in
            # the list as provenance, now in their true position.
            resolution.candidates = ([chosen] + [c for c in allowed if c is not chosen]
                                     + [c for c in resolution.candidates if c not in allowed])
            resolution.best = chosen

        if not resolution.pdf_url:
            detail = "; ".join(f"{k}={v}" for k, v in resolution.errors.items()) or "no open-access PDF found"
            return Outcome(label, "unresolved", detail=detail, resolution=resolution.as_dict())

        dest = out_dir / slugify(resolution)
        if dest.exists():
            return Outcome(label, "skipped", path=str(dest), resolution=resolution.as_dict())

        try:
            await download(client, resolution.pdf_url, dest, hosts)
        except Exception as exc:
            return Outcome(label, "failed", detail=f"{type(exc).__name__}: {exc}",
                           resolution=resolution.as_dict())

        return Outcome(label, "downloaded", path=str(dest), resolution=resolution.as_dict())


# --------------------------------------------------------------------------- #

async def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve paper titles to PDFs from allowed sites.")
    parser.add_argument("title", nargs="*", help="one or more paper titles")
    parser.add_argument("--doi", help="DOI for the single title given (exact-identity resolution)")
    parser.add_argument("--from-file", type=Path, help="batch input: one paper per line")
    parser.add_argument("--sources", default="sources.txt", type=Path, help="allowed-site list")
    parser.add_argument("--out", default="output_pdf", type=Path)
    parser.add_argument("--workers", type=int, default=4, help="concurrent papers in flight")
    parser.add_argument("--any-host", action="store_true", help="ignore sources.txt and accept any host")
    args = parser.parse_args()

    entries: list[tuple[str | None, str | None]] = []
    if args.from_file:
        if not args.from_file.exists():
            print(f"input file not found: {args.from_file}", file=sys.stderr)
            return 1
        entries += parse_titles(args.from_file)
    if args.title:
        entries += [(t, args.doi if len(args.title) == 1 else None) for t in args.title]
    elif args.doi and not args.from_file:
        entries.append((None, args.doi))

    if not entries:
        parser.print_usage(sys.stderr)
        print("\ngive a title, a --doi, or --from-file", file=sys.stderr)
        return 1

    hosts = None
    if not args.any_host:
        if not args.sources.exists():
            print(f"allowlist not found: {args.sources} (use --any-host to skip)", file=sys.stderr)
            return 1
        hosts = load_sources(args.sources)
        if not hosts:
            print(f"{args.sources} lists no sites", file=sys.stderr)
            return 1

    args.out.mkdir(parents=True, exist_ok=True)
    scope = f"{len(hosts)} allowed sites" if hosts else "any host"
    print(f"resolving {len(entries)} paper(s) from {scope} -> {args.out}/\n")

    semaphore = asyncio.Semaphore(args.workers)
    async with PaperResolver() as resolver, httpx.AsyncClient(
        timeout=60.0, follow_redirects=True, headers=resolver._client.headers
    ) as client:
        outcomes = await asyncio.gather(
            *(process(e, resolver, client, args.out, hosts, semaphore) for e in entries)
        )

    manifest = args.out / "manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as fh:
        for o in outcomes:
            fh.write(json.dumps(o.__dict__, ensure_ascii=False) + "\n")

    marks = {"downloaded": "OK", "skipped": "HAVE", "unresolved": "MISS",
             "blocked": "BLOCK", "failed": "FAIL"}
    for o in outcomes:
        print(f"{marks[o.status]:<5} {o.query[:70]}")
        if o.detail:
            print(f"      {o.detail[:150]}")

    counts = {s: sum(1 for o in outcomes if o.status == s) for s in marks}
    print("\n" + "  ".join(f"{k}={v}" for k, v in counts.items() if v))
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
