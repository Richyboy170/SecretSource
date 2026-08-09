"""Index research PDFs as ColPali page embeddings, then rank pages by description.

    # index a folder the source tools filled
    extract_embeddings.py index --pdf-dir ../source-semantic-tool/output_pdf

    # rank pages against a description
    extract_embeddings.py search "retrieval augmented generation for open-domain QA"

    # reuse the description the source tool already recorded
    extract_embeddings.py search --pdf-dir ../source-semantic-tool/output_pdf

    # confirm the model's token layout before trusting a reported region
    extract_embeddings.py probe some.pdf

This is the pipeline's Retrieval node: it searches INSIDE documents already
acquired, where ../source-semantic-tool finds which documents to acquire.

Scores are MaxSim sums, not cosines -- unbounded, and comparable only within one
query. There is no threshold; the ranking is the answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from manifest import AmbiguousQuery, ManifestEntry, read_manifest, resolve_query
from page_embedder import DEFAULT_MODEL, MODELS

DEFAULT_TOP_K = 10
DEFAULT_QDRANT_URL = "http://localhost:6333"


# -- shared ------------------------------------------------------------------ #


def collect_pdfs(paths: list[Path], pdf_dir: Path | None) -> list[Path]:
    """PDFs named directly plus every PDF in `--pdf-dir`, deduped and sorted."""
    found: list[Path] = []
    for path in paths:
        if path.is_file():
            found.append(path.resolve())
        else:
            print(f"skip: not a file: {path}", file=sys.stderr)
    if pdf_dir:
        if not pdf_dir.is_dir():
            print(f"skip: not a directory: {pdf_dir}", file=sys.stderr)
        else:
            found.extend(p.resolve() for p in sorted(pdf_dir.glob("*.pdf")))
    return sorted(set(found))


class StoreUnreachable(Exception):
    """Qdrant is not answering, with an explanation a reader can act on."""


def open_store(args):
    """Connect, honouring --qdrant-path (embedded) over --qdrant-url (server).

    The server default points at localhost, and there may well be nothing there:
    Qdrant normally runs in Docker, which needs WSL2 on Windows 11 Home and is
    not installed on every machine this tool runs on. Probing once turns a raw
    connection traceback -- thrown from deep inside collection_exists, after the
    user has already typed a long command -- into a sentence naming the two ways
    out. The probe costs one round trip against a local server.
    """
    from vector_store import connect

    if args.qdrant_path:
        return connect(path=str(args.qdrant_path))

    client = connect(url=args.qdrant_url)
    try:
        client.get_collections()
    except Exception as error:
        raise StoreUnreachable(
            f"cannot reach Qdrant at {args.qdrant_url}: "
            f"{type(error).__name__}: {error}\n"
            "Either start the server (see README: docker run -p 6333:6333 qdrant/qdrant)\n"
            "or use the embedded store instead, which needs no server:\n"
            "    --qdrant-path ./qdrant_data"
        ) from error
    return client


def report_ambiguous(error: AmbiguousQuery) -> int:
    """Print the choices instead of guessing which description was meant."""
    if not error.queries:
        print(
            "no description found in the manifest; pass one as an argument",
            file=sys.stderr,
        )
        return 2
    print(
        f"manifest holds {len(error.queries)} descriptions -- pass one explicitly, "
        f"or --query-index N:",
        file=sys.stderr,
    )
    for i, query in enumerate(error.queries, 1):
        print(f"  {i}. {query}", file=sys.stderr)
    return 2


# -- index ------------------------------------------------------------------- #


def cmd_index(args) -> int:
    from pdf_pages import pdf_sha256, render_pages
    from page_embedder import PageEmbedder
    from vector_store import ensure_collection, indexed_pages, point_id_for, upsert_page

    pdfs = collect_pdfs(args.pdfs, args.pdf_dir)
    if not pdfs:
        print("nothing to index", file=sys.stderr)
        return 2

    # Titles and DOIs live in the source tool's manifest, not in the PDF. Read
    # every parent directory once so a mixed --pdfs list still gets enriched.
    entries: dict[Path, ManifestEntry] = {}
    for directory in {p.parent for p in pdfs}:
        entries.update(read_manifest(directory))

    embedder = PageEmbedder(args.model)
    client = open_store(args)
    ensure_collection(client)

    total_pages = 0
    total_bytes = 0
    for pdf in pdfs:
        sha = pdf_sha256(pdf)
        done = set() if args.reindex else indexed_pages(client, sha, args.model)
        entry = entries.get(pdf)
        label = entry.title if entry else pdf.stem
        print(f"\n{pdf.name}\n  {label[:100]}")

        indexed = skipped = 0
        for page in render_pages(pdf, dpi=args.dpi, max_pages=args.max_pages):
            if page.page_no in done:
                skipped += 1
                continue
            embedding = embedder.embed_page(page.image)
            payload = {
                "pdf_path": str(pdf),
                "pdf_name": pdf.name,
                "pdf_sha": sha,
                "page_no": page.page_no,
                "page_count": page.page_count,
                "width_pt": page.width_pt,
                "height_pt": page.height_pt,
                "grid_rows": embedding.grid_rows,
                "grid_cols": embedding.grid_cols,
                "image_token_index": embedding.image_token_index,
                "grid_ok": embedding.grid_ok,
                "model": args.model,
            }
            if entry:
                payload.update(entry.payload())
            upsert_page(
                client,
                point_id=point_id_for(sha, page.page_no, args.model),
                vectors=embedding.vectors,
                payload=payload,
            )
            indexed += 1
            total_pages += 1
            total_bytes += embedding.n_vectors * 128 * 4
            print(
                f"  page {page.page_no}/{page.page_count}: "
                f"{embedding.n_vectors} vectors"
                f"{'' if embedding.grid_ok else '  [no grid - region attribution off]'}"
            )
        print(f"  indexed {indexed}, skipped {skipped} already present")

    print(f"\n{total_pages} pages indexed, ~{total_bytes / 1e6:.1f} MB of vectors")
    if total_pages:
        print(f"average {total_bytes / total_pages / 1024:.0f} KB/page")
    return 0


# -- search ------------------------------------------------------------------ #


def cmd_search(args) -> int:
    from attribution import attribute
    from page_embedder import PageEmbedder
    from vector_store import search

    description = " ".join(args.description).strip()
    if not description:
        directory = args.pdf_dir or args.from_manifest
        if not directory:
            print(
                "give a description, or --pdf-dir/--from-manifest to reuse the one "
                "the source tool recorded",
                file=sys.stderr,
            )
            return 2
        # Accept either the folder or the manifest file itself -- "--from-manifest
        # output_pdf/manifest.jsonl" is the more natural thing to type.
        directory = Path(directory)
        if directory.is_file():
            directory = directory.parent
        entries = read_manifest(directory)
        try:
            description = resolve_query(entries, args.query_index)
        except AmbiguousQuery as error:
            return report_ambiguous(error)
        print(f"description from manifest: {description!r}\n")

    embedder = PageEmbedder(args.model)
    query_vectors = embedder.embed_query(description)
    client = open_store(args)

    # with_vectors: attribution needs each hit's patch matrix to recover which
    # patch won each query token. Paid only for the pages actually returned.
    hits = search(client, query_vectors, limit=args.top_k, with_vectors=True)
    if not hits:
        print("no pages indexed for this store", file=sys.stderr)
        return 1

    results = []
    for rank, hit in enumerate(hits, 1):
        pdf_path = Path(str(hit.payload.get("pdf_path") or ""))
        regions = (
            attribute(
                hit.vectors,
                query_vectors,
                hit.payload,
                pdf_path if pdf_path.is_file() else None,
                top_regions=args.regions,
            )
            if hit.vectors
            else []
        )
        results.append((rank, hit, regions))

    if args.json:
        # Same envelope as search_pages(), so a consumer can move between the CLI
        # and the agent tool without reshaping anything.
        print(
            json.dumps(
                {
                    "description": description,
                    "model": args.model,
                    "pages": [_as_record(r, h, g) for r, h, g in results],
                },
                indent=2,
            )
        )
    else:
        _print_ranking(results, description)
    return 0


def _as_record(rank: int, hit, regions) -> dict:
    return {
        "rank": rank,
        "score": round(hit.score, 4),
        "pdf_name": hit.pdf_name,
        "pdf_path": hit.payload.get("pdf_path"),
        "title": hit.payload.get("title"),
        "doi": hit.payload.get("doi"),
        "page_no": hit.page_no,
        "page_count": hit.payload.get("page_count"),
        "regions": [
            {
                "bbox_pt": [round(v, 1) for v in region.bbox_pt],
                "share": round(region.share, 4),
                "text": region.text,
            }
            for region in regions
        ],
    }


def _print_ranking(results, description: str) -> None:
    print(f'ranking pages against: "{description}"')
    print("scores are MaxSim sums, not cosines - comparable only within this query\n")
    for rank, hit, regions in results:
        title = hit.payload.get("title") or hit.pdf_name
        print(f"RANK {rank}  score {hit.score:.2f}")
        print(f"  paper: {title}")
        print(f"  file:  {hit.pdf_name}")
        print(f"  page {hit.page_no} of {hit.payload.get('page_count', '?')}")
        if not regions:
            if not hit.payload.get("grid_ok"):
                print("  (no region: this model's token layout has no single grid)")
        for region in regions:
            print(f"  top region: {region.bbox_str()}  ({region.share:.0%} of score)")
            if region.text:
                print("  text near region:")
                for line in _wrap(region.text, 72):
                    print(f"    {line}")
        print()


def _wrap(text: str, width: int) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


# -- probe ------------------------------------------------------------------- #


def cmd_probe(args) -> int:
    """Report the model's token layout for one page. See README."""
    from pdf_pages import render_pages
    from page_embedder import PageEmbedder

    pages = list(render_pages(args.pdf, dpi=args.dpi, max_pages=1))
    if not pages:
        print(f"no pages in {args.pdf}", file=sys.stderr)
        return 1

    info = PageEmbedder(args.model).describe_layout(pages[0].image)
    print(json.dumps(info, indent=2))
    if info["grid_ok"]:
        page = pages[0]
        cell_w = page.width_pt / info["grid_cols"]
        cell_h = page.height_pt / info["grid_rows"]
        print(
            f"\nOK: region attribution is valid. "
            f"{info['grid_rows']}x{info['grid_cols']} grid "
            f"({info['grid_cells']} cells) over a "
            f"{page.width_pt:.0f}x{page.height_pt:.0f} pt page "
            f"= {cell_w:.1f}x{cell_h:.1f} pt per cell.\n"
            f"{info['n_vectors']} vectors/page, "
            f"{info['bytes_per_page'] / 1024:.0f} KB/page stored."
        )
    else:
        print(
            f"\nNO GRID: {info['n_vectors']} vectors could not be reconciled with "
            f"any rows x cols grid"
            f"{'' if info['grid_contiguous'] else ' (image tokens are not contiguous'}"
            f"{'' if info['grid_contiguous'] else ', and no tile markers were found)'}."
            "\nRegion attribution is disabled for this model; page ranking is unaffected.",
            file=sys.stderr,
        )
    return 0


# -- cli --------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed research-paper pages with ColPali and rank them by description."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
        p.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
        p.add_argument(
            "--qdrant-path",
            type=Path,
            default=None,
            help="embedded store directory; use instead of a server",
        )

    index = sub.add_parser("index", help="embed PDF pages into Qdrant")
    index.add_argument("pdfs", nargs="*", type=Path, default=[])
    index.add_argument("--pdf-dir", type=Path, default=None)
    index.add_argument("--dpi", type=int, default=150)
    index.add_argument("--max-pages", type=int, default=None)
    index.add_argument(
        "--reindex", action="store_true", help="re-embed pages already stored"
    )
    add_common(index)
    index.set_defaults(func=cmd_index)

    search_p = sub.add_parser("search", help="rank indexed pages against a description")
    search_p.add_argument("description", nargs="*", default=[])
    search_p.add_argument(
        "--pdf-dir", type=Path, default=None, help="read the description from its manifest"
    )
    search_p.add_argument("--from-manifest", type=Path, default=None)
    search_p.add_argument(
        "--query-index",
        type=int,
        default=None,
        help="1-based pick when the manifest holds several descriptions",
    )
    search_p.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K)
    search_p.add_argument("--regions", type=int, default=1)
    search_p.add_argument("--json", action="store_true")
    add_common(search_p)
    search_p.set_defaults(func=cmd_search)

    probe = sub.add_parser("probe", help="report the model's token layout for one page")
    probe.add_argument("pdf", type=Path)
    probe.add_argument("--dpi", type=int, default=150)
    probe.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
    probe.set_defaults(func=cmd_probe)

    return parser


def _force_utf8_output() -> None:
    """Stop the Windows console encoding from killing a successful search.

    Python on Windows defaults stdout to cp1252, which cannot encode most of what
    a research paper contains: the author-footnote star U+2217, dashes, ligatures,
    Greek in equations. Printing a snippet that holds one raises
    UnicodeEncodeError *after* the model has run and the ranking is in hand --
    observed on the first real query, where the title line of the top hit ended
    the process with a traceback.

    `errors="replace"` rather than strict, because a character that will not
    survive the terminal should degrade to a "?" and not lose the result.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass  # redirected to something that cannot be reconfigured; fine


def main() -> int:
    _force_utf8_output()
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except StoreUnreachable as error:
        print(error, file=sys.stderr)
        return 2


# -- agent tool surface ------------------------------------------------------- #

SEARCH_PAGES_TOOL = {
    "name": "search_paper_pages",
    "description": (
        "Rank pages of already-downloaded research PDFs against a free-text "
        "description, using ColPali visual page embeddings. Returns the "
        "best-matching pages with the region of each page that carried the match "
        "and the text found there. Scores are MaxSim sums, not cosines: they are "
        "comparable only within one query and have no meaningful threshold."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "What the evidence should be about, in plain language",
            },
            "top_k": {"type": "integer", "description": "Max pages (default 10)"},
        },
        "required": ["description"],
    },
}


def search_pages(
    description: str,
    top_k: int = DEFAULT_TOP_K,
    *,
    model: str = DEFAULT_MODEL,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    qdrant_path: str | None = None,
) -> dict:
    """Programmatic form of `search`, for the stage-5 verifier agents."""
    from attribution import attribute
    from page_embedder import PageEmbedder
    from vector_store import connect, search

    embedder = PageEmbedder(model)
    query_vectors = embedder.embed_query(description)
    client = (
        connect(path=qdrant_path) if qdrant_path else connect(url=qdrant_url)
    )
    hits = search(client, query_vectors, limit=top_k, with_vectors=True)

    records = []
    for rank, hit in enumerate(hits, 1):
        pdf_path = Path(str(hit.payload.get("pdf_path") or ""))
        regions = (
            attribute(
                hit.vectors,
                query_vectors,
                hit.payload,
                pdf_path if pdf_path.is_file() else None,
            )
            if hit.vectors
            else []
        )
        records.append(_as_record(rank, hit, regions))
    return {"description": description, "model": model, "pages": records}


if __name__ == "__main__":
    raise SystemExit(main())
