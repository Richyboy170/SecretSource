"""Offline tests. No network, no model download, no Qdrant server.

    .\\.venv\\Scripts\\python.exe test_extractor.py

Same shape as ../source-semantic-tool/test_selection.py: no pytest, no unittest,
a `check` helper and a count at the end. What is covered here is the arithmetic
that turns a score into a claim about a location on a page -- the part that fails
SILENTLY when it is wrong, producing a confident citation of the wrong paragraph
rather than an error.

The only PDF touched is one this file builds in memory with PyMuPDF, so the tests
run in a fresh clone with no corpus.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from page_embedder import PageEmbedder
from attribution import _cells_to_points, _heat_map, _hot_components, attribute
from manifest import AmbiguousQuery, read_manifest, resolve_query
from pdf_pages import (
    lines_in_bbox,
    page_lines,
    page_words,
    pdf_sha256,
    render_pages,
    words_in_bbox,
)
from vector_store import maxsim, point_id_for, rank_locally

results: list[bool] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{'' if condition else '  <- ' + detail}")
    results.append(bool(condition))
    return bool(condition)


def close(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(a - b) <= tol


# -- maxsim ------------------------------------------------------------------- #


def maxsim_tests() -> None:
    print("\nmaxsim")

    # Two query tokens, three page vectors, all axis-aligned so the answer is
    # readable by hand: q0 matches p1 exactly (1.0), q1 matches p2 exactly (1.0).
    query = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    page = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    score, per_token = maxsim(query, page)
    check("score is the sum of per-token maxima", close(score, 2.0), f"got {score}")
    check("argmax per query token", [i for i, _ in per_token] == [1, 2], str(per_token))
    check(
        "contributions match the maxima",
        all(close(c, 1.0) for _, c in per_token),
        str(per_token),
    )

    # Scaling the page vectors scales the score linearly -- the property that
    # makes MaxSim unbounded and NOT a cosine. Guards against anyone "fixing"
    # this by normalizing and quietly changing every score in the corpus.
    score2, _ = maxsim(query, [[v * 3 for v in p] for p in page])
    check("unbounded: scaling the page scales the score", close(score2, 6.0), f"got {score2}")

    check("empty query scores zero", maxsim([], page)[0] == 0.0)
    check("empty page scores zero", maxsim(query, [])[0] == 0.0)

    # Ranking must follow the score.
    pages = [
        ("far", [[0.0, 0.0, 1.0]], {}),
        ("near", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], {}),
    ]
    ranked = rank_locally(query, pages, limit=2)
    check("rank_locally orders by score", [r.point_id for r in ranked] == ["near", "far"],
          str([r.point_id for r in ranked]))


# -- point ids ---------------------------------------------------------------- #


def point_id_tests() -> None:
    print("\npoint ids")
    a = point_id_for("sha-a", 3, "m1")
    check("stable across calls", a == point_id_for("sha-a", 3, "m1"))
    check("differs by page", a != point_id_for("sha-a", 4, "m1"))
    check("differs by document", a != point_id_for("sha-b", 3, "m1"))
    # Two models' vectors live in different spaces and must never share a point.
    check("differs by model", a != point_id_for("sha-a", 3, "m2"))


# -- attribution geometry ------------------------------------------------------ #


def geometry_tests() -> None:
    print("\nattribution geometry")

    # 4 rows x 2 cols over a 100x200 pt page: each cell is 50 x 50 pt.
    # Cell 0 is the TOP-left, because PDF text coordinates grow downward from the
    # top of page.rect and the image rasters the same way. This is the assertion
    # that catches a vertical flip.
    box = _cells_to_points([0], rows=4, cols=2, width_pt=100.0, height_pt=200.0)
    check("cell 0 is the top-left cell", box == (0.0, 0.0, 50.0, 50.0), str(box))

    # Bottom-right cell of the grid is index rows*cols - 1.
    box = _cells_to_points([7], rows=4, cols=2, width_pt=100.0, height_pt=200.0)
    check("last cell is the bottom-right cell", box == (50.0, 150.0, 100.0, 200.0), str(box))

    # Row-major: cell 2 is row 1, col 0 -- NOT column 1. Catches a transpose.
    box = _cells_to_points([2], rows=4, cols=2, width_pt=100.0, height_pt=200.0)
    check("indexing is row-major", box == (0.0, 50.0, 50.0, 100.0), str(box))

    # A blob spans its full extent.
    box = _cells_to_points([0, 1], rows=4, cols=2, width_pt=100.0, height_pt=200.0)
    check("blob bbox covers all its cells", box == (0.0, 0.0, 100.0, 50.0), str(box))


def heat_tests() -> None:
    print("\nattribution heat map")

    # Grid cell -> output-sequence position. Offset by 5 and non-trivial so a
    # code path that forgets to invert the map cannot pass by coincidence.
    index = [5, 6, 7, 8, 9, 10]  # 3 rows x 2 cols
    heat = _heat_map([(9, 2.0), (5, 1.0)], index, rows=3, cols=2)
    check("contributions land in the mapped cell", heat == [1.0, 0, 0, 0, 2.0, 0], str(heat))

    # A query token matching a NON-image vector has no location and is dropped
    # rather than smeared over the grid.
    heat = _heat_map([(99, 5.0), (5, 1.0)], index, rows=3, cols=2)
    check("non-image tokens are excluded", heat == [1.0, 0, 0, 0, 0, 0], str(heat))
    check("all-non-image returns None", _heat_map([(99, 5.0)], index, 3, 2) is None)

    # Component finding runs on a BLURRED copy of the map, so these fixtures use
    # a realistic grid: the real one is 32x24, where a 3x3 blur is local. On a
    # 4x2 toy grid the same blur covers nearly everything and proves nothing.
    ROWS, COLS = 16, 12
    N = ROWS * COLS

    # One hot corner produces one component containing that corner, and the blur
    # keeps it local rather than spreading it across the page.
    hot = [0.1] * N
    hot[0] = 10.0
    components = _hot_components(hot, rows=ROWS, cols=COLS)
    check("hottest component contains the hot cell", 0 in components[0][0], str(components[0][0]))
    check("blur keeps the region local", len(components[0][0]) <= 9, str(len(components[0][0])))

    # Two well-separated hot spots stay separate, so a bbox cannot span the
    # unrelated text between a heading and the figure it refers to.
    split = [0.0] * N
    split[0] = 5.0
    split[N - 1] = 4.0
    components = _hot_components(split, rows=ROWS, cols=COLS)
    check("distant hot spots are separate components", len(components) == 2, str(len(components)))
    check("heaviest component first", 0 in components[0][0], str(components[0][0]))

    # The fix the blur exists for: two tokens landing one cell apart belong to
    # the same phrase and must merge, not come back as two one-cell regions.
    near = [0.0] * N
    near[COLS * 5 + 4] = 3.0
    near[COLS * 5 + 6] = 3.0
    components = _hot_components(near, rows=ROWS, cols=COLS)
    check("adjacent hot cells merge into one region", len(components) == 1, str(len(components)))
    check(
        "merged region spans both cells",
        {COLS * 5 + 4, COLS * 5 + 6} <= set(components[0][0]),
        str(sorted(components[0][0])),
    )

    # Weight comes from the RAW map, so blurring cannot inflate a reported share.
    check("component weight is the raw total", close(components[0][1], 6.0), str(components[0][1]))


def attribute_tests() -> None:
    print("\nattribute()")

    ROWS, COLS = 16, 8  # realistic enough that the 3x3 blur stays local
    N = ROWS * COLS
    payload = {
        "grid_ok": True,
        "grid_rows": ROWS,
        "grid_cols": COLS,
        "image_token_index": list(range(N)),
        "width_pt": 100.0,
        "height_pt": 200.0,
        "page_no": 1,
    }
    # Query token 0 matches the last page vector -- the bottom-right cell -- and
    # nothing else, so the region must land in the bottom-right corner.
    query = [[1.0, 0.0]]
    page = [[0.0, 0.1]] * (N - 1) + [[1.0, 0.0]]
    regions = attribute(page, query, payload, None)
    check("region reported", len(regions) == 1, str(regions))
    if regions:
        x0, y0, x1, y1 = regions[0].bbox_pt
        check("region reaches the bottom-right corner", (x1, y1) == (100.0, 200.0), str(regions[0].bbox_pt))
        check("region stays in the bottom-right quadrant", x0 >= 50.0 and y0 >= 100.0, str(regions[0].bbox_pt))
        check("share is the whole score", close(regions[0].share, 1.0), str(regions[0].share))

    # grid_ok=False means the model's layout could not be reconciled with a grid.
    # Reporting no region is required; inventing one would cite the wrong place.
    check(
        "no region when grid_ok is False",
        attribute(page, query, {**payload, "grid_ok": False}, None) == [],
    )
    check(
        "no region without page geometry",
        attribute(page, query, {**payload, "width_pt": 0}, None) == [],
    )


# -- manifest ------------------------------------------------------------------ #


def tiled_layout_tests() -> None:
    """Reassembly of per-tile patch runs into one page-wide grid.

    Reproduces the layout measured on vidore/colSmol-500M without loading it: a
    2x2 arrangement of tiles, each 2x2 patches, each run preceded by a
    <row_i_col_j> marker, plus a location-free <global-img> run. The real model
    is the same shape at 4x3 tiles of 8x8.
    """
    print("\ntiled layout")

    IMG = 900  # image-token id
    VOCAB = {
        10: "<row_1_col_1>", 11: "<row_1_col_2>",
        12: "<row_2_col_1>", 13: "<row_2_col_2>",
        14: "<global-img>", IMG: "<image>",
    }

    class FakeTokenizer:
        def decode(self, ids):
            return VOCAB.get(ids[0], "<other>")

    class FakeProcessor:
        """Only what _tiled_layout touches: a tokenizer and an image-token id."""

        tokenizer = FakeTokenizer()
        image_token_id = IMG

    class Row(list):
        """Stands in for a torch tensor row, which is what the real batch holds."""

        def tolist(self):
            return list(self)

    def batch_of(ids):
        return {"input_ids": [Row(ids)]}

    ids, expected_tiles = [], {}
    for marker in (10, 11, 12, 13, 14):
        ids.append(marker)
        start = len(ids)
        ids.extend([IMG] * 4)  # 2x2 patches per tile
        expected_tiles[marker] = list(range(start, start + 4))

    layout = PageEmbedder._tiled_layout(FakeProcessor(), batch_of(ids), len(ids))
    check("tiled layout recognized", layout is not None)
    if layout is None:
        return
    rows, cols, index = layout
    check("grid is tiles x patches per side", (rows, cols) == (4, 4), f"{rows}x{cols}")
    check("global view excluded from the grid", len(index) == 16, str(len(index)))
    check(
        "global run is not in the index",
        not set(expected_tiles[14]) & set(index),
        str(index),
    )

    # Cell (0,0) is tile (1,1)'s first patch; cell (0,2) crosses into tile (1,2),
    # NOT tile (2,1). This is the assertion that catches a tile-order transpose,
    # which would place every region in the mirrored quadrant of the page.
    check("top-left cell comes from tile (1,1)", index[0] == expected_tiles[10][0], str(index[:4]))
    check("cell (0,2) comes from tile (1,2)", index[2] == expected_tiles[11][0], str(index[:4]))
    check("cell (2,0) comes from tile (2,1)", index[2 * 4] == expected_tiles[12][0], str(index))
    check("bottom-right cell comes from tile (2,2)", index[-1] == expected_tiles[13][-1], str(index))

    # Ragged tiles cannot form a rectangle -- must decline rather than guess.
    ragged = [10, IMG, IMG, IMG, IMG, 11, IMG, IMG]
    check(
        "ragged tiles are rejected",
        PageEmbedder._tiled_layout(FakeProcessor(), batch_of(ragged), len(ragged)) is None,
    )
    # An incomplete cover (3 of 4 tiles) must also be rejected.
    partial = []
    for marker in (10, 11, 12):
        partial.append(marker)
        partial.extend([IMG] * 4)
    check(
        "incomplete tile cover is rejected",
        PageEmbedder._tiled_layout(FakeProcessor(), batch_of(partial), len(partial)) is None,
    )
    # No markers at all -> not a tiled layout; the flat path handles it.
    flat = [IMG] * 4
    check(
        "untiled sequence returns None",
        PageEmbedder._tiled_layout(FakeProcessor(), batch_of(flat), len(flat)) is None,
    )
    # Output vectors must align 1:1 with input tokens, or positions mean nothing.
    check(
        "misaligned vector count is rejected",
        PageEmbedder._tiled_layout(FakeProcessor(), batch_of(ids), len(ids) + 1) is None,
    )


def manifest_tests() -> None:
    print("\nmanifest")

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        (directory / "alpha_aaaaaaaaaa.pdf").write_bytes(b"%PDF-1.4\n")
        (directory / "beta_bbbbbbbbbb.pdf").write_bytes(b"%PDF-1.4\n")
        (directory / "gone_cccccccccc.pdf").write_bytes(b"%PDF-1.4\n")

        records = [
            # source-semantic-tool shape: has `paper`
            {
                "query": "topic one",
                "status": "downloaded",
                "path": "output_pdf\\alpha_aaaaaaaaaa.pdf",
                "paper": {"title": "Alpha Paper", "doi": "10.1/alpha"},
            },
            # source-tool shape: five keys, no `paper`
            {
                "query": "topic two",
                "status": "downloaded",
                "path": "output_pdf\\beta_bbbbbbbbbb.pdf",
                "detail": None,
                "resolution": {"doi": "10.2/beta"},
            },
            # never downloaded -> no file to index
            {"query": "topic one", "status": "unresolved", "path": None},
            {
                "query": "topic three",
                "status": "failed",
                "path": "output_pdf\\gone_cccccccccc.pdf",
            },
            "{ not json",
        ]
        with open(directory / "manifest.jsonl", "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    (record if isinstance(record, str) else json.dumps(record)) + "\n"
                )

        entries = read_manifest(directory)
        check("only downloaded records kept", len(entries) == 2, str(sorted(e.name for e in entries)))
        check("malformed lines skipped without failing", True)

        alpha = entries.get((directory / "alpha_aaaaaaaaaa.pdf").resolve())
        check("title from paper block", alpha is not None and alpha.title == "Alpha Paper")
        check("doi from paper block", alpha is not None and alpha.doi == "10.1/alpha")
        check("query carried through", alpha is not None and alpha.query == "topic one")

        beta = entries.get((directory / "beta_bbbbbbbbbb.pdf").resolve())
        check(
            "source-tool record falls back to the filename slug",
            beta is not None and beta.title == "beta",
            beta.title if beta else "missing",
        )
        check("doi from resolution block", beta is not None and beta.doi == "10.2/beta")

        # Two distinct queries -> refuse to guess.
        raised = False
        try:
            resolve_query(entries)
        except AmbiguousQuery as error:
            raised = True
            check("ambiguous error lists both", error.queries == ["topic one", "topic two"],
                  str(error.queries))
        check("several descriptions raise instead of guessing", raised)
        check("--query-index picks one", resolve_query(entries, 2) == "topic two")

        out_of_range = False
        try:
            resolve_query(entries, 9)
        except AmbiguousQuery:
            out_of_range = True
        check("out-of-range index raises", out_of_range)

    # A single-description manifest resolves without argument.
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        (directory / "solo_dddddddddd.pdf").write_bytes(b"%PDF-1.4\n")
        with open(directory / "manifest.jsonl", "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "query": "only topic",
                        "status": "downloaded",
                        "path": "output_pdf\\solo_dddddddddd.pdf",
                        "paper": {"title": "Solo", "doi": None},
                    }
                )
                + "\n"
            )
        check("single description resolves", resolve_query(read_manifest(directory)) == "only topic")

    with tempfile.TemporaryDirectory() as tmp:
        check("missing manifest is not an error", read_manifest(Path(tmp)) == {})


# -- pdf ----------------------------------------------------------------------- #


def make_pdf(path: Path, *, rotate: int = 0) -> None:
    """A one-page PDF with a known word near the top-left of the visible page."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=400, height=600)
    page.insert_text((50, 60), "MARKERWORD", fontsize=18)
    page.insert_text((50, 500), "BOTTOMWORD", fontsize=18)
    if rotate:
        page.set_rotation(rotate)
    doc.save(path)
    doc.close()


def column_tests() -> None:
    """Two-column ordering -- the common case in this corpus, not an edge one.

    Sorting hit lines purely by (y, x) zips the two columns together. Observed on
    a real region spanning both columns of the RAG paper: "the output such as the
    answer, the sentence where Conventional Query Expansion. the answer belongs
    to, ..." -- two unrelated sentences interleaved. Geometry is built by hand so
    the fixture IS the interleaved case rather than approximating it.
    """
    print("\ncolumn ordering")

    two_col = [
        (50, 100, 240, 112, "left one"),
        (300, 100, 490, 112, "right one"),
        (50, 120, 240, 132, "left two"),
        (300, 120, 490, 132, "right two"),
    ]
    ordered = lines_in_bbox(two_col, (0, 0, 600, 600))
    check(
        "columns are not interleaved",
        ordered == ["left one", "left two", "right one", "right two"],
        str(ordered),
    )
    # A full-width heading overlaps both columns and must stay one group.
    with_heading = [(50, 80, 490, 92, "heading")] + two_col
    check(
        "a full-width line does not split the page into columns",
        lines_in_bbox(with_heading, (0, 0, 600, 600)) == [
            "heading", "left one", "left two", "right one", "right two",
        ],
        str(lines_in_bbox(with_heading, (0, 0, 600, 600))),
    )
    # Single-column text keeps plain top-to-bottom order.
    one_col = [(50, 120, 490, 132, "second"), (50, 100, 490, 112, "first")]
    check(
        "single column stays in reading order",
        lines_in_bbox(one_col, (0, 0, 600, 600)) == ["first", "second"],
        str(lines_in_bbox(one_col, (0, 0, 600, 600))),
    )


def pdf_tests() -> None:
    print("\npdf rendering and text")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "sample.pdf"
        make_pdf(pdf)

        pages = list(render_pages(pdf, dpi=100))
        check("one page rendered", len(pages) == 1)
        page = pages[0]
        check("page numbering is 1-based", page.page_no == 1)
        check("page geometry from page.rect", (page.width_pt, page.height_pt) == (400.0, 600.0),
              f"{page.width_pt}x{page.height_pt}")
        check(
            "image size follows dpi zoom",
            close(page.image.width, 400 * 100 / 72, tol=2.0),
            str(page.image.size),
        )

        words = page_words(pdf, 1)
        markers = [w for w in words if w[4] == "MARKERWORD"]
        check("text extracted", len(markers) == 1, str([w[4] for w in words]))
        if markers:
            x0, y0, x1, y1 = markers[0][:4]
            check("word sits in the top half", y1 < 300, f"y1={y1}")
            # The bbox filter must find it, and must not find the bottom word.
            found = words_in_bbox(words, (0, 0, 400, 300))
            check("words_in_bbox finds the top word", "MARKERWORD" in found, str(found))
            check("words_in_bbox excludes the bottom word", "BOTTOMWORD" not in found, str(found))

        # Lines, not words, are what a snippet is built from -- a grid cell is
        # only a few characters wide, so word-level filtering reads as fragments.
        lines = page_lines(pdf, 1)
        check("lines extracted", len(lines) == 2, str([l[4] for l in lines]))
        # A box clipping only part of the top line must still return it whole.
        clipped = lines_in_bbox(lines, (55, 45, 70, 55))
        check("a clipped line comes back whole", clipped == ["MARKERWORD"], str(clipped))
        check(
            "a box over both lines returns both in reading order",
            lines_in_bbox(lines, (0, 0, 400, 600)) == ["MARKERWORD", "BOTTOMWORD"],
            str(lines_in_bbox(lines, (0, 0, 400, 600))),
        )
        check("a box touching nothing returns nothing", lines_in_bbox(lines, (300, 200, 390, 300)) == [])

        check("max_pages limits rendering", len(list(render_pages(pdf, max_pages=0))) == 0)
        check("sha256 is stable", pdf_sha256(pdf) == pdf_sha256(pdf))
        check("page_words on a bad page number returns empty", page_words(pdf, 99) == [])

    # Rotated page: page.rect must report the ROTATED dimensions, and the words
    # must land inside them. Landscape figure/table pages rotate like this and
    # are exactly what a visual retriever surfaces, so this is not a rare tail.
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "rotated.pdf"
        make_pdf(pdf, rotate=90)
        page = next(iter(render_pages(pdf, dpi=100)))
        check(
            "rotated page reports swapped dimensions",
            (page.width_pt, page.height_pt) == (600.0, 400.0),
            f"{page.width_pt}x{page.height_pt}",
        )
        check(
            "rendered image is landscape too",
            page.image.width > page.image.height,
            str(page.image.size),
        )
        words = page_words(pdf, 1)
        inside = [
            w for w in words
            if 0 <= w[0] and w[2] <= page.width_pt and 0 <= w[1] and w[3] <= page.height_pt
        ]
        check(
            "word coordinates lie within the rotated page.rect",
            len(words) > 0 and len(inside) == len(words),
            f"{len(inside)}/{len(words)} inside {page.width_pt}x{page.height_pt}",
        )


def main() -> int:
    maxsim_tests()
    point_id_tests()
    geometry_tests()
    heat_tests()
    attribute_tests()
    tiled_layout_tests()
    manifest_tests()
    column_tests()
    pdf_tests()
    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
