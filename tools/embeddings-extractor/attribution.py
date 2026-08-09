"""Answer "where on this page did it match?" from a page's MaxSim breakdown.

Qdrant returns a page's score but not the arithmetic behind it. MaxSim, though,
is a sum of per-query-token maxima, so once the page's patch vectors are back in
hand the winning patch for each query token can be recovered exactly -- and a
patch is a cell of a grid laid over the page image, which maps to a rectangle in
PDF points, which maps to the words printed there.

    regions = attribute(page_vectors, query_vectors, payload, pdf_path)
    regions[0].bbox_pt, regions[0].share, regions[0].text

`share` is a fraction of the page's own score, so it says which part of THIS page
carried the match. It is attribution, not confidence: a page that barely matched
still has a highest-contributing region.

When the layout could not be reconciled with a single grid (see
page_embedder.PageEmbedder._layout), this returns no regions. Reporting the page
alone is a smaller loss than pointing at the wrong paragraph.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from pdf_pages import lines_in_bbox, page_lines
from vector_store import maxsim

# Cells count as "hot" at mean + HOT_SIGMA standard deviations of the page's own
# contribution map. Relative rather than absolute because MaxSim has no fixed
# scale; 0.5 keeps a genuine hot spot together without swallowing half the page.
HOT_SIGMA = 0.5

# Box-blur radius applied before thresholding, in grid cells. 1 (a 3x3 window)
# merges tokens that landed within a cell of each other -- roughly a line of text
# on A4 -- without bridging separate regions on opposite sides of the page.
SMOOTH_RADIUS = 1

# Snippets are for a human confirming the hit and for a verifier agent's context
# window. Past a few hundred characters both are better served by opening the PDF.
MAX_SNIPPET_CHARS = 300


@dataclass
class Region:
    """A rectangle on a page, how much of the score it carried, and its text."""

    bbox_pt: tuple[float, float, float, float]  # x0, y0, x1, y1 in PDF points
    share: float  # fraction of the page's MaxSim score, 0-1
    text: str
    cells: int  # grid cells merged into this region, for debugging

    def bbox_str(self) -> str:
        x0, y0, x1, y1 = self.bbox_pt
        return f"x0={x0:.0f} y0={y0:.0f} x1={x1:.0f} y1={y1:.0f}"


def attribute(
    page_vectors: Sequence[Sequence[float]],
    query_vectors: Sequence[Sequence[float]],
    payload: dict[str, Any],
    pdf_path: Path | None = None,
    *,
    top_regions: int = 1,
) -> list[Region]:
    """Highest-contributing regions of a page, best first."""
    rows = int(payload.get("grid_rows") or 0)
    cols = int(payload.get("grid_cols") or 0)
    index = list(payload.get("image_token_index") or [])
    if not payload.get("grid_ok") or rows <= 0 or cols <= 0 or not index:
        return []

    score, per_token = maxsim(query_vectors, page_vectors)
    if score <= 0 or not per_token:
        return []

    heat = _heat_map(per_token, index, rows, cols)
    if heat is None:
        return []

    width_pt = float(payload.get("width_pt") or 0.0)
    height_pt = float(payload.get("height_pt") or 0.0)
    if width_pt <= 0 or height_pt <= 0:
        return []

    lines = (
        page_lines(pdf_path, int(payload.get("page_no") or 0)) if pdf_path else []
    )

    regions: list[Region] = []
    for component, weight in _hot_components(heat, rows, cols)[:top_regions]:
        bbox = _cells_to_points(component, rows, cols, width_pt, height_pt)
        snippet = " ".join(lines_in_bbox(lines, bbox)) if lines else ""
        if len(snippet) > MAX_SNIPPET_CHARS:
            snippet = snippet[:MAX_SNIPPET_CHARS].rstrip() + "..."
        regions.append(
            Region(
                bbox_pt=bbox,
                share=weight / score if score else 0.0,
                text=snippet,
                cells=len(component),
            )
        )
    return regions


def _heat_map(
    per_token: list[tuple[int, float]],
    image_token_index: list[int],
    rows: int,
    cols: int,
) -> list[float] | None:
    """Sum each query token's contribution into the grid cell that won it.

    `image_token_index` is the map from grid cell (row-major) to position in the
    output vector sequence, so it has to be inverted. Query tokens whose best
    match is NOT an image patch -- a separator or an instruction token -- are
    dropped: they carry real score but no location, and spreading them over the
    grid would blur every region toward the page centre.
    """
    position_to_cell = {pos: cell for cell, pos in enumerate(image_token_index)}
    heat = [0.0] * (rows * cols)
    hit = False
    for position, contribution in per_token:
        cell = position_to_cell.get(position)
        if cell is None or contribution <= 0:
            continue
        heat[cell] += contribution
        hit = True
    return heat if hit else None


def _smooth(heat: list[float], rows: int, cols: int, radius: int = SMOOTH_RADIUS) -> list[float]:
    """Box-blur the heat map so neighbouring hot cells reinforce each other.

    Without this, regions come out as single cells. A query contributes at most
    one cell per token -- perhaps 18 of them -- scattered over a 768-cell grid,
    so 4-connected components on the raw map are almost all singletons. Measured
    on the RAG query against its own paper (2026-08-08), every region came back
    one cell wide and the snippets read "gen- generation-".

    Blurring first lets tokens that landed near each other merge into one blob,
    which is what a phrase matching a line of text actually looks like. The
    weight of a component is still summed from the RAW map, so widening the
    search for a region does not inflate its reported share.
    """
    out = [0.0] * (rows * cols)
    for r in range(rows):
        for c in range(cols):
            total, n = 0.0, 0
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        total += heat[rr * cols + cc]
                        n += 1
            out[r * cols + c] = total / n
    return out


def _hot_components(
    heat: list[float], rows: int, cols: int
) -> list[tuple[list[int], float]]:
    """Group above-threshold cells into 4-connected blobs, heaviest blob first.

    Connected components rather than one bounding box over every hot cell: a
    query often matches two separate places on a page (a heading and the figure
    it refers to), and a single box spanning both would cover the unrelated text
    between them and pull it into the snippet.
    """
    if not any(v > 0 for v in heat):
        return []
    raw = heat
    heat = _smooth(heat, rows, cols)
    # Statistics over EVERY cell, including the zeros. A heat map is sparse --
    # only as many cells are non-zero as there are query tokens -- so taking the
    # mean over just the non-zero ones makes the threshold track the hot cells
    # themselves and cuts the weaker of two genuine hot spots. Measured on a
    # two-spot map of 5.0 and 4.0 over 8 cells: non-zero-only gives a threshold
    # of 4.75 and loses the 4.0 spot; over all cells it is 2.11 and keeps both.
    # The zeros are real information -- those cells matched nothing.
    mean = sum(heat) / len(heat)
    variance = sum((v - mean) ** 2 for v in heat) / len(heat)
    threshold = mean + HOT_SIGMA * (variance ** 0.5)

    hot = {i for i, v in enumerate(heat) if v >= threshold and v > 0}
    if not hot:
        hot = {max(range(len(heat)), key=lambda i: heat[i])}

    components: list[tuple[list[int], float]] = []
    unvisited = set(hot)
    while unvisited:
        stack = [unvisited.pop()]
        blob = []
        while stack:
            cell = stack.pop()
            blob.append(cell)
            row, col = divmod(cell, cols)
            for nrow, ncol in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if 0 <= nrow < rows and 0 <= ncol < cols:
                    neighbour = nrow * cols + ncol
                    if neighbour in unvisited:
                        unvisited.discard(neighbour)
                        stack.append(neighbour)
        # Weight from the RAW map: the blur decides WHERE a region is, but the
        # share it reports must be score the page actually earned there.
        components.append((blob, sum(raw[c] for c in blob)))

    components.sort(key=lambda item: item[1], reverse=True)
    return components


def _cells_to_points(
    cells: list[int], rows: int, cols: int, width_pt: float, height_pt: float
) -> tuple[float, float, float, float]:
    """Bounding rectangle of a set of grid cells, in PDF points.

    Row-major indexing, and rows run down the page: cell 0 is the TOP-left,
    matching both the image raster order and PDF text coordinates, which also
    grow downward from the top of `page.rect`. Getting this backwards flips every
    box vertically and is the single most likely silent bug here, so the tests
    pin a known corner.
    """
    positions = [divmod(cell, cols) for cell in cells]
    row0 = min(r for r, _ in positions)
    row1 = max(r for r, _ in positions) + 1
    col0 = min(c for _, c in positions)
    col1 = max(c for _, c in positions) + 1

    cell_w = width_pt / cols
    cell_h = height_pt / rows
    return (col0 * cell_w, row0 * cell_h, col1 * cell_w, row1 * cell_h)
