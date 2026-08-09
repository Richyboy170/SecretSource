"""Turn a PDF into page images and page text, both in the same coordinate space.

    from pdf_pages import render_pages, page_words
    for page in render_pages(Path("paper.pdf"), dpi=150):
        print(page.page_no, page.image.size, page.width_pt, page.height_pt)

PyMuPDF rather than pdf2image: pdf2image shells out to poppler's `pdftoppm`,
which is not on PATH on this machine and has no pip-installable Windows build.
PyMuPDF is a self-contained abi3 wheel, so `pip install` is the whole setup.

The one thing this module exists to get right is that the rendered image and the
extracted words describe the SAME rectangle -- attribution.py maps a patch of the
image back to a box on the page and then asks what text is in that box, so a
mismatch there produces confidently-wrong citations rather than an error.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:  # keeps PIL out of the import path for callers that never render
    from PIL import Image

# 150 gives ~1240x1750 px for A4. ColPali-family processors downscale to their
# own resolution anyway, so more DPI buys nothing for the embedding -- it only
# costs render time and RAM. Kept configurable because the rendered PNG is also
# what a human looks at when checking a reported region by hand.
DEFAULT_DPI = 150

POINTS_PER_INCH = 72.0


@dataclass(frozen=True)
class PageImage:
    """One rendered page, plus the geometry needed to map pixels back to points.

    `width_pt`/`height_pt` come from `page.rect`, NOT from the mediabox: `.rect`
    is the cropped, rotation-applied rectangle, and it is the space
    `page.get_text("words")` reports coordinates in. Using the mediabox instead
    silently offsets every box on a cropped page and transposes every box on a
    rotated one -- and rotated landscape pages are exactly the wide figure and
    table pages this tool is best at retrieving, so they show up near the top of
    results rather than in some rare tail.
    """

    pdf_path: Path
    page_no: int  # 1-based, matching what a PDF reader shows
    page_count: int
    image: "Image.Image"
    width_pt: float
    height_pt: float
    zoom: float  # pixels per point; image.width == round(width_pt * zoom)


def render_pages(
    pdf: Path,
    *,
    dpi: int = DEFAULT_DPI,
    max_pages: int | None = None,
) -> Iterator[PageImage]:
    """Render pages to PIL images, one at a time.

    A generator on purpose: a 40-page paper at 150 DPI is ~250 MB of bitmap if
    materialized as a list, and the caller only ever needs one page at a time
    because embedding happens page by page.
    """
    # `import pymupdf`, not the older `import fitz` alias, which 1.28 warns is
    # deprecated. Imported here so `--help` does not pay for it.
    import pymupdf
    from PIL import Image

    zoom = dpi / POINTS_PER_INCH
    matrix = pymupdf.Matrix(zoom, zoom)

    with pymupdf.open(pdf) as doc:
        page_count = doc.page_count
        limit = page_count if max_pages is None else min(max_pages, page_count)
        for index in range(limit):
            page = doc.load_page(index)
            rect = page.rect  # see PageImage docstring
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes(
                "RGB", (pixmap.width, pixmap.height), pixmap.samples
            )
            yield PageImage(
                pdf_path=pdf,
                page_no=index + 1,
                page_count=page_count,
                image=image,
                width_pt=float(rect.width),
                height_pt=float(rect.height),
                zoom=zoom,
            )


def page_words(pdf: Path, page_no: int) -> list[tuple[float, float, float, float, str]]:
    """Words on a page as `(x0, y0, x1, y1, word)`, in the space `page.rect` uses.

    Returned in PyMuPDF's natural order, which is block/line/word order -- close
    enough to reading order that joining the survivors of a bbox filter produces
    a readable snippet.

    ROTATION. `get_text("words")` reports coordinates in the page's UNROTATED
    space, while `page.rect` -- and the pixmap `render_pages` produces -- are
    rotated. On a /Rotate 90 page those disagree: measured on a 400x600 page
    rotated to 600x400, a word came back at y=480..505, outside the 400-point
    height it is displayed in. Left uncorrected, every box on such a page points
    somewhere plausible and wrong, with nothing raising to say so. Multiplying by
    `page.rotation_matrix` moves the words into the displayed space, where the
    grid laid over the rendered image also lives. The matrix is the identity on
    an unrotated page, so this costs nothing in the common case.

    `page_no` is 1-based; an out-of-range page returns an empty list rather than
    raising, so a stale index in the vector store degrades to "no text found"
    instead of crashing a search.
    """
    import pymupdf

    with pymupdf.open(pdf) as doc:
        if not 1 <= page_no <= doc.page_count:
            return []
        page = doc.load_page(page_no - 1)
        matrix = page.rotation_matrix
        words: list[tuple[float, float, float, float, str]] = []
        # get_text("words") yields (x0, y0, x1, y1, word, block_no, line_no, word_no)
        for w in page.get_text("words"):
            rect = pymupdf.Rect(w[0], w[1], w[2], w[3]) * matrix
            rect.normalize()  # rotation can swap the corners
            words.append(
                (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1), str(w[4]))
            )
        return words


def page_lines(pdf: Path, page_no: int) -> list[tuple[float, float, float, float, str]]:
    """Text lines on a page as `(x0, y0, x1, y1, text)`, in `page.rect` space.

    Lines rather than words, because a region is narrow -- a grid cell on A4 is
    about 25 points across, which is four or five characters -- and a snippet
    assembled from only the words strictly inside it reads like "gen- generation-"
    instead of a sentence. Taking whole lines that the region touches gives back
    the sentence the region points at, and respects two-column layouts, since a
    line's bbox stops at its own column.

    Same rotation correction as `page_words`; see that docstring.
    """
    import pymupdf

    with pymupdf.open(pdf) as doc:
        if not 1 <= page_no <= doc.page_count:
            return []
        page = doc.load_page(page_no - 1)
        matrix = page.rotation_matrix
        lines: list[tuple[float, float, float, float, str]] = []
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                text = "".join(span.get("text", "") for span in line.get("spans", []))
                if not text.strip():
                    continue
                x0, y0, x1, y1 = line["bbox"]
                rect = pymupdf.Rect(x0, y0, x1, y1) * matrix
                rect.normalize()
                lines.append(
                    (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1),
                     text.strip())
                )
        return lines


def lines_in_bbox(
    lines: list[tuple[float, float, float, float, str]],
    bbox: tuple[float, float, float, float],
) -> list[str]:
    """Text of every line whose box overlaps `bbox` at all, in reading order.

    Any overlap counts, unlike `words_in_bbox`'s area fraction: the point is to
    recover the whole line a narrow region lands in, so clipping one character of
    it should still pull the sentence in.
    """
    x0, y0, x1, y1 = bbox
    hits = [
        line for line in lines
        if min(line[2], x1) > max(line[0], x0) and min(line[3], y1) > max(line[1], y0)
    ]
    return [line[4] for column in _columns(hits) for line in column]


def _columns(
    lines: list[tuple[float, float, float, float, str]],
) -> list[list[tuple[float, float, float, float, str]]]:
    """Split lines into columns, each ordered top to bottom, left column first.

    Sorting purely by (y, x) interleaves the two columns of a two-column paper
    whenever a region is wide enough to touch both, producing snippets like
    "the output such as the answer, the sentence where Conventional Query
    Expansion. the answer belongs to, and ..." -- two unrelated sentences zipped
    together. Most papers in this corpus are two-column, so this is the common
    case rather than an edge one.

    Lines are grouped by horizontal overlap: a line joins a column if it overlaps
    that column's current span. No page-layout analysis, and none needed -- the
    input is already restricted to the lines one region touches.

    Full-width lines are pulled out FIRST, because they bridge. A section heading
    spanning both columns overlaps each of them, so a purely greedy pass merges
    the two into one group and the interleaving this function exists to prevent
    comes straight back. They are emitted as their own leading group, in vertical
    order, which is also where a heading belongs in a snippet.
    """
    if not lines:
        return []

    left = min(line[0] for line in lines)
    right = max(line[2] for line in lines)
    span = right - left
    # 0.7 rather than 1.0: a heading rarely fills the measured span exactly, and
    # body text in a two-column layout sits near 0.5, so the gap is wide.
    wide = [line for line in lines if span > 0 and (line[2] - line[0]) >= 0.7 * span]
    rest = [line for line in lines if line not in wide]

    columns: list[list] = []
    spans: list[tuple[float, float]] = []
    for line in sorted(rest, key=lambda l: l[0]):  # left to right
        for i, (sx0, sx1) in enumerate(spans):
            if min(line[2], sx1) > max(line[0], sx0):
                columns[i].append(line)
                spans[i] = (min(sx0, line[0]), max(sx1, line[2]))
                break
        else:
            columns.append([line])
            spans.append((line[0], line[2]))

    if wide:
        columns.insert(0, list(wide))
    for column in columns:
        column.sort(key=lambda l: (round(l[1], 1), l[0]))
    return columns


def words_in_bbox(
    words: list[tuple[float, float, float, float, str]],
    bbox: tuple[float, float, float, float],
    *,
    min_overlap: float = 0.3,
) -> list[str]:
    """Words whose area overlaps `bbox` by at least `min_overlap` of the word.

    Overlap fraction rather than plain intersection: a region edge that clips one
    pixel off a neighbouring column should not drag that column's words into the
    snippet. Measured against the word's own area, so a small word inside a big
    region always counts.
    """
    x0, y0, x1, y1 = bbox
    kept: list[str] = []
    for wx0, wy0, wx1, wy1, text in words:
        ox = min(wx1, x1) - max(wx0, x0)
        oy = min(wy1, y1) - max(wy0, y0)
        if ox <= 0 or oy <= 0:
            continue
        area = (wx1 - wx0) * (wy1 - wy0)
        if area <= 0:
            continue
        if (ox * oy) / area >= min_overlap:
            kept.append(text)
    return kept


def pdf_sha256(pdf: Path) -> str:
    """Content hash, used as the PDF's identity in the vector store.

    Content rather than path: the two source tools name files by title slug plus
    a hash of the download URL, so the same paper fetched twice from different
    hosts lands under two names. Hashing the bytes means re-indexing that second
    copy is a no-op instead of a duplicate set of points.
    """
    digest = hashlib.sha256()
    with open(pdf, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
