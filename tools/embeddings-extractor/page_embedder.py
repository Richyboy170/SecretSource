"""Embed page images and query text with a ColPali-family late-interaction model.

    embedder = PageEmbedder()             # nothing loaded yet
    page = embedder.embed_page(image)     # ~n x 128 patch vectors
    query = embedder.embed_query("...")   # ~t x 128 token vectors

Deliberately shaped like ../source-semantic-tool/embedder.py: lazy load, so torch
is imported only when embedding actually runs and `--help` stays instant, and
HF_HUB_OFFLINE set once the snapshot is on disk so later runs need no network.
`_is_cached` below is a copy of that module's, not an import -- the two tools do
not import from each other, so either folder can be moved on its own.

THE MODEL IS NOT THE ONE source-semantic-tool USES, AND THE SPACES DO NOT MEET.
That tool embeds a description with SPECTER into one 768-dim vector. This one
embeds the same description into a sequence of 128-dim token vectors. The
integration point between the two tools is the description STRING; a vector from
one is meaningless to the other.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

# Idefics3/SmolVLM writes one of these before each tile's patch run; the numbers
# are 1-based. Parsing them is what lets attribution reassemble a page-wide grid.
_TILE_MARKER = re.compile(r"<row_(\d+)_col_(\d+)>")

if TYPE_CHECKING:
    from PIL import Image

# SmolVLM-500M backbone: ~5-10 s/page on CPU, which makes a 15-page paper about
# two minutes. The 3B ColQwen2.5 scores better but costs ~30-60 s/page on this
# CPU-only box; it is one --model away for anyone running on a GPU.
DEFAULT_MODEL = "vidore/colSmol-500M"

# (model class, processor class) in colpali_engine.models. Resolved by name so
# adding a model is a dict entry, and so an unknown --model fails with a list of
# what IS supported rather than an AttributeError from deep inside the library.
MODELS: dict[str, tuple[str, str]] = {
    "vidore/colSmol-500M": ("ColIdefics3", "ColIdefics3Processor"),
    "vidore/colSmol-256M": ("ColIdefics3", "ColIdefics3Processor"),
    "vidore/ColSmolVLM-Instruct-500M-base": ("ColIdefics3", "ColIdefics3Processor"),
    "vidore/colqwen2.5-v0.2": ("ColQwen2_5", "ColQwen2_5_Processor"),
    "vidore/colqwen2.5-v0.1": ("ColQwen2_5", "ColQwen2_5_Processor"),
    "vidore/colpali-v1.3": ("ColPali", "ColPaliProcessor"),
}


@dataclass
class PageEmbedding:
    """A page's vectors, plus what attribution needs to read them as a grid.

    `vectors` is EVERY output vector, image patches and any surrounding special
    tokens alike. That is deliberate: the model was trained to be scored with its
    full output, so dropping tokens to make attribution tidier would change the
    ranking. The layout fields let attribution look at just the image patches
    without touching what gets scored.

    `grid_rows`/`grid_cols` are the patch grid, and `image_token_index` lists the
    positions in `vectors` that are image patches, in row-major order. When the
    processor splits an image into tiles those positions are NOT contiguous --
    hence an explicit index list rather than an offset and a length.
    `grid_ok` is False when the layout could not be reconciled with a single
    grid, and attribution then declines to report a region rather than pointing
    at the wrong part of the page.
    """

    vectors: list[list[float]]
    grid_rows: int
    grid_cols: int
    image_token_index: list[int]
    grid_ok: bool

    @property
    def n_vectors(self) -> int:
        return len(self.vectors)


def _is_cached(model_name: str) -> bool:
    """True when the model snapshot is already in the HuggingFace cache.

    Filesystem-only, for the same reason as in source-semantic-tool's embedder:
    importing huggingface_hub to ask would fix the offline flag at ITS import
    time, which is what we are trying to set first. A miss costs a network check,
    not a failure.
    """
    cache = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME")
        or os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
        "hub",
    )
    snapshots = os.path.join(
        cache, "models--" + model_name.replace("/", "--"), "snapshots"
    )
    return os.path.isdir(snapshots) and bool(os.listdir(snapshots))


class PageEmbedder:
    """Lazy wrapper around a ColPali-family model. Safe to construct eagerly."""

    def __init__(self, model_name: str = DEFAULT_MODEL, *, device: str | None = None):
        if model_name not in MODELS:
            raise ValueError(
                f"unknown model {model_name!r}; supported: {', '.join(sorted(MODELS))}"
            )
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None

    # -- loading ------------------------------------------------------------- #

    def _load(self):
        if self._model is not None:
            return self._model, self._processor

        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        if _is_cached(self.model_name):
            # Makes this module's "no network after the first run" promise true
            # rather than aspirational, and silences the unauthenticated-request
            # nag that no logging filter reaches.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        import torch  # this line is what pulls in the ~250 MB of torch
        import colpali_engine.models as cem

        model_cls_name, processor_cls_name = MODELS[self.model_name]
        model_cls = getattr(cem, model_cls_name)
        processor_cls = getattr(cem, processor_cls_name)

        device = self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        # float32 on CPU: bfloat16 matmul on CPU is emulated and slower than the
        # dtype suggests, and the memory it saves is not the constraint here.
        dtype = torch.float32 if device == "cpu" else torch.bfloat16

        self._model = model_cls.from_pretrained(
            self.model_name, torch_dtype=dtype, device_map=device
        ).eval()
        self._processor = processor_cls.from_pretrained(self.model_name)
        return self._model, self._processor

    @property
    def processor(self):
        return self._load()[1]

    # -- embedding ----------------------------------------------------------- #

    def embed_query(self, text: str) -> list[list[float]]:
        """Per-token vectors for a search description."""
        import torch

        model, processor = self._load()
        batch = processor.process_queries([text]).to(model.device)
        with torch.no_grad():
            out = model(**batch)
        return out[0].to(torch.float32).cpu().tolist()

    def embed_page(self, image: "Image.Image") -> PageEmbedding:
        """Patch vectors for one page image, plus its grid layout."""
        import torch

        model, processor = self._load()
        batch = processor.process_images([image]).to(model.device)
        with torch.no_grad():
            out = model(**batch)
        vectors = out[0].to(torch.float32).cpu().tolist()

        rows, cols, index, ok = self._layout(processor, batch, image, len(vectors))
        return PageEmbedding(
            vectors=vectors,
            grid_rows=rows,
            grid_cols=cols,
            image_token_index=index,
            grid_ok=ok,
        )

    # -- layout -------------------------------------------------------------- #

    def _layout(
        self, processor, batch, image: "Image.Image", n_vectors: int
    ) -> tuple[int, int, list[int], bool]:
        """Work out which output vectors are image patches, and their grid shape.

        This decides whether "which region of the page matched" can be answered.
        The result is a rows x cols grid plus `index`, mapping each cell (row
        major) to its position in the output vector sequence.

        THE DEFAULT MODEL DOES NOT EMIT ONE FLAT GRID, so the obvious
        implementation is wrong. Measured on vidore/colSmol-500M with a 1241x1754
        page (2026-08-08): 875 output vectors, of which 832 are image tokens, in
        13 non-contiguous runs of 64. Idefics3/SmolVLM splits the page into tiles
        -- `pixel_values` came back as (1, 13, 3, 512, 512) -- emitting one run
        per tile, each preceded by a `<row_i_col_j>` marker, plus a downsampled
        whole-page view after `<global-img>`. Here that was a 4x3 tile grid.
        Meanwhile `processor.get_n_patches()` reported 147x104 = 15288, the raw
        pre-merge patch count, which matches nothing.

        Reading the markers recovers a REAL grid, and a finer one than a single
        tile would give: 12 tiles x 8x8 cells each = a 32x24 grid over the page,
        or about one line of text per cell on A4. The 64 global-view vectors are
        excluded -- they describe the whole page, so they carry no location.

        Two facts make the mapping exact rather than approximate. The tiles are a
        complete rectangular cover (4x3 with no gaps), and the page is *resized*
        into that cover rather than padded -- verified by per-tile pixel standard
        deviation, where a padded tile would be constant and none was. So cell
        (r, c) of the assembled grid is exactly the (r/rows, c/cols) fraction of
        the page.

        Falls back to a single flat grid when the processor does not split, and
        to grid_ok=False when nothing reconciles -- in which case attribution
        reports the page without a region rather than inventing one.
        """
        tiled = self._tiled_layout(processor, batch, n_vectors)
        if tiled is not None:
            rows, cols, index = tiled
            return rows, cols, index, True

        # No tiling markers: a flat run of patches, as ColPali/PaliGemma emits.
        index = self._image_token_index(processor, batch, n_vectors)
        rows, cols = self._patch_grid(processor, image)
        if index and rows > 0 and cols > 0 and rows * cols == len(index):
            return rows, cols, index, True

        # Last resort: a square grid, if the count happens to be a perfect square.
        if index:
            side = math.isqrt(len(index))
            if side * side == len(index):
                return side, side, index, True
        return rows, cols, index, False

    @staticmethod
    def _tiled_layout(
        processor, batch, n_vectors: int
    ) -> tuple[int, int, list[int]] | None:
        """Reassemble per-tile patch runs into one page-wide grid.

        Returns `(rows, cols, index)` or None when the sequence does not look
        tiled. See `_layout` for the measured layout this parses.
        """
        try:
            ids = batch["input_ids"][0].tolist()
        except Exception:
            return None
        if len(ids) != n_vectors:
            # Output vectors must align 1:1 with input tokens for positions to
            # mean anything. If a model ever pools, bail rather than guess.
            return None

        positions = PageEmbedder._image_token_index(processor, batch, n_vectors)
        if not positions:
            return None

        runs: list[list[int]] = []
        for pos in positions:
            if runs and pos == runs[-1][-1] + 1:
                runs[-1].append(pos)
            else:
                runs.append([pos])

        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            return None

        tiles: dict[tuple[int, int], list[int]] = {}
        for run in runs:
            if run[0] == 0:
                continue
            try:
                marker = tokenizer.decode([ids[run[0] - 1]])
            except Exception:
                continue
            match = _TILE_MARKER.match(marker.strip())
            if match:  # anything else (the global view) has no location
                tiles[(int(match.group(1)), int(match.group(2)))] = run

        if not tiles:
            return None

        sizes = {len(run) for run in tiles.values()}
        if len(sizes) != 1:
            return None  # ragged tiles cannot form a rectangular grid
        per_tile = sizes.pop()
        side = math.isqrt(per_tile)
        if side * side != per_tile:
            return None  # non-square tile; no unambiguous row/col split

        tile_rows = max(r for r, _ in tiles)
        tile_cols = max(c for _, c in tiles)
        if len(tiles) != tile_rows * tile_cols:
            return None  # not a complete rectangular cover

        rows, cols = tile_rows * side, tile_cols * side
        index = [-1] * (rows * cols)
        for (tile_row, tile_col), run in tiles.items():
            for r in range(side):
                for c in range(side):
                    cell = ((tile_row - 1) * side + r) * cols + (tile_col - 1) * side + c
                    index[cell] = run[r * side + c]
        if any(v < 0 for v in index):
            return None
        return rows, cols, index

    @staticmethod
    def _image_token_index(processor, batch, n_vectors: int) -> list[int]:
        """Positions of image-patch vectors in the output sequence."""
        # Preferred: the processor tells us directly.
        getter = getattr(processor, "get_image_mask", None)
        if getter is not None:
            try:
                mask = getter(batch)[0]
                return [i for i, flag in enumerate(mask.tolist()) if flag]
            except Exception:
                pass
        # Fallback: match the tokenizer's image-token id against input_ids.
        try:
            image_token_id = getattr(processor, "image_token_id", None)
            ids = batch["input_ids"][0].tolist()
            if image_token_id is not None and len(ids) == n_vectors:
                return [i for i, t in enumerate(ids) if t == image_token_id]
        except Exception:
            pass
        return []

    @staticmethod
    def _patch_grid(processor, image: "Image.Image") -> tuple[int, int]:
        """(rows, cols) of the patch grid, if the processor will say."""
        getter = getattr(processor, "get_n_patches", None)
        if getter is None:
            return 0, 0
        # Signature differs across model families; try the documented forms.
        attempts: list[tuple[tuple, dict[str, Any]]] = [
            ((image.size,), {}),
            ((image.size,), {"patch_size": getattr(processor, "patch_size", 14)}),
            (
                (image.size,),
                {
                    "patch_size": getattr(processor, "patch_size", 14),
                    "spatial_merge_size": getattr(processor, "spatial_merge_size", 2),
                },
            ),
        ]
        for args, kwargs in attempts:
            try:
                rows, cols = getter(*args, **kwargs)
                return int(rows), int(cols)
            except Exception:
                continue
        return 0, 0

    # -- introspection ------------------------------------------------------- #

    def describe_layout(self, image: "Image.Image") -> dict[str, Any]:
        """One page's token layout, for the measurement the README documents.

        Run this before trusting any reported region, and after changing --model:
        it is the check that the contiguous-grid assumption in attribution.py
        actually holds for the model in use, and it is where the real per-page
        vector count (hence the storage estimate) comes from.
        """
        embedding = self.embed_page(image)
        _, processor = self._load()
        image_processor = getattr(processor, "image_processor", None)
        dim = len(embedding.vectors[0]) if embedding.vectors else 0
        return {
            "model": self.model_name,
            "image_size": list(image.size),
            "n_vectors": embedding.n_vectors,
            "dim": dim,
            # Cells in the grid attribution uses. Smaller than the model's total
            # image-token count when a downsampled global view is present -- that
            # view describes the whole page, so it is excluded as location-free.
            "grid_rows": embedding.grid_rows,
            "grid_cols": embedding.grid_cols,
            "grid_cells": len(embedding.image_token_index),
            "grid_ok": embedding.grid_ok,
            "grid_contiguous": _contiguous(embedding.image_token_index),
            "do_image_splitting": getattr(image_processor, "do_image_splitting", None),
            "bytes_per_page": embedding.n_vectors * dim * 4,
        }


def _contiguous(index: Sequence[int]) -> bool:
    """True when positions form one unbroken run -- the tiling tell-tale."""
    return bool(index) and list(index) == list(range(index[0], index[0] + len(index)))
