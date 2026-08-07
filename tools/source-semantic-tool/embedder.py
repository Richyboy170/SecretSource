"""Embed text with a local sentence-transformers model.

Kept in its own module so the torch import is paid only when ranking actually
runs — `--help`, query expansion and the retrieval fan-out never touch it.

No API key and no network after the first run: the model downloads once into the
HuggingFace cache and everything afterwards is local.
"""

from __future__ import annotations

import math
import os

# Fast and small (~90 MB). See the README's Tuning table for the alternatives —
# `allenai-specter` is trained on scientific papers, `all-mpnet-base-v2` scores
# better and runs slower. Changing the model invalidates MIN_SIMILARITY.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _is_cached(model_name: str) -> bool:
    """True when the model snapshot is already in the HuggingFace cache.

    Deliberately filesystem-only: importing `huggingface_hub` to ask it would
    fix the offline flag at its import time, which is exactly what we are trying
    to set first. The layout (`<cache>/models--<org>--<name>/snapshots/<rev>`) is
    HuggingFace's documented one; a miss just means we stay online, so being
    wrong here costs a network check, not a failure.
    """
    cache = (os.environ.get("HUGGINGFACE_HUB_CACHE")
             or os.path.join(os.environ.get("HF_HOME")
                             or os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
                             "hub"))
    snapshots = os.path.join(cache, "models--" + model_name.replace("/", "--"), "snapshots")
    return os.path.isdir(snapshots) and bool(os.listdir(snapshots))


class Embedder:
    """Lazy wrapper around a SentenceTransformer. Safe to construct eagerly."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            # Loading prints a "Loading weights: 100%|...|" bar ahead of any
            # result. encode()'s show_progress_bar covers encoding only.
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

            # With the model already cached, sentence-transformers still calls
            # the Hub to check for updates — which costs a round-trip on every
            # run and prints "You are sending unauthenticated requests... set a
            # HF_TOKEN". That nag comes from a native extension, so no Python
            # logging or warnings filter reaches it; going offline is what
            # actually stops it, and it makes this module's promise of "no
            # network after the first run" true rather than aspirational.
            #
            # Set only when the snapshot is on disk, so a first run can still
            # download. Both vars must be set before the import below reads them.
            if _is_cached(self.model_name):
                os.environ.setdefault("HF_HUB_OFFLINE", "1")

            # Imported here, not at module scope: this line is what pulls in torch.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch to L2-normalized vectors."""
        if not texts:
            return []
        vectors = self._load().encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return [[float(x) for x in v] for v in vectors]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity.

    encode() already returns normalized vectors, so this is a dot product in
    practice — but the norms are recomputed anyway so a caller passing raw
    vectors gets a real cosine instead of an unbounded dot product.
    """
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)
