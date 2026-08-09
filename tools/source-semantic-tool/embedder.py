"""Embed text with a local sentence-transformers model.

Kept in its own module so the torch import is paid only when ranking actually
runs — `--help`, query expansion and the retrieval fan-out never touch it.

No API key and no network after the first run: the model downloads once into the
HuggingFace cache and everything afterwards is local.
"""

from __future__ import annotations

import math
import os

# Trained on scientific title+abstract pairs, 512-token window, ~250 MB.
#
# What this ranks is a title plus an abstract, and the previous default —
# `all-MiniLM-L6-v2`, 256 tokens — could not read one. Measured over a frozen
# pool of 363 candidates (bench_ranking.py, 2026-08-08): 27-53% of papers per
# query overran that window, the longest at 977 word-pieces, so the score came
# from the title and the opening sentences rather than from the abstract. Under
# specter that falls to 0-6 papers per query, and the top-5 stops being led by
# abstract-less hits.
#
# Fully qualified so `_is_cached` finds the snapshot — the bare `allenai-specter`
# alias resolves at load time but not on disk, which would leave every run making
# a Hub round-trip and break offline use.
#
# Changing this invalidates MIN_SIMILARITY: cosines are not comparable across
# models. Re-run the calibration in the README before shipping a swap.
DEFAULT_MODEL = "sentence-transformers/allenai-specter"


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

    @property
    def max_seq_length(self) -> int:
        """Input window in word-pieces. Anything past it is silently truncated.

        Exposed because the thing being embedded is a title plus an abstract:
        `all-MiniLM-L6-v2` stops at 256, which clips a normal 200-250 word
        abstract, so the score ends up driven by the title and the opening
        sentences. Callers that want to know whether the abstract actually
        reached the model have to be able to ask.
        """
        return int(self._load().max_seq_length)

    def join_paper_text(self, title: str | None, abstract: str | None) -> str:
        """Join a title and abstract the way THIS model was trained to read them.

        SPECTER was trained on `title [SEP] abstract` and is documented to be fed
        that way; the all-* MiniLM/MPNet models have no such convention and take
        a plain newline. Getting this wrong does not raise — it just quietly
        scores worse, which is indistinguishable from the model being a bad fit.

        A title-less paper is dropped in `_dedup` before it reaches here, and an
        abstract-less one is scored on its title alone (and penalized by the
        caller), so neither side is padded with a separator it did not earn.
        """
        title = (title or "").strip()
        abstract = (abstract or "").strip()
        if not abstract:
            return title
        if not title:
            return abstract
        return f"{title}{self.pair_separator}{abstract}"

    @property
    def pair_separator(self) -> str:
        """Separator for `join_paper_text`. `[SEP]` for SPECTER-family models."""
        if "specter" in self.model_name.lower():
            # The literal token, not a hardcoded string: SPECTER's tokenizer maps
            # it to a single special id, whereas any other spelling would be
            # tokenized as ordinary text and land the model off-distribution.
            return self._load().tokenizer.sep_token
        return "\n"


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
