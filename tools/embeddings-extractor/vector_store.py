"""Qdrant storage and MaxSim scoring for ColPali-style page multi-vectors.

A page is not one vector here -- it is a few hundred to a thousand 128-dim patch
vectors, and a query is one vector per query token. The score between them is
MaxSim (late interaction): for each query token take its best-matching patch,
then sum. Qdrant supports this natively via `MultiVectorComparator.MAX_SIM`.

    ensure_collection(client)
    upsert_page(client, point_id=..., vectors=page_vecs, payload={...})
    hits = search(client, query_vecs, limit=10)

WHAT THE SCORES ARE, AND ARE NOT. MaxSim is an unbounded sum over query tokens --
a 15-token query lands roughly in 0-20, a 5-token query in a much lower range,
and neither is a cosine. Two consequences the CLI and README repeat:

  * scores are comparable only WITHIN one query. Ranking is meaningful; the
    absolute number is not, and there is no threshold to compare it against.
  * ../source-semantic-tool's MIN_SIMILARITY = 0.70 floor carries over to
    nothing here. That constant was calibrated for SPECTER cosines, and as
    that tool's embedder.py puts it, "cosines are not comparable across models."
    These are not even cosines.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

COLLECTION = "research_pages"

# Named vector rather than the default unnamed one: a second representation
# (a mean-pooled single vector for two-stage prefetch, if the corpus ever
# outgrows a brute-force scan) can then be added without a migration.
VECTOR_NAME = "colpali"

# ColBERT-style late-interaction models all project to 128 dims regardless of
# backbone size -- ColPali, ColQwen and ColSmol alike. Still a parameter, since
# getting it wrong is a silent dimension mismatch at upsert time.
DEFAULT_DIM = 128

# Namespace for deterministic point ids. Any fixed UUID works; this one is
# arbitrary and must never change, or every already-indexed page becomes
# unreachable and re-indexing silently duplicates the corpus.
_NAMESPACE = uuid.UUID("6f0f4d9e-8b3a-5c21-9a77-4c2d1e6b8f30")


@dataclass
class ScoredPage:
    """One page hit. `vectors` is filled in only when attribution needs it."""

    point_id: str
    score: float
    payload: dict[str, Any]
    vectors: list[list[float]] | None = field(default=None, repr=False)

    @property
    def pdf_name(self) -> str:
        return str(self.payload.get("pdf_name") or "?")

    @property
    def page_no(self) -> int:
        return int(self.payload.get("page_no") or 0)


def point_id_for(pdf_sha: str, page_no: int, model: str) -> str:
    """Stable id for (document, page, model).

    Deterministic so that re-running `index` overwrites rather than duplicates,
    and so an interrupted run can resume: a CPU-only pass over a long paper takes
    minutes, and losing it to a Ctrl-C would make the tool unusable in practice.
    The model is part of the key because two models' vectors are not comparable
    and must not collide on one point.
    """
    return str(uuid.uuid5(_NAMESPACE, f"{pdf_sha}:{page_no}:{model}"))


# -- client ------------------------------------------------------------------ #


def connect(url: str | None = None, path: str | None = None):
    """Open a Qdrant client against a server (`url`) or an embedded store (`path`).

    Embedded mode keeps the tool runnable before Docker exists and is what the
    offline tests use; a server is the deployment target. Same API either way,
    which is the reason for routing both through one function.
    """
    from qdrant_client import QdrantClient

    if url and path:
        raise ValueError("pass a Qdrant url or a path, not both")
    if path:
        return QdrantClient(path=path)
    return QdrantClient(url=url or "http://localhost:6333")


def ensure_collection(client, *, dim: int = DEFAULT_DIM, name: str = COLLECTION) -> None:
    """Create the multi-vector collection if it is missing. Idempotent."""
    from qdrant_client import models

    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config={
            VECTOR_NAME: models.VectorParams(
                size=dim,
                distance=models.Distance.DOT,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
                # HNSW off. MaxSim compares SETS of vectors, and a proximity
                # graph built over individual vectors cannot represent that, so
                # Qdrant requires m=0 and answers with a full scan. Fine at the
                # few-hundred-page scale this tool targets; the growth path is a
                # mean-pooled prefetch vector, noted in the README.
                hnsw_config=models.HnswConfigDiff(m=0),
            )
        },
    )
    # Payload indexes for the skip-if-indexed lookup, which filters on these two
    # every run. Without them that filter is also a full scan.
    #
    # Embedded mode warns that payload indexes have no effect there and ignores
    # them; that is expected and harmless, since a local store is small enough
    # for the scan. The call is still made so a server-backed collection gets the
    # index it needs.
    for field_name in ("pdf_sha", "model"):
        client.create_payload_index(
            collection_name=name,
            field_name=field_name,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def upsert_page(
    client,
    *,
    point_id: str,
    vectors: Sequence[Sequence[float]],
    payload: dict[str, Any],
    name: str = COLLECTION,
) -> None:
    from qdrant_client import models

    client.upsert(
        collection_name=name,
        points=[
            models.PointStruct(
                id=point_id,
                vector={VECTOR_NAME: [list(map(float, v)) for v in vectors]},
                payload=payload,
            )
        ],
    )


def indexed_pages(client, pdf_sha: str, model: str, *, name: str = COLLECTION) -> set[int]:
    """Page numbers already stored for this (document, model).

    Drives `index`'s resume: pages in this set are skipped unless --reindex.
    """
    from qdrant_client import models

    if not client.collection_exists(name):
        return set()
    flt = models.Filter(
        must=[
            models.FieldCondition(key="pdf_sha", match=models.MatchValue(value=pdf_sha)),
            models.FieldCondition(key="model", match=models.MatchValue(value=model)),
        ]
    )
    pages: set[int] = set()
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=name,
            scroll_filter=flt,
            limit=256,
            offset=offset,
            with_payload=["page_no"],
            with_vectors=False,
        )
        for record in records:
            page_no = (record.payload or {}).get("page_no")
            if page_no is not None:
                pages.add(int(page_no))
        if offset is None:
            break
    return pages


def search(
    client,
    query_vectors: Sequence[Sequence[float]],
    *,
    limit: int = 10,
    with_vectors: bool = False,
    name: str = COLLECTION,
) -> list[ScoredPage]:
    """Rank pages by MaxSim against the query's per-token vectors.

    `with_vectors=True` pulls each hit's full patch matrix back so attribution
    can recompute which patch won each query token -- Qdrant returns the summed
    score but not the argmaxes behind it, so that second pass is unavoidable.
    It is only paid for the top `limit` pages.
    """
    response = client.query_points(
        collection_name=name,
        query=[list(map(float, v)) for v in query_vectors],
        using=VECTOR_NAME,
        limit=limit,
        with_payload=True,
        with_vectors=with_vectors,
    )
    hits: list[ScoredPage] = []
    for point in response.points:
        vectors = None
        if with_vectors and point.vector:
            raw = point.vector
            vectors = raw[VECTOR_NAME] if isinstance(raw, dict) else raw
        hits.append(
            ScoredPage(
                point_id=str(point.id),
                score=float(point.score),
                payload=dict(point.payload or {}),
                vectors=vectors,
            )
        )
    return hits


# -- scoring ----------------------------------------------------------------- #


def maxsim(
    query_vectors: Sequence[Sequence[float]],
    page_vectors: Sequence[Sequence[float]],
) -> tuple[float, list[tuple[int, float]]]:
    """MaxSim score, plus the winning page-vector index for each query token.

    Returns `(score, [(argmax_index, contribution), ...])` with one entry per
    query token, in query-token order.

    This duplicates what Qdrant computes, and that is the point: Qdrant hands
    back the sum, while attribution needs to know WHICH patch each query token
    matched. Recomputing locally over the top-k pages is cheap (k x n x 128
    dot products) and keeps the "where on the page" answer honest -- it is
    derived from the same arithmetic that produced the ranking, not a separate
    heuristic that could disagree with it.
    """
    import numpy as np

    if len(query_vectors) == 0 or len(page_vectors) == 0:
        return 0.0, []

    q = np.asarray(query_vectors, dtype=np.float32)
    p = np.asarray(page_vectors, dtype=np.float32)
    sims = q @ p.T  # (n_query_tokens, n_page_vectors)
    best = sims.argmax(axis=1)
    contributions = sims[np.arange(sims.shape[0]), best]
    return float(contributions.sum()), [
        (int(i), float(c)) for i, c in zip(best, contributions)
    ]


def rank_locally(
    query_vectors: Sequence[Sequence[float]],
    pages: Iterable[tuple[str, Sequence[Sequence[float]], dict[str, Any]]],
    *,
    limit: int = 10,
) -> list[ScoredPage]:
    """Brute-force MaxSim ranking without a Qdrant server.

    The fallback path if embedded Qdrant turns out not to support multivector
    collections, and the way the offline tests exercise ranking without standing
    anything up. Same ordering as `search`, just without persistence.
    """
    scored = [
        ScoredPage(
            point_id=pid,
            score=maxsim(query_vectors, vecs)[0],
            payload=dict(payload),
            vectors=[list(map(float, v)) for v in vecs],
        )
        for pid, vecs, payload in pages
    ]
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:limit]
