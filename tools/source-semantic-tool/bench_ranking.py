"""Compare embedding models by re-ranking one frozen candidate pool.

MIN_SIMILARITY is model-specific and has to be measured (README, "Calibrating
the floor"), and so does the choice of model itself. The obvious way to measure
— run fetch_semantic once per model and compare — does not work: retrieval is
the noisy part of this pipeline. The last batch run's manifest shows OpenAlex
and Semantic Scholar returning 429 on every query, so two runs minutes apart see
different candidate pools, and a model comparison across them is really a
comparison of which sources happened to answer.

So the pool is frozen once (`fetch_semantic.py --dump-pool`) and every model
re-ranks that same list. `SemanticResolver._rank` is pure over
(description, papers), so this needs no network, no API keys, and no changes to
the resolver — only an Embedder swapped underneath it.

    py fetch_semantic.py -f topics.txt --dry-run --rank-only \
        --min-similarity 0.0 --dump-pool pool.jsonl
    py bench_ranking.py pool.jsonl --models all-MiniLM-L6-v2,allenai-specter

What to read in the output:

  * `truncated` — how many papers are longer than the model's input window. This
    is the number that motivated the whole exercise: at 256 word-pieces a normal
    200-250 word abstract is clipped, so the score comes from the title plus the
    opening sentences rather than from the abstract.
  * the score distribution — feeds MIN_SIMILARITY. Cosines are NOT comparable
    across models, so a floor tuned on one is meaningless on another.
  * the top-k titles — the only real quality signal. A model can post a tidier
    distribution and still rank worse; run a control query you know is nonsense
    and check that its best score sits below the real queries' correct answers.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import runcost  # first local import: its clocks start when it is imported
from embedder import DEFAULT_MODEL, Embedder
from semantic_resolver import Paper, SemanticResolver

# Models worth comparing for this tool, and why. Not a default — pass --models.
CANDIDATE_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",      # incumbent; 256-token window
    "allenai-specter",                             # trained on scientific title+abstract
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",  # short query -> long passage
]


def load_pools(path: Path) -> list[dict]:
    """Read the JSONL written by --dump-pool. One record per description."""
    records = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"{path}:{lineno}: skipped unparseable record ({exc})", file=sys.stderr)
    return records


def to_papers(record: dict) -> list[Paper]:
    """Rehydrate the dumped pool into the objects _rank expects.

    Only the fields _rank reads are restored; the dump deliberately carries no
    similarity, since recomputing it is the entire point.
    """
    return [
        Paper(
            key=p.get("key") or "",
            title=p.get("title"),
            abstract=p.get("abstract"),
            doi=p.get("doi"),
            pdf_urls=list(p.get("pdf_urls") or []),
            landing_url=p.get("landing_url"),
            retrieval_sources=list(p.get("retrieval_sources") or []),
            matched_queries=list(p.get("matched_queries") or []),
        )
        for p in record.get("papers") or []
    ]


def ranker(model: str) -> SemanticResolver:
    """A resolver that can only rank.

    __new__ skips __init__ so no HTTP client, rate limiter or API key is
    constructed — _rank touches nothing but self.embedder. test_selection.py
    uses the same trick to exercise _select offline.
    """
    resolver = SemanticResolver.__new__(SemanticResolver)
    resolver.embedder = Embedder(model)
    return resolver


def token_lengths(embedder: Embedder, texts: list[str]) -> list[int]:
    """Word-piece length of each embedded text, before truncation.

    Tokenizing past the window is the measurement, not a mistake, so the
    transformers warning about it ("will result in indexing errors" — it will
    not; nothing is fed to the model here) is silenced rather than printed once
    per model run.
    """
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    tokenizer = embedder._load().tokenizer
    return [len(tokenizer.encode(t, truncation=False, add_special_tokens=True))
            for t in texts]


def summarize(scores: list[float]) -> str:
    """Same distribution line fetch_semantic prints, so the two are comparable."""
    if not scores:
        return "no scores"
    n = len(scores)
    at = lambda p: scores[min(n - 1, int(n * p))]  # scores arrive sorted descending
    return (f"max={scores[0]:.3f}  top10%={at(0.10):.3f}  "
            f"median={at(0.50):.3f}  min={scores[-1]:.3f}")


def bench(records: list[dict], models: list[str], top: int) -> dict:
    """Rank every pool with every model. Returns a JSON-serializable report."""
    report = {"models": {}}
    for model in models:
        resolver = ranker(model)
        window = resolver.embedder.max_seq_length
        print(f"\n{'=' * 78}\n{model}  (window: {window} word-pieces)\n{'=' * 78}")

        per_query = []
        for record in records:
            description = record.get("description") or ""
            papers = to_papers(record)
            if not papers:
                print(f'\n"{description}"\n  empty pool — skipped')
                continue

            texts = [resolver.embedder.join_paper_text(p.title, p.abstract)
                     for p in papers]
            lengths = token_lengths(resolver.embedder, texts)
            clipped = [n for n in lengths if n > window]

            ranked = resolver._rank(description, papers)
            scores = [p.similarity for p in ranked]

            print(f'\n"{description}"')
            print(f"  pool={len(ranked)}  {summarize(scores)}")
            print(f"  truncated: {len(clipped)}/{len(lengths)} papers exceed the window"
                  f"  (longest={max(lengths)} word-pieces)")
            title_only = sum(1 for p in ranked if p.title_only)
            if title_only:
                print(f"  scored on title alone (no abstract): {title_only}")
            for paper in ranked[:top]:
                flag = " (title only)" if paper.title_only else ""
                print(f"  {paper.similarity:.3f}  "
                      f"{(paper.title or '(untitled)')[:84]}{flag}")

            per_query.append({
                "description": description,
                "pool": len(ranked),
                "max": round(scores[0], 4),
                "median": round(scores[min(len(scores) - 1, len(scores) // 2)], 4),
                "min": round(scores[-1], 4),
                "truncated": len(clipped),
                "longest_tokens": max(lengths),
                "title_only": title_only,
                "top": [{"similarity": round(p.similarity, 4), "title": p.title}
                        for p in ranked[:top]],
            })

        report["models"][model] = {"window": window, "queries": per_query}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-rank a frozen candidate pool with several embedding models.")
    parser.add_argument("pool", type=Path, help="JSONL from fetch_semantic --dump-pool")
    parser.add_argument("--models", default=DEFAULT_MODEL,
                        help="comma-separated sentence-transformers models "
                             f"(candidates: {', '.join(CANDIDATE_MODELS)})")
    parser.add_argument("--top", type=int, default=10, help="titles to print per query")
    parser.add_argument("--json", type=Path, default=None,
                        help="also write the report here, for the README table")
    args = parser.parse_args()

    if not args.pool.exists():
        print(f"pool not found: {args.pool}\n"
              f"generate one with: py fetch_semantic.py -f topics.txt --dry-run "
              f"--rank-only --min-similarity 0.0 --dump-pool {args.pool}",
              file=sys.stderr)
        return 1

    records = load_pools(args.pool)
    if not records:
        print(f"{args.pool} holds no pool records", file=sys.stderr)
        return 1

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    report = bench(records, models, args.top)

    # Cost belongs in the comparison: a model that ranks marginally better for
    # twice the RAM is a real trade-off, and this is the run that surfaces it.
    # The figure covers every model in `--models` together — peak RAM is a
    # process high-water mark, so it does not attribute to one of them.
    stats = runcost.measure()
    report["usage"] = stats.as_dict()

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"\nwrote {args.json}")
    runcost.report(stats)
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        runcost.report()
    raise SystemExit(code)
