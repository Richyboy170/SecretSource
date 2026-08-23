"""Re-derive VALIDATION.md's floor-survivor columns from the frozen pool.

bench_ranking.py prints max/median/min and a pool-wide title_only count, but not
the 'survive 0.70' / 'of those, title_only' columns the report tabulates. Same
_rank, same pool, so this is the same measurement -- only the aggregation differs.
"""
import json
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL))

from bench_ranking import load_pools, to_papers, ranker  # noqa: E402

resolver = ranker("sentence-transformers/allenai-specter")
rows = []
for record in load_pools(TOOL / "validation_pool.jsonl"):
    ranked = resolver._rank(record.get("description") or "", to_papers(record))
    scores = [p.similarity for p in ranked]
    surv70 = [p for p in ranked if p.similarity >= 0.70]
    surv75 = [p for p in ranked if p.similarity >= 0.75]
    rows.append({
        "description": record.get("description"),
        "pool": len(ranked),
        "max": round(scores[0], 3),
        "median": round(scores[len(scores) // 2], 3),
        "min": round(scores[-1], 3),
        "survive_070": len(surv70),
        "title_only_of_survivors": sum(1 for p in surv70 if p.title_only),
        "survive_075": len(surv75),
        "top5": [(round(p.similarity, 3), (p.title or "")[:70]) for p in ranked[:5]],
    })

print(json.dumps(rows, indent=2, ensure_ascii=False))
for r in rows:
    print(f"{r['description'][:52]:54s} pool={r['pool']:3d} max={r['max']:.3f} "
          f"med={r['median']:.3f} min={r['min']:.3f} "
          f">=0.70: {r['survive_070']}/{r['pool']} "
          f"({round(100*r['survive_070']/r['pool'])}%) title_only={r['title_only_of_survivors']} "
          f">=0.75: {r['survive_075']}/{r['pool']}")
