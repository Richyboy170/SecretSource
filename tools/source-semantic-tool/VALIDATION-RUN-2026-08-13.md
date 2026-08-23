# Validation re-run — 2026-08-13

A re-execution of [`VALIDATION.md`](VALIDATION.md)'s reproduce block, five days
after the original. Every command, its wall time, its output, and the similarity
numbers it produced. This file records **what happened when the commands were
actually run**; `VALIDATION.md` remains the report of record.

**Result: everything reproduces.** 127/127 offline assertions pass, the score
table re-derives with every number identical, and all six live cases exercised
behave as documented. Five small deltas, all in the retrieval layer where
nondeterminism is expected — see [Deltas](#deltas-vs-2026-08-08). No finding was
overturned; F1, F2, F5 and F6 all reproduced live.

Environment identical to the original: Windows 11 (10.0.26200), CPython 3.14.6,
specter cached, Ollama `qwen3:1.7b` reachable, `S2_API_KEY` **unset**, stock
29-host `sources.txt`. All live runs wrote to a scratch `--out`; `output_pdf/`
and its manifest were untouched.

---

## Summary of every case

| # | Case | Command | Time | Output | Verdict |
|---|---|---|---:|---|---|
| A1 | Selection / backoff / pacing suite | `py test_selection.py` | **1.2 s** | `27/27 passed`, exit 0 | ✅ good |
| A2 | Parsing / dedup / download / allowlist suite | `py test_validation.py` | **3.2 s** | `98/98` + `2/2` = **100**, exit 0 | ✅ good |
| B1 | Re-rank frozen pool | `py bench_ranking.py validation_pool.jsonl --models sentence-transformers/allenai-specter` | **191 s** | 5 queries, all maxima identical to the report | ✅ good |
| B2 | Floor-survivor columns | `py survivors.py` | **144 s** | all five survivor counts identical | ✅ good |
| L1 | Normal query, expansion on | `py fetch_semantic.py "attention mechanisms for neural machine translation" -k 5 --out val_out` | **132 s** | **5/5 `OK`**, sim 0.850→0.831, pool 281 | ✅ good |
| L2 | Idempotency re-run | same + `--no-expand`, same `--out` | **45 s** | **5/5 `HAVE`**, `skipped=5`, nothing re-downloaded | ✅ good |
| L3 | Nonsense control | `py fetch_semantic.py "quantum tunneling … bookbinding adhesives" -k 5 --out val_nonsense` | **75 s** | **5/5 `OK`**, 5 PDFs written, sim 0.763→0.747 | ⚠️ **bad by design — F1** |
| L4 | Allowlist `BLOCK` path | same query + `--sources <only zenodo.org>` `-k 3` | **37 s** | **3/3 `BLOCK`**, **zero files**, only `manifest.jsonl` | ✅ good |
| L8 | Agent tool surface | `import semantic_resolver` + schema inspect | <1 s | Schema well-formed and honest about non-verification | ✅ good |
| L9 | CLI error handling | no args / bad `--sources` / bad `--from-file` | <1 s each | 3 clear messages, exit 1, no traceback | ✅ good |

Offline tier: **5.5 minutes**. Live tier: **4.8 minutes**. Total ≈ 10 minutes.

---

## The similarity scores

### Live output — L1, a well-phrased query (5/5 downloaded)

Command — expansion on (the default), scratch `--out` so `output_pdf/` stays clean:

```powershell
py fetch_semantic.py "attention mechanisms for neural machine translation" -k 5 --out <scratch>/val_out
```

```
OK    [1] Interrogating the Explanatory Power of Attention in Neural Machine Transla
      sim=0.850  arxiv+crossref
OK    [2] Effective Approaches to Attention-based Neural Machine Translation
      sim=0.844  openalex
OK    [3] Neural Machine Translation: A Review and Survey
      sim=0.841  arxiv
OK    [4] Six Challenges for Neural Machine Translation
      sim=0.832  arxiv  (#7 by score)
OK    [5] Advancing Explainability in Neural Machine Translation: Analytical Metrics
      sim=0.831  arxiv  (#8 by score)
downloaded=5
```

All five are on-topic. Ranks 4 and 5 print `(#7 by score)` / `(#8 by score)` —
fetchable-first reordering announcing itself (L5 in the report). The manifest
carries both: `rank` 4,5 against `similarity_rank` 7,8.

### Live output — L3, the nonsense control (also 5/5 downloaded)

Same flags, same defaults — only the description changed:

```powershell
py fetch_semantic.py "quantum tunneling effects in medieval manuscript bookbinding adhesives" -k 5 --out <scratch>/val_nonsense
```

```
OK    [1] Searching for Coherent States: From Origins to Quantum Gravity
      sim=0.763  arxiv  (#5 by score)
OK    [2] Quantum networks theory                                   sim=0.757
OK    [3] Transcribing Medieval Manuscripts for Machine Learning    sim=0.754
OK    [4] Single-particle quantum tunneling in ionic traps          sim=0.750
OK    [5] When Simpler Is Better: Evaluating Translation Pipelines
        for Medieval Latin Manuscripts                              sim=0.747
downloaded=5
```

**This is the finding that matters.** A query with no possible answer produced
the same five `OK` markers, the same `downloaded=5` line, and five real PDFs on
disk. The scores *are* lower (0.763–0.747 against 0.850–0.831), but nothing in
the output labels them weak, and by limitation #1 nothing could — cosine is not
comparable across queries.

**F1's shape reproduced; its tail moved slightly.** Both papers F1 named come
back at identical scores — *Coherent States* 0.763, *Transcribing Medieval
Manuscripts* 0.754 — and the top of the band is unchanged. The bottom moved:
0.717 in the report against **0.747** today. The live pool differed (**90** here,
with expansion on), so the tail is drawn from a different candidate set. F1 never
listed all five titles, so a full title-by-title comparison isn't possible either
way.

### Frozen-pool score table — every column re-derived

`bench_ranking.py` prints max/median/min but not the floor-survivor columns, so
`survivors.py` (added by this run, next to `bench_ranking.py`) aggregates them
from the same `_rank` over the same committed `validation_pool.jsonl` — making
every column below re-derivable from the repo rather than taken on trust.

| Query | pool | max | median | min | survive 0.70 | of those, title-only | survive 0.75 | vs 2026-08-08 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| attention mechanism for NMT | 75 | 0.880 | 0.770 | 0.552 | 58 / 75 (77%) | 13 | 48 | identical ✓ |
| GNN for molecular property prediction | 65 | 0.895 | 0.778 | 0.616 | 56 / 65 (86%) | 14 | 38 | identical ✓ |
| RAG for open-domain QA | 74 | 0.912 | 0.806 | 0.667 | 72 / 74 (97%) | 11 | 61 | identical ✓ |
| loose paraphrase of query 1 | 80 | 0.900 | 0.708 | 0.552 | 48 / 80 (60%) | 15 | 21 | identical ✓ |
| **nonsense control** | 38 | **0.793** | 0.682 | 0.599 | **14 / 38 (37%)** | 3 | **3** | identical ✓ |

Every cell matches the report. Two consequences of that:

- **F2 stands.** 37% of a junk pool clears the shipped `0.70` floor — against the
  README's documented 12%. The floor still works in *direction* (0.793 nonsense
  ceiling below every real query's 0.84–0.91 top band) but on ~0.05 of margin.
- **F2's proposed remedy also reproduces:** at `0.75` the nonsense pool collapses
  to 3/38 — but so does the paraphrase query (80 → 21), which is the cost the
  README already warns about.

**F4 reproduced.** The loose paraphrase still tops out at *"Beyond Words"*
(0.900) — higher than any correct answer on the three well-phrased queries — with
no attention/NMT paper anywhere in the top five.

---

## Guards and honesty fields, verified live

| Check | Evidence this run |
|---|---|
| Downloads are real PDFs | All 5 L1 files start `%PDF-1.4` / `%PDF-1.5`; 190 KB – 2.8 MB |
| No partial-file residue | Zero `.part` files in any scratch dir |
| `BLOCK` writes nothing | `val_block/` contains **only** `manifest.jsonl` |
| `BLOCK` names the hosts | `no allowed host among: arxiv.org, www.aclweb.org` |
| Manifest contract | `resolution` 7 keys, `paper` 14, `run` 10; `query` holds the **input description**, not a result title; abstracts on 100% of records |
| Unverified-confidence discount | Blocked record: `similarity` 0.841 → `confidence` 0.589 = exactly 0.7× |
| **F5 — S2 down** | `errors: {semantic_scholar: 429}` on **every** live run. Still a four-source tool without a key; every run completed anyway |
| **F6 — cap-bound retrieval** | `truncated_sources: [arxiv, crossref, europepmc, openalex]` — all four, on every real-query run this session (one query, three runs), at default `--per-source 20`. The nonsense query truncated only arxiv+crossref |
| **L8 — agent tool surface** | `FIND_PAPERS_BY_DESCRIPTION_TOOL` imports, JSON-serializable, `name`/`description`/`input_schema` present; props `description`/`top_k`/`min_similarity`, `description` required. Text states results are **NOT verified** to be any specific paper and that an empty list ≠ nothing exists; points at `find_paper_pdf` |
| **L6 — expansion widens the pool** | 4 on-topic variants generated; pool **75 → 281 (3.7×)** |
| **F3 — README undercounts** | `test_selection.py` reports 27, README still says 24 |

---

## Deltas vs 2026-08-08

None change a verdict. All four sit in the retrieval layer, which the report
itself documents as nondeterministic.

| | 2026-08-08 | 2026-08-13 | Reading |
|---|---|---|---|
| L1 wall time | 3 m 02 s | **2 m 12 s** | Network variance |
| L1 pool with expansion | 285 | **281** | Live retrieval jitter; 5th-place sim 0.832 → 0.831 |
| L2 idempotency | 4 `HAVE` + 1 `OK` | **5 `HAVE`** | Cleaner — the one paper that missed the cache last time didn't |
| L3 score band | 0.763–0.717 | 0.763–**0.747** | Tail jitter; live L3 pool was 90 here vs the frozen 38. Both titles F1 named reproduce at identical scores |
| `verified_window` on L4 | (not recorded) | **45** vs 15 elsewhere | Explained by `_select` (`semantic_resolver.py:839`): batch = `max(k, 3k)`, widening until `k` fetchable papers are found, capped at `VERIFY_WINDOW_CAP = 45`. L1/L3 found 5 in the first 15-wide batch and stopped; L4 could never find any (only zenodo allowed) so it widened to the cap. Documented behaviour, not a delta in the code |

---

## Still not validated

Unchanged from the report, and this run adds nothing to any of it:

- **No ground truth.** Nothing here measures whether the returned papers are the
  *right* papers. Every check above is a distribution, a guard, or a reproduction.
- **`FAIL` status / alternate-host fallback** — no download failed again, so the
  multi-host retry loop still never executed.
- `--any-host`, `--workers` > 2, `--variants` ≠ 4, `--per-source` ≠ 20,
  `--model` ≠ specter, batch `--from-file` beyond 5.
- Single configuration: one OS, one Python, one embedding model, one expander.
- Not a security audit.

---

## Verdict

The tool is stable across five days: the guards hold, the parsers hold, the
manifest keeps its contract, dedup and idempotency work, degraded sources don't
stop a run, and every self-reported field (`truncated_sources`, `errors`,
`similarity_rank`, `verified_window`) told the truth about the run that produced it.

The one bad case is the one the report already named: **a query with no answer
returned five confident PDFs, and the output gave no signal of failure.** That
is not a regression — it reproduced exactly, and it is a property of semantic
search, not a bug. It means the README's advice is mandatory, not optional:
**read the abstract and the `similarity` in the manifest before citing anything.**
