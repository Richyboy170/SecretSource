# Validation report

What has actually been checked about this tool, how, and what has **not**.

**Validated 2026-08-08** on Windows 11 / Python 3.14.6, against the live APIs and
a local Ollama. Result: **127 automated assertions pass (27 + 100), and 9 live
end-to-end cases behave as documented.** Two numbers in the README's calibration
table did not reproduce, and one documented consequence is understated — see
[Findings](#findings).

> This report validates that the tool **does what its README says**. It does not
> establish that the tool returns the *right papers* — that is limitation #1, and
> it is not a bug to be fixed but a property of semantic search. See
> [F1](#f1-a-nonsense-query-downloads-five-confident-looking-pdfs) first.

---

## Environment

| | |
|---|---|
| OS / Python | Windows 11 (10.0.26200) / CPython 3.14.6 |
| `httpx` | 0.28.1 |
| `sentence-transformers` / `torch` | 5.7.0 / 2.13.0+cpu |
| Embedding model | `sentence-transformers/allenai-specter`, 512-token window, cached |
| Query expansion | Ollama, `qwen3:1.7b` (2.0B params, Q4_K_M) — reachable |
| `S2_API_KEY` | **not set** — Semantic Scholar was unavailable throughout (expected, see [F5](#f5-semantic-scholar-was-unavailable-for-every-query)) |
| Allowlist | stock `sources.txt`, 29 hosts |

Live runs wrote to a scratch `--out` directory, so nothing in this report
polluted `output_pdf/` or its manifest.

---

## How to reproduce

```powershell
# 1. Offline suites — deterministic, no network, no torch. Run these first.
py test_selection.py            # selection, Retry-After, probe pacing   -> 27/27
py test_validation.py           # parsing, dedup, download, allowlist    -> 100/100

# 2. Re-derive the score table from the pool THIS report measured. Retrieval is
#    nondeterministic (F2), so the pool is committed rather than re-fetched.
py bench_ranking.py validation_pool.jsonl --models sentence-transformers/allenai-specter

# 2b. Or freeze a fresh pool the same way it was made (the README's own workflow).
py fetch_semantic.py --from-file topics.txt --dry-run --rank-only `
    --min-similarity 0.0 --no-expand --dump-pool pool.jsonl

# 3. Live end-to-end. Use a scratch --out so the real manifest stays clean.
py fetch_semantic.py "attention mechanisms for neural machine translation" -k 5 --out val_out
py fetch_semantic.py "attention mechanisms for neural machine translation" -k 5 --out val_out  # HAVE
py fetch_semantic.py "quantum tunneling effects in medieval manuscript bookbinding adhesives" -k 5 --out val_nonsense
```

The five queries used throughout are the README calibration table's: the three
demo topics in `topics.txt`, plus a loose paraphrase of the first
(*"how computers learn to pay attention to the right words when turning one
language into another"*) and the nonsense control from `semantic_resolver.py`'s
own smoke test (*"quantum tunneling effects in medieval manuscript bookbinding
adhesives"*).

---

## 1. Automated coverage

### `test_selection.py` — 27/27 pass

Pre-existing. Covers top-k selection, `Retry-After` backoff, probe pacing.
*(The README says "24 asserting offline tests"; the file now holds 27 — see
[F3](#f3-the-readme-undercounts-its-own-test-suite).)*

### `test_validation.py` — 100/100 pass

Added by this validation, targeting exactly what limitation #7 named as
untested: **"Retrieval, parsing, dedup and download remain untested."**

| Area | Assertions | What it guards |
|---|---:|---|
| Allowlist parsing | 2 | Full-URL / bare-domain / `www.` forms normalize alike; inline `#` comments stripped |
| Allowlist matching | 7 | Suffix-anchored match. Includes the **lookalike attack** `arxiv.org.example.com` and the no-dot-boundary case `notarxiv.org` |
| Download guard chain | 9 | `%PDF-` sniff rejects an HTML paywall; `MIN_PDF_BYTES` floor; off-allowlist redirect rejected; **nothing — not even a `.part` — reaches disk on failure** |
| OpenAlex parsing | 4 | Inverted-index abstract reconstruction, `doi.org` prefix stripping, every distinct OA location emitted, PDF-less works still contribute |
| Europe PMC parsing | 3 | Only `documentStyle == "pdf"` becomes a `pdf_url`; HTML-only records yield an honest MISS |
| Semantic Scholar parsing | 4 | `doi.org` demoted to `landing_url`; arXiv-id fallback; **`null` entries in `data` skipped, not crashed on** |
| Crossref parsing | 3 | JATS markup stripped before embedding; only `application/pdf` links taken |
| arXiv parsing | 3 | Atom parsing, PDF link chosen by `title="pdf"` not by order, empty-after-cleaning query makes no request |
| Dedup | 8 | **Preprint + published merge into one work despite different DOIs**; longest abstract wins; title-less dropped; distinct titles not merged |
| Text helpers | 9 | Accent folding, punctuation flattening, Lucene-operator stripping, `None`-safety |
| `cosine` | 3 | Identity, orthogonality, zero-vector safety (no `ZeroDivisionError`) |
| `join_paper_text` | 3 | No dangling separator when a title or abstract is missing |
| Floor × penalty coupling | 3 | Asserts the documented product: a bare title needs `raw ≥ 0.824` |
| Confidence discount | 3 | 0.7× when unverified; `similarity` never mutated |
| Fetch-target selection | 3 | With an allowlist, the first *acceptable* URL is probed |
| Manifest schema | 9 | source-tool's keys present on **both** the hit and the miss path; JSON-serializable |
| Filename slugs | 5 | Stable (this is what makes `HAVE` work), URL-hashed, overlong titles truncated |
| Query expander | 13 | Every documented failure path returns `(description, reason)` and **never raises** |
| Rate limiter | 4 | Minimum-interval gate honoured; arXiv keeps its published 1-per-3s |

---

## 2. Live end-to-end cases

| # | Case | Command | Result |
|---|---|---|---|
| L1 | Normal query, expansion on | `"attention mechanisms for neural machine translation" -k 5` | **5/5 `OK`** in 3m02s. Pool 285, `verified_window=15`. All five files real PDFs (190 KB–2.8 MB), correct magic bytes, no `.part` residue |
| L2 | Re-run (idempotency) | same, `--no-expand` | **4 `HAVE` + 1 `OK`** — nothing re-downloaded |
| L3 | Nonsense control | `"quantum tunneling … bookbinding adhesives" -k 5` | **5/5 `OK`** — see [F1](#f1-a-nonsense-query-downloads-five-confident-looking-pdfs) |
| L4 | `BLOCK` path | `--sources <only zenodo.org>` | **3/3 `BLOCK`**, each naming the hosts found. **Zero files written** — only `manifest.jsonl` |
| L5 | Fetchable-first reordering | L1, L2 | Live `(#8 by score)` printed on rank 5; `rank` vs `similarity_rank` both in the manifest |
| L6 | Query expansion | L1 | 4 distinct on-topic variants; pool **75 → 285** (3.8×) vs `--no-expand` |
| L7 | Degraded-source resilience | all runs | Semantic Scholar `429`d on **every** query; all runs completed on the other four |
| L8 | Agent tool surface | import check | Schema well-formed; description states results are **not** identity-verified and that empty ≠ nonexistent; points at `find_paper_pdf` |
| L9 | CLI error handling | no args / bad `--sources` / bad `--from-file` | Each exits with a clear message, no traceback |

### Manifest contract, verified on a live record

All four blocks present and populated: top-level `query` / `status` / `path` /
`detail` / `label`, plus `resolution` (7 keys), `paper` (14 keys), `run` (10
keys). `query` held the **input description**, not a result title. Abstracts
present on all five records.

### SPECTER pairing (model-backed)

`pair_separator` → `'[SEP]'`, joining as `Title[SEP]Abstract`. The tokenizer maps
`[SEP]` to a **single** token id (`102`), confirming the README's claim that any
other spelling would be tokenized as ordinary text. Window: 512.

---

## 3. README claims — evidence and verdict

**Confirmed** (all verified this session):

| Claim | Evidence |
|---|---|
| Suffix-anchored allowlist blocks lookalikes | `test_validation.py`, 7 cases |
| `%PDF-` sniff / redirect re-check / size floor | `test_validation.py`, 9 cases |
| Dedup merges preprint + published on title, not DOI | `test_validation.py` |
| `doi.org` filter, `documentStyle == "pdf"`, JATS stripping | `test_validation.py` |
| `Retry-After` capped at 30 s; long values abandon the source | `test_selection.py` |
| Probe concurrency exactly 3 per host, not serialized across hosts | `test_selection.py` |
| Unauthenticated S2 `429` fails fast; run continues on the rest | L7 — 10+ queries |
| "The five sources overlap far less than you would expect" | **332 distinct papers, only 11 (3.3%) found by >1 source** (README: ~2.8%) |
| Fetchable-first selection reorders, and says so | L5 |
| Manifest is a superset of source-tool's | L1 + 9 offline assertions |
| specter 512-token window; "0–6 papers per query" overrun | **0–5 per query** over the frozen pool |
| SPECTER `[SEP]` is one token | id `102` |
| `MIN_SIMILARITY / TITLE_ONLY_PENALTY = 0.824` | asserted in `test_validation.py` |
| Expansion widens the pool | 75 → 285 |
| Expander never breaks a run | 13 offline assertions |

**Score distribution** — the frozen 332-paper pool, `--no-expand`, specter.
Committed as `validation_pool.jsonl`, so this table can be re-derived rather than
taken on trust:

| Query | max | median | min | survive `0.70` | of those, `title_only` | README max |
|---|---|---|---|---|---|---|
| attention mechanism for NMT | 0.880 | 0.770 | 0.552 | 58 / 75 | 13 | 0.880 ✓ |
| GNN for molecular property prediction | 0.895 | 0.778 | 0.616 | 56 / 65 | 14 | 0.895 ✓ |
| RAG for open-domain QA | 0.912 | 0.806 | 0.667 | 72 / 74 | 11 | 0.907 ≈ |
| loose paraphrase of query 1 | 0.900 | 0.708 | 0.552 | 48 / 80 | 15 | 0.848 — n/a |
| **nonsense control** | **0.793** | 0.682 | 0.599 | **14 / 38** | 3 | 0.777 ≈ |

Maxima reproduce **exactly** on the two queries whose phrasing the README states
verbatim, and within 0.005 on the third.

**The last two rows are not reproductions, and should not be read as
contradictions.** The README never gives the text of its paraphrase, so the one
used here is self-authored — that row is a *new datapoint*, not a re-measurement,
and the 0.052 gap says nothing about the README. The nonsense control is
*assumed* to be `semantic_resolver.py`'s smoke-test string (the tool's own
control); on that assumption the 0.793 vs 0.777 gap is within retrieval noise,
since the pool differed in size too (38 vs 50). What does not reduce to noise is
the **survivor fraction** — see [F2](#f2-the-similarity-floor-filters-a-smaller-fraction-of-the-tail-than-documented).

**Not re-verifiable** — dated observations about third-party behaviour, recorded
as documented and *not* re-checked: the three bare `curl` probes to S2 on
2026-08-07; OpenAlex answering `Retry-After: 36494`; the 15 s completion when
OpenAlex was out of quota; the 2026-08-07 `"best time for playing instruments"`
measurements. These describe API states that cannot be summoned on demand. The
*code paths* they motivated are all covered by `test_selection.py`.

---

## Findings

### F1. A nonsense query downloads five confident-looking PDFs

The clearest live demonstration of limitation #1. At the shipped default floor,
`"quantum tunneling effects in medieval manuscript bookbinding adhesives"`
returned **five `OK` results and wrote five PDFs to disk** — including
*"Searching for Coherent States: From Origins to Quantum Gravity"* (0.763) and
*"Transcribing Medieval Manuscripts for Machine Learning"* (0.754). Each is a
real paper; none answers the query, because the query has no answer.

The README documents this as a *property* ("a tail filter, not an identity
guard"), but frames the consequence as a count of returned results. The
user-visible consequence is a **populated output directory that carries no
signal of failure**: L3 printed the same five `OK` markers and the same
`downloaded=5` summary as L1.

The similarity numbers do differ — 0.850–0.832 on L1 against 0.763–0.717 on L3 —
but nothing marks the lower band as weak, and by limitation #1 nothing could:
cosine is not comparable across queries, so there is no reference value the tool
could compare 0.763 against. The scores are only interpretable next to *another
run*, which is not what a user has in front of them.

Not a defect in the code — it behaves exactly as designed and documented. It is
the single most important thing for a downstream consumer to know, and it argues
for the README's own advice being treated as mandatory: read the abstract and
`similarity` in the manifest before citing anything.

### F2. The similarity floor filters a smaller fraction of the tail than documented

On the nonsense control, at the shipped floor of `0.70`:

| | README | Measured 2026-08-08 |
|---|---|---|
| Survivors at `0.70` | 6 / 50 — **12%** | 14 / 38 — **37%** |
| Best score | 0.777 | 0.793 |
| Margin, nonsense best → real-query top-5 band | 0.07 | ~0.05 |

The **survivor fraction** is the finding: 37% of a junk pool clearing the floor
is three times the documented rate, and it is a ratio, so it does not reduce to
the pools being different sizes. The two score rows are within retrieval noise
and are reported for completeness only.

The floor still works in *direction* — the nonsense pool's ceiling (0.793) stays
below every real query's top-5 band (0.84–0.91) — but it removes less of the tail
than the table implies, and on ~0.05 of margin. Raising the floor to `0.75` would
cut the nonsense pool to 3/38, but the README already documents why that is not
free: the title-only bar would rise to 0.882 and categorically exclude
abstract-less papers.

This supports the README's own warning that the table was measured on five
queries and should not be read as tight.

### F3. The README undercounts its own test suite

Limitation #7 says "**24** asserting offline tests". `test_selection.py` now
reports **27/27**. Documentation drift only.

### F4. The loose paraphrase surfaces no correct answer

On *"how computers learn to pay attention to the right words when turning one
language into another"*, the top three were *"Beyond Words"* (0.900), *"Teaching
natural language to computers"* (0.865) and *"Large Language Models Lack
Understanding of Character Composition of Words"* (0.834). No attention/NMT paper
appears — and the 0.900 top hit outscores every correct answer on the three
well-phrased queries.

This independently reproduces limitation #4 with a different paraphrase than the
README used: **when the description avoids the field's terminology, the tool
fails quietly and confidently.**

### F5. Semantic Scholar was unavailable for every query

Without `S2_API_KEY`, S2 returned `429` on **every one** of the 10+ queries in
this validation. Limitation #3 is accurate, and in practice this is a
**four-source tool** unless a key is set. Every run completed regardless — the
resilience claim holds — but the recall ceiling in this report is a four-source
ceiling.

### F6. All four working sources hit the `--per-source` cap on every real query

`truncated_sources` listed arxiv, crossref, europepmc and openalex on all four
non-nonsense queries at the default `--per-source 20`. Limitation #2 is not
theoretical at default settings: **retrieval is cap-bound, not exhausted.** The
tool reports this honestly rather than capping silently, which is the documented
behaviour — but any recall claim at defaults is a claim about the first 20 hits
per source.

---

## Not validated

Explicitly out of scope for this report — do not read the sections above as
covering these.

**No ground truth.** There is no gold-standard set of correct answers, so
retrieval *recall* and ranking *precision* are unmeasured. Every ranking check
above is a distribution or a reproduction of a documented number, never
"did it find the right paper?" This is the largest remaining gap, and it is the
same one limitation #7 names: ranking is inspectable but still not asserted on.

**Untested code paths:**

- **`FAIL` status and the alternate-host fallback.** No download failed during
  validation, so the multi-host retry loop in `process()` never executed. Its
  guard chain is covered offline; its *loop* is not.
- **`--any-host`, `--workers` above 2, `--variants` other than 4, `--per-source`
  other than 20, `--model` other than specter.**
- **OpenAlex quota exhaustion.** Not reproducible on demand; the `Retry-After`
  cap that handles it is covered offline.
- **Batch `--from-file` beyond 5 descriptions.**

**Single-configuration results.** One OS (Windows), one Python (3.14.6), one
embedding model, one expansion model (`qwen3:1.7b`). The README's cross-model
calibration claims — that `all-MiniLM-L6-v2` truncates 27–53% of papers and that
carrying its `0.35` floor onto specter filters nothing — were **not** re-run
here; only specter was benched. macOS/Linux are untested.

**Expansion quality** was observed helping once (L6, 3.8× pool) but not measured
against outcomes. The README's `qwen3:8b` suggestion remains untested.

**Not a security audit.** The allowlist and download guards are tested against
the failure modes the README names, not against an adversary.

---

## Verdict

The tool behaves as documented on every mechanism this report could exercise:
the guards hold, the parsers are correct, dedup works, degraded sources do not
stop a run, and the manifest keeps its contract. The two reproduction failures
are both in the *calibration table*, both in the conservative direction (the
floor is weaker than advertised), and both consistent with the README's own
caveat that those numbers came from five queries.

The tool's honesty is its strongest validated property — `truncated_sources`,
`errors`, `similarity_rank` and `verified_window` all reported accurately on
live runs, so a run describes its own degradation. Use that. **The tool tells
you what it did; it cannot tell you the answer is right.**
