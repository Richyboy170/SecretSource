# Validation record

What has been checked, in which cases, and what has not. Last run **2026-08-08** against the live scholarly APIs — Python 3.14.6, httpx 0.28.1, Windows 11.

| | |
|---|---|
| Offline checks | **52 / 52 pass** — `py validate_offline.py` |
| CLI error paths | **3 / 3 pass** |
| Live end-to-end cases | **9 run**, all five statuses exercised |
| Defects found | 4 — **all fixed and regression-tested** |

Re-run the offline half in under a second with `py validate_offline.py`. The live half is re-run by hand with the commands in [Live cases](#live-cases); it needs network and takes about a minute.

---

## Live cases

Every live run wrote to a scratch directory, not to `output_pdf/`.

| # | Command | Status | Result | Verdict |
|---|---|---|---|---|
| 1 | `py fetch_papers.py "Attention Is All You Need"` | `OK` | `arxiv.org/pdf/1706.03762` → 2,215,244 B · conf 1.0 · openalex · 12.8 s | pass |
| 2 | `py fetch_papers.py --doi 10.1038/sdata.2016.18` | `OK` | `nature.com/articles/sdata201618.pdf` → 254,899 B · "The FAIR Guiding Principles…" · conf 1.0 · 7.7 s | pass |
| 3 | `py fetch_papers.py --from-file batch.txt` (4 papers) | — | `HAVE` 1 · `OK` 1 · `FAIL` 1 · `MISS` 1 in 18.9 s | pass |
| 3a | `doi:10.1371/journal.pmed.0020124` | `OK` | `journals.plos.org` → 255,629 B · "Why Most Published Research Findings Are False" · conf 1.0 | correct paper |
| 3b | `Deep learning \| 10.1038/nature14539` | `FAIL` | Selected `hal.science/hal-04206682`, DOI matches, not a PDF → *"response is not a PDF"* | **correct refusal** — see [F1](#defects-found-and-fixed) |
| 3c | `"A paper title that does not exist anywhere at all"` | `MISS` | `pdf_url: null` · conf 0.0 · nothing written | correct negative |
| 4 | `--sources <zenodo only>` on case 1 | `BLOCK` | *"no allowed host among: arxiv.org, langtaosha.org.cn"* · nothing written | allowlist held |
| 5 | `py paper_resolver.py` | exit 0 | 3 resolutions, no downloads, 21.8 s | runs |
| 6–8 | no args / `--sources nope.txt` / `--from-file nope.txt` | exit 1 | usage or a named-file error on stderr | pass |

### What case 3b proves

This is the case that exposed the worst defect, so it is the one worth re-running after any change to matching or ranking.

`Deep learning` is a generic title and `10.1038/nature14539` (LeCun, Bengio & Hinton, *Nature* 2015) is paywalled. The correct outcome is a refusal, because no open-access copy of *that* paper is reachable. Before the fix the tool returned `OK` and downloaded arXiv 1807.07987 — *Deep Learning* by Polson & Sokolov, an unrelated statistics review — under the filename `deep-learning_10-1038-nature14539.pdf`.

Grade it strictly:

- `FAIL` or `BLOCK` → **pass**
- `OK` on a copy whose `doi` is `10.1038/nature14539` → **pass**
- `OK` on any arXiv URL → **still broken**

---

## Offline cases

`py validate_offline.py`, 52 assertions, no network — up from 29 before the fixes. Cases tagged `[F1]`–`[F4]` encode the post-fix contract for each defect and must not be relaxed.

Three of the original 29 expectations changed by design: the two `identity_score` cases that now require `doi_keyed`, and the `slugify` case whose filename changed. Each is paired with a sibling asserting the rejecting branch. The harness has not been run against the pre-fix code — several assertions reference symbols that did not exist then.

| Group | Covers |
|---|---|
| Allowlist parsing | `www.` normalisation, comments, blank lines |
| Allowlist guard | Exact host, subdomain, `www.`, PMC ftp host — and rejection of `arxiv.org.evil.com`, `notarxiv.org`, unlisted hosts |
| Batch grammar | All four `--from-file` forms, comments, blanks |
| Title matching | Normalisation, exact match, index truncation, and rejection of prepended-word impostors `[F2]` |
| Identity scoring | DOI match, DOI conflict, DOI-keyed silence, and rejection of title-search hits against DOI-pinned queries `[F1]` |
| Ranking | Identity outranks fetchability; verification breaks ties only within a tier `[F1]` |
| Copy identity | arXiv DOI derived from URL, work DOI preserved separately `[F3]` |
| Filenames | Copy DOI used, requested DOI never borrowed `[F1]`, registrant prefix survives truncation `[F4]` |
| Tool surface | Schema shape, resolution keys |

---

## Defects found and fixed

All four were found by this validation on 2026-08-08 and fixed the same day. None had a regression test before; all four do now.

### F1 — critical: a DOI-pinned query was satisfied by a candidate with no DOI

Asking for *Deep learning* by **title and DOI together** downloaded a different paper at confidence 1.0. Asking for the same DOI **alone** resolved correctly — supplying more identity produced a worse answer.

Three behaviours compounded:

1. `identity_score()` compared DOIs only when the *candidate* carried one. A title-search hit with `doi=None` skipped the DOI check entirely and fell through to title-only scoring, where the generic title "Deep Learning" scored a perfect 1.0.
2. Ranking on `confidence` let the 0.7 unverified discount overrule identity, so the impostor (verified, 1.0) outranked the genuine *Nature* record (paywalled, 0.7).
3. `slugify()` fell back to the *requested* DOI when the winner had none, stamping `10.1038/nature14539` onto a file that never matched it.

**Fixed** by `Candidate.doi_keyed` (true only for records fetched by DOI lookup; a title-search hit scores 0.0 against a DOI-pinned query), `pick_best()` (identity first, fetchability as tie-break only), and `slugify()` (only the matched copy's own DOI may name a file).

### F2 — high: the containment rule scored a different paper at 0.95

`title_similarity()` returned a flat 0.95 whenever *either* title contained the other, so a candidate that added words scored as a match. Querying "Attention Is All You Need" put **"Not All Attention Is All You Need"** and **"Tensor Product Attention Is All You Need"** in the pool at 0.95.

**Fixed** — the bonus now applies only when the found title is the shorter one, which is the index-truncation case the rule exists for. Prepended words are held under the threshold. Both impostors are gone from the pool.

*Not fully solved:* titles differing by a negation are not separable by string similarity alone. The prepend rule catches the observed cases; closing the general problem needs author or year metadata.

### F3 — the predatory-mirror DOI reached the manifest and the filename

Confirmed the behaviour the README previously listed as unfixed: the correct arXiv PDF was filed under `10.65215/2q58a426`, a junk DOI a predatory site attached to OpenAlex work `W2626778328`.

**Fixed** — `location_doi()` derives the copy's own identity from the URL where the host mints one. The manifest now reports both:

```json
"doi":      "10.48550/arXiv.1706.03762",
"work_doi": "10.65215/2q58a426"
```

> **Side effect:** filenames changed. `attention-is-all-you-need_10-65215-2q58a426.pdf` already in `output_pdf/` no longer matches the name the tool now generates, so that paper re-downloads once as `attention-is-all-you-need_10-48550-arxiv-1706-03762.pdf`. The old file was left in place; delete it when convenient.

### F4 — minor: the DOI slug kept its tail instead of its head

`[-24:]` cut the registrant prefix off the front, leaving `..._-1038-s41598-025-25616-x.pdf` — the `10.` gone and the DOI unrecoverable from the name. **Fixed** by keeping the head.

---

## Not covered

Be explicit about this — it is the part a reader will otherwise assume.

- **`--any-host`** — never run. README §6 notes a mirror and the publisher tie at 1.0 with insertion order deciding; the identity-first ranking does not break that tie, so the caveat stands.
- **Sustained throttling** — Semantic Scholar's `429`/`404` responses were absorbed as designed during these runs, but a full multi-source outage and the retry budget's exhaustion path were not simulated.
- **Post-redirect allowlist re-check** — verified by reading and by unit checks. No live redirect off the allowlist was observed, so that branch has not run against a real server.
- **Concurrency above 4 workers** — the default was used throughout.
- **PMC `.tar.gz`-only records** — still unreachable, unchanged (README §3).
- **`MISS` escalation** — still not built (README §4).
- **Response-shape drift** — every source's parsing is exercised only against whatever the API returned on the day. There are no recorded fixtures, so an API changing its JSON shape breaks the tool silently until the next live run.
