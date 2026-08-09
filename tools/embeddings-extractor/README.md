# embeddings-extractor

**Pipeline stage 4 — Evidence flow, "Retrieval" node.**

Searches *inside* research PDFs that have already been acquired. Each page is
rendered to an image, embedded with a ColPali-family visual model into a set of
patch vectors, and stored in [Qdrant](https://qdrant.tech). A free-text
description is then ranked against those pages with MaxSim late interaction —
reporting not just which page matched, but which **region** of that page carried
the match and what text is printed there.

This is the other half of stage 4 from [`source-tool/`](../source-tool/README.md)
and [`source-semantic-tool/`](../source-semantic-tool/README.md). Those two find
*which papers to fetch*; this one finds *where in a fetched paper the evidence
is*. It reads their `output_pdf/` folders directly, including the description
they recorded in `manifest.jsonl`.

```
description ──► ColPali query encoder ──► t x 128 token vectors
                                                    │
PDF ──► page images ──► ColPali page encoder ──► n x 128 patch vectors ──► Qdrant
                                                    │
                                            MaxSim (late interaction)
                                                    ▼
                        ranked pages + the region of each page that matched
```

---

## Setup

### 1. Python environment — must be its own venv

```powershell
cd tools\embeddings-extractor
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Do not install these globally.** `colpali-engine` pins `torch<2.12`, so a
global install downgrades torch 2.13.0 → 2.11.0 — and 2.13.0 is what
[`source-semantic-tool/requirements.txt`](../source-semantic-tool/requirements.txt)
documents itself as verified against. Separate venvs mean neither tool can break
the other. Verified on this machine:

| | this tool (`.venv`) | source-semantic-tool (global `py -3.14`) |
|---|---|---|
| torch | 2.11.0+cpu | 2.13.0+cpu |

Resolved on Python 3.14 / win_amd64, 2026-08-08: `colpali-engine 0.3.17`,
`qdrant-client 1.19.0`, `pymupdf 1.28.2`, `pillow 12.3.0`, `transformers 5.14.1`.

Poppler is **not** required — pages are rasterized with PyMuPDF, which ships as a
self-contained wheel. There is nothing to put on `PATH`.

### 2. Qdrant

Either run a server:

```powershell
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 `
  -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

…or skip Docker entirely and use the embedded store, which persists to a local
directory and needs no server:

```powershell
.\.venv\Scripts\python.exe extract_embeddings.py index --pdf-dir ... --qdrant-path .\qdrant_data
```

> **Docker on Windows 11 Home** needs the WSL2 backend. If `docker` is not
> installed: `wsl --install` (reboot), then
> `winget install -e --id Docker.DockerDesktop` (reboot), then accept the licence
> on first launch. Both steps need an administrator prompt. Use `--qdrant-path`
> until that is done — every command below works either way.
>
> **The server path is currently UNVERIFIED.** Everything below was validated
> against the embedded store, because no Docker exists on this machine. The
> collection config and the client calls are the same either way, but one line is
> worth checking first — `vector_store.search` unwraps the returned vectors with
> `raw[VECTOR_NAME] if isinstance(raw, dict) else raw`. Embedded mode returns a
> dict there. If a server returns a different shape, ranking will still look
> correct while **region attribution silently returns nothing**. After starting a
> server, run one `search` and confirm regions are still reported.

---

## Usage

### Index

Every command takes either `--qdrant-path <dir>` (embedded, no server) or
`--qdrant-url <url>` (server, the default `http://localhost:6333`). The examples
below use the embedded store so they work without Docker; drop the flag once a
server is running. If neither is reachable the tool says so and exits 2 rather
than throwing a connection traceback.

```powershell
# a whole folder the source tools filled
.\.venv\Scripts\python.exe extract_embeddings.py index --pdf-dir ..\source-semantic-tool\output_pdf --qdrant-path .\qdrant_data

# works the same on the citation-based tool's output
.\.venv\Scripts\python.exe extract_embeddings.py index --pdf-dir ..\source-tool\output_pdf --qdrant-path .\qdrant_data

# specific files, first two pages only
.\.venv\Scripts\python.exe extract_embeddings.py index paper.pdf --max-pages 2 --qdrant-path .\qdrant_data
```

Titles and DOIs come from the `manifest.jsonl` beside the PDFs, so results are
readable rather than a list of filename slugs.

Indexing is **resumable and idempotent**. A page's point id is derived from
`(pdf content hash, page number, model)`, so re-running skips pages already
stored and re-indexing the same paper under a different filename is a no-op.
`--reindex` forces re-embedding.

### Search

```powershell
# explicit description
.\.venv\Scripts\python.exe extract_embeddings.py search "retrieval augmented generation for open-domain question answering" --qdrant-path .\qdrant_data

# reuse the description the source tool recorded in its manifest
.\.venv\Scripts\python.exe extract_embeddings.py search --pdf-dir ..\source-semantic-tool\output_pdf --qdrant-path .\qdrant_data
```

Real output, against a store holding 24 pages from four unrelated papers:

```
ranking pages against: "retrieval augmented generation for open-domain question answering"
scores are MaxSim sums, not cosines - comparable only within this query

RANK 1  score 15.53
  paper: Generation-Augmented Retrieval for Open-domain Question Answering
  file:  generation-augmented-retrieval-for-open-domain-question-answering_ff19279b9a.pdf
  page 1 of 12
  top region: x0=298 y0=26 x1=422 y1=105  (15% of score)
  text near region:
    Generation-Augmented Retrieval for Open-Domain Question Answering Yuning
    Mao1*, Pengcheng He2, Xiaodong Liu3, Yelong Shen2,

RANK 2  score 13.29
  ...
  page 3 of 12
  top region: x0=99 y0=737 x1=248 y1=842  (20% of score)
  text near region:
    trieve more relevant passages in terms of both quan- tity and quality.
    For the task of OpenQA where the query is a question, we take the
    following three
```

`--json` emits the same records for the stage-5 verifier agents; `search_pages()`
and `SEARCH_PAGES_TOOL` in `extract_embeddings.py` are the programmatic form.

A manifest can hold **several** descriptions — `source-semantic-tool` appends to
one file across runs, and `output_pdf-o/manifest.jsonl` in this repo already
spans four unrelated topics. When it does, the tool prints the list and exits
non-zero rather than guessing; pass a description explicitly or `--query-index N`.

### Probe

```powershell
.\.venv\Scripts\python.exe extract_embeddings.py probe paper.pdf
```

Reports the model's token layout for one page — vector count, reconstructed grid,
cell size in points, and whether region attribution is valid. **Run this after
changing `--model`.** See *Region attribution* below for why.

```
OK: region attribution is valid. 32x24 grid (768 cells) over a 595x842 pt page
= 24.8x26.3 pt per cell.
875 vectors/page, 438 KB/page stored.
```

---

## What the scores mean

**MaxSim scores are not cosines.** For each query token, take its best-matching
patch on the page; sum over tokens. That sum is unbounded and scales with query
length — roughly 0–20 for a 15-token query, a different range for a 5-token one.

Two consequences:

- Scores are comparable **only within one query**. The ranking is the answer; the
  absolute number is not, and there is no threshold to compare it against.
- `source-semantic-tool`'s `MIN_SIMILARITY = 0.70` floor **does not transfer**.
  That constant was calibrated for SPECTER cosines over a 363-paper pool. As that
  tool's `embedder.py` puts it, *"cosines are not comparable across models"* —
  and these are not even cosines.

The description is likewise **re-encoded**, never carried over as a vector.
`source-semantic-tool` embeds it with SPECTER into one 768-dim vector; this tool
embeds it into a sequence of 128-dim token vectors. The seam between the two
tools is the description *string*.

To sanity-check an index, run a query you know is nonsense — the control from
`semantic_resolver.py`'s smoke test works well. Since MaxSim has no zero point,
that comparison is how you tell a working index from a broken one. Measured on
the 24-page four-topic store (2026-08-08):

| query | top-5 scores | top-5 papers |
|---|---|---|
| retrieval augmented generation for open-domain QA | 15.5 → 12.2 | all 5 the RAG paper |
| collaborative accessible digital musical instrument | 16.0 → 11.8 | all 5 the LoopBoxes paper |
| graph neural networks for molecular property prediction | 12.0 → 10.1 | all 5 the molecular GNN paper |
| *quantum tunneling in medieval bookbinding adhesives* | **8.8 → 8.3** | **scattered across 3 papers** |

A real query concentrates on one paper and spreads ~4 points top to bottom; the
control flattens to half a point and picks no cluster at all.

---

## Region attribution

Qdrant returns a page's score but not the arithmetic behind it. So the top-k
pages' patch vectors are pulled back and MaxSim is recomputed locally, which
recovers the winning patch for each query token. Those patches are cells of a
grid over the page image, and the grid maps to a rectangle in PDF points, and the
rectangle maps to the words printed there.

`share` is the fraction of *that page's* score the region carried. It is
attribution, not confidence — a page that barely matched still has a
highest-contributing region.

**The grid is not just "the model's patches", and assuming so gives wrong boxes.**
Measured on `colSmol-500M` with a 1241×1754 page: 875 output vectors, of which
832 are image tokens — in **13 non-contiguous runs of 64**, because Idefics3
splits the page into tiles (`pixel_values` came back as `(1, 13, 3, 512, 512)`).
Each run is preceded by a `<row_i_col_j>` marker; here a 4×3 tile arrangement
plus one downsampled whole-page view after `<global-img>`. Meanwhile
`processor.get_n_patches()` reports 147×104 = 15288, the raw pre-merge patch
count, which matches nothing.

Reading those markers reassembles a real grid — and a finer one than a single
tile would give: 12 tiles × 8×8 = a **32×24 grid**, about 25×26 pt per cell, or
one line of text on A4. The 64 global-view vectors are excluded, since they
describe the whole page and carry no location. The mapping is exact rather than
approximate because the tiles are a complete rectangular cover and the page is
*resized* into it rather than padded — confirmed by per-tile pixel standard
deviation, where a padded tile would be constant and none was.

`probe` verifies all of this per model. When a layout cannot be reconciled with
any grid, `grid_ok` is false and the tool reports the page **without** a region
rather than inventing one. Page ranking is unaffected either way.

Two further details that make snippets readable rather than technically correct:

- **The heat map is blurred before regions are found.** A query contributes at
  most one cell per token — maybe 18 — scattered over 768 cells, so connected
  components on the raw map are nearly all singletons. Before the 3×3 blur, every
  region came back one cell wide and snippets read `"gen- generation-"`. Region
  *weights* are still summed from the unblurred map, so widening the search
  cannot inflate a reported share.
- **Snippets are whole text lines**, not the words strictly inside the box. A
  cell is ~25 pt across — four or five characters — so word-level filtering
  returns fragments. Taking lines the region touches returns the sentence, and
  respects two-column layouts since a line's bbox stops at its own column.

Two geometry details that are easy to get wrong and fail silently:

- Grid cell 0 is the **top-left**; rows run down the page, matching both the
  image raster order and PDF text coordinates.
- On a rotated page, `page.rect` is rotated but `get_text("words")` returns
  **unrotated** coordinates. Measured on a 400×600 page at `/Rotate 90`, a word
  came back at y=480..505 — outside the 400-point height it displays in.
  `page_words()` multiplies by `page.rotation_matrix` to correct this. It matters
  because rotated landscape pages are exactly the wide figure and table pages a
  visual retriever surfaces.

---

## Models

| `--model` | Backbone | CPU cost / page |
|---|---|---|
| `vidore/colSmol-500M` *(default)* | SmolVLM-500M | **26 s** (measured) |
| `vidore/colSmol-256M` | SmolVLM-256M | ~10 s (estimated) |
| `vidore/colqwen2.5-v0.2` | Qwen2.5-VL-3B | minutes (estimated) |

**Measured 2026-08-08** on this CPU-only machine, `colSmol-500M`, A4 pages at
150 DPI: model load 14.1 s once per run, then 24.4–29.0 s per page (mean 26.0 s
over 4 pages). A larger page producing 1139 vectors instead of 875 costs
proportionally more — a 9-page run over `source-tool/output_pdf` averaged ~34 s.
Only the default is measured; the other two rows are scaled estimates, so treat
them as such.

Budget roughly **7 minutes per 15-page paper** on CPU. It is a one-time cost —
embeddings persist, and `index` skips pages already stored. **Queries are fast
regardless**: encoding a description took 0.23 s, and search is a vector scan.

There is no GPU here (`torch.cuda.is_available()` is `False`; the adapter is an
Intel UHD integrated GPU), which is why the 500M model is the default. On a CUDA
box, `--model vidore/colqwen2.5-v0.2` scores better and needs no code change —
the model loads to `cuda:0` in bf16 automatically.

Vectors from two models are **not** comparable, so the model is part of every
point's identity. Indexing the same corpus under a second model adds points
beside the first rather than replacing them.

---

## Storage and scale

Per page: `n_vectors × 128 dims × 4 bytes`. `n_vectors` depends on both the model
and the page size, since tiling scales with the page: measured with
`colSmol-500M`, 875 vectors (438 KB) for an A4 page and 1139 (570 KB) for a
larger one, **averaging 504 KB/page** over a 24-page mixed run. The `index`
command prints the running average and `probe` reports it for a single page —
prefer those numbers to any estimate here.

MaxSim compares *sets* of vectors, which a proximity graph over individual
vectors cannot represent, so the collection is created with
`hnsw_config=HnswConfigDiff(m=0)` and **every search is a full scan**. That is
correct and fast enough at the few-hundred-page scale this tool targets. Past
roughly 5k pages, the fix is Qdrant's two-stage pattern — a mean-pooled single
vector for `prefetch`, the full multivector for rerank. Not implemented; the
named vector `colpali` leaves room to add one without a migration.

---

## Tests

```powershell
.\.venv\Scripts\python.exe test_extractor.py
```

No pytest, no network, no model download, no Qdrant server — matching
[`source-semantic-tool/test_selection.py`](../source-semantic-tool/test_selection.py).
The one PDF involved is built in memory by the tests themselves.

77 checks covering the arithmetic that turns a score into a claim about a
location on a page: MaxSim and its argmaxes, tile-run reassembly into a grid
(including the transpose and incomplete-cover cases it must refuse), the
row-major grid→points mapping and its corners, heat-map blurring and component
splitting, exclusion of query tokens that matched non-image vectors, line-level
snippet extraction and column ordering, manifest parsing for both tools'
formats, and the rotated-page coordinate correction.

Two bugs these caught during development, both of which would have produced
confident wrong citations rather than errors: `get_text("words")` returning
unrotated coordinates on a rotated page, and a heat threshold computed over only
non-zero cells, which discarded the weaker of two genuine hot spots.

---

## Limitations

1. **Page granularity, not sections.** ColPali's unit is a page image. Region
   attribution narrows that to a rectangle, but nothing here parses a document
   into `Abstract` / `3.2 Retrieval` structure — that would need GROBID or
   Docling, which the pipeline assigns to the parse stage.
2. **No text retrieval half.** The pipeline specifies BM25 + ColBERTv2 for text
   alongside ColPali/ColQwen for visual. Only the visual half exists here.
3. **Full-scan search.** See *Storage and scale*.
4. **Region attribution is model-dependent.** See above; `probe` tells you.
5. **Scanned PDFs have no text layer**, so a region is reported with an empty
   snippet. Ranking still works — the model reads the image, not the text.
6. **No cross-encoder rerank.** Ranking is the raw MaxSim order.
7. **Column detection is geometric, not a layout parse.** Lines are grouped by
   horizontal overlap, with full-width lines pulled out first so a heading cannot
   bridge two columns. It handles the one- and two-column papers in this corpus;
   it is not a general layout analyser, and a three-column or floating-sidebar
   page may still order oddly.
8. **The Qdrant server path is unverified** — see the note under *Setup*.
9. **Indexing is slow on CPU**: ~26 s/page, so a 15-page paper takes about seven
   minutes. One-time per paper, and queries are unaffected.
