# tools

Standalone tools the SecretSource pipeline depends on. Each tool lives in its own subfolder with its own README — this file is just the index.

| Tool | Pipeline stage | Purpose |
|---|---|---|
| [`source-tool/`](source-tool/README.md) | Stage 4 — Evidence flow ("Source tools" node) | Resolves a citation (title and/or DOI) to a verified open-access PDF via Crossref, OpenAlex, Semantic Scholar, Unpaywall, arXiv, and PubMed/PMC, downloading only from an allowlisted set of hosts. |
| [`source-semantic-tool/`](source-semantic-tool/README.md) | Stage 4 — Evidence flow ("Source tools" node) | Finds open-access papers from a free-text **description** rather than a citation: expands the query with a local LLM, fans out to OpenAlex, Semantic Scholar, Europe PMC, arXiv, and Crossref, ranks candidates by embedding similarity, and downloads the top-k from an allowlisted set of hosts. |
| [`embeddings-extractor/`](embeddings-extractor/README.md) | Stage 4 — Evidence flow ("Retrieval" node) | Searches *inside* the PDFs the source tools acquired: embeds each page with a ColPali-family visual model into multi-vectors, stores them in Qdrant, and ranks pages against a free-text description by MaxSim — reporting which region of which page carried the match, and the text printed there. |

The two source tools are the same pipeline node entered from different directions. Use `source-tool/` when you have a citation and need the paper it names — it verifies identity by DOI or title. Use `source-semantic-tool/` when you only know the topic — it cannot verify identity, and returns nearest matches instead.

`embeddings-extractor/` is the stage that follows both. The source tools decide *which papers to fetch*; it decides *where in a fetched paper the evidence is*, reading their `output_pdf/` folders and the description they recorded in `manifest.jsonl`. It implements the **visual** half of the Retrieval node (ColPali/ColQwen); the text half (BM25 + ColBERTv2) is still unbuilt and remains a candidate for [`skills/`](../skills/README.md).

> **Two different similarity searches, easily confused.** `source-semantic-tool/` ranks *candidate papers* by SPECTER cosine over title+abstract, floored at 0.70. `embeddings-extractor/` ranks *pages* by unbounded MaxSim sums, which are not cosines and have no threshold. The two scores share no scale, and a vector from one is meaningless to the other — the only thing that crosses between them is the description string.

See the [root README](../README.md#pipeline) for how this stage fits into the full pipeline.
