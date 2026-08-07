# tools

Standalone tools the SecretSource pipeline depends on. Each tool lives in its own subfolder with its own README — this file is just the index.

| Tool | Pipeline stage | Purpose |
|---|---|---|
| [`source-tool/`](source-tool/README.md) | Stage 4 — Evidence flow ("Source tools" node) | Resolves a citation (title and/or DOI) to a verified open-access PDF via Crossref, OpenAlex, Semantic Scholar, Unpaywall, arXiv, and PubMed/PMC, downloading only from an allowlisted set of hosts. |
| [`source-semantic-tool/`](source-semantic-tool/README.md) | Stage 4 — Evidence flow ("Source tools" node) | Finds open-access papers from a free-text **description** rather than a citation: expands the query with a local LLM, fans out to OpenAlex, Semantic Scholar, Europe PMC, arXiv, and Crossref, ranks candidates by embedding similarity, and downloads the top-k from an allowlisted set of hosts. |

The two source tools are the same pipeline node entered from different directions. Use `source-tool/` when you have a citation and need the paper it names — it verifies identity by DOI or title. Use `source-semantic-tool/` when you only know the topic — it cannot verify identity, and returns nearest matches instead.

> **Not the "Retrieval" node.** `source-semantic-tool/` also does embedding search, but the pipeline's separate *Retrieval* node (BM25 + ColBERTv2 for text, ColPali/ColQwen for visual) searches *inside* documents already acquired, and is claimed by [`skills/`](../skills/README.md). Neither implements the other.

See the [root README](../README.md#pipeline) for how this stage fits into the full pipeline.
