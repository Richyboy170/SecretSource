# tools

Standalone tools the SecretSource pipeline depends on. Each tool lives in its own subfolder with its own README — this file is just the index.

| Tool | Pipeline stage | Purpose |
|---|---|---|
| [`source-tool/`](source-tool/README.md) | Stage 4 — Evidence flow ("Source tools" node) | Resolves a citation (title and/or DOI) to a verified open-access PDF via Crossref, OpenAlex, Semantic Scholar, Unpaywall, arXiv, and PubMed/PMC, downloading only from an allowlisted set of hosts. |

See the [root README](../README.md#pipeline) for how this stage fits into the full pipeline.
