# skills

Reusable skills/workflows shared across the agents in [`agents/`](../agents/README.md) — the parts of the pipeline that are common building blocks (e.g. parsing a PDF, scoring evidence, formatting the final report) rather than a single agent's judgment call.

See the [pipeline diagram](../docs/project_pipeline.png) in [`docs/`](../docs/README.md) for where these fit: candidates include the parse/quality-gate step (GROBID + Docling), the retrieval step (BM25 + ColBERTv2, ColPali/ColQwen), and report generation.

**Status:** scaffold — no skills are implemented here yet. This README will be updated with usage instructions as each one lands.
