# agents

Agent definitions for the SecretSource pipeline (see the [pipeline diagram](../docs/project_pipeline.png) in [`docs/`](../docs/README.md)).

Each pipeline stage that involves judgment rather than a deterministic transform is meant to live here as its own agent:

| Agent | Pipeline stage | Job |
|---|---|---|
| Claim extractor | 3 · Claims + citations | Pull atomic, typed, cite-worthy claims out of the parsed document. |
| Evidence verifier (text + visual) | 5 · Specialist review | Judge whether retrieved evidence (text or figures/tables) supports a claim. |
| Quantitative verifier | 5 · Specialist review | Check numeric/statistical claims against the evidence. |
| Validity / overclaim verifier | 5 · Specialist review | Flag claims that overstate what the cited evidence actually shows. |
| Aggregator | 6 · Decide, explain, learn | Fuse the three verifiers' verdicts, calibrate confidence, and route uncertain cases to human review. |

**Status:** scaffold — no agents are implemented here yet. This README will be updated with usage instructions as each agent lands.

Related: [`tools/source-tool/`](../tools/source-tool/README.md) implements the deterministic "Source tools" node (stage 4) these agents rely on for evidence.
