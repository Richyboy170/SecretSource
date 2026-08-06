# docs

Reference material for the SecretSource project. Not code — background reading and the architecture diagram everything else in the repo implements.

| File | What it is |
|---|---|
| `project_pipeline.png` | The end-to-end pipeline diagram: upload → parse/quality gate → claim extraction → citation scope → source resolution → evidence retrieval → 3 verifier agents → aggregator → report → human review (with feedback looping back in). Referenced from the [root README](../README.md#pipeline) — start there for the walkthrough. |
| `AI Engineering Owl Book.pdf` | Background reading on AI engineering practices, kept here for the team to reference while designing the agents in [`agents/`](../agents/README.md). |

If you add a new design doc, spec, or reference PDF, drop it here and add a row to the table above.
