"""Expand a free-text description into several retrieval queries with a local LLM.

Retrieval breadth is the quality ceiling of this whole tool: no amount of
embedding rerank can recover a paper the scholarly APIs never returned. One small
local model turns a single description into several differently-worded queries —
key phrases, domain synonyms, a title-shaped paraphrase — so the fan-out covers
more of the literature than one phrasing would.

Entirely optional, and deliberately unable to break a run. Every failure path
(Ollama not running, model not pulled, timeout, invalid JSON, valid JSON of the
wrong shape) returns just the original description plus a reason string.

    py query_expander.py "protein structure prediction with transformers"
"""

from __future__ import annotations

import asyncio
import json
import re

import httpx

DEFAULT_MODEL = "qwen3:1.7b"
DEFAULT_URL = "http://localhost:11434"
DEFAULT_VARIANTS = 4

PROMPT = """You rewrite a research topic into search queries for academic databases \
(OpenAlex, Semantic Scholar, Europe PMC, arXiv, Crossref).

Topic: {description}

Write {n} alternative search queries for this topic. Make them genuinely different \
from each other and from the topic as written:
- one using the standard technical terminology of the field
- one using synonyms or an alternative name for the same idea
- one phrased like the title of a paper on the topic
- one narrower, naming a specific method, model, or subproblem

Write each query as ordinary English words separated by spaces, 3-12 words, exactly \
as you would type it into a search box. Never use underscores, snake_case, camelCase, \
or run words together. No boolean operators, no quotes, no field prefixes, no \
numbering, no explanation.

Good: "attention mechanism sequence to sequence models"
Bad: "attention_mechanism_seq2seq" or "AttentionMechanismSeq2Seq"

Reply with JSON only, in exactly this form:
{{"queries": ["...", "...", "...", "..."]}}"""


def _parse(payload: str, limit: int) -> tuple[list[str], str | None]:
    """Pull the query list out of a model response.

    Ollama's `format: "json"` guarantees syntactically valid JSON — it does not
    guarantee the shape asked for. A bare list, or an object under a different
    key, is valid JSON that would crash a naive parser, so a shape mismatch is
    treated as an ordinary expansion failure rather than an exception.
    """
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, TypeError) as exc:
        return [], f"unparseable response: {exc}"

    if isinstance(data, dict):
        # Accept the documented key, then any single list value — small models
        # reach for "search_queries" or "results" often enough to be worth it.
        found = data.get("queries")
        if found is None:
            lists = [v for v in data.values() if isinstance(v, list)]
            found = lists[0] if len(lists) == 1 else None
    elif isinstance(data, list):
        found = data  # lenient alias for {"queries": [...]}
    else:
        found = None

    if not isinstance(found, list):
        return [], f"unexpected JSON shape: {type(data).__name__}"

    queries = [_normalize(q) for q in found if isinstance(q, str) and q.strip()]
    queries = [q for q in queries if q]
    if not queries:
        return [], "model returned no usable queries"
    return queries[:limit], None


def _normalize(query: str) -> str:
    """Force a query into plain space-separated words.

    Small models reach for snake_case and camelCase identifiers even when told
    not to — qwen3:1.7b returned "neural_networks_focus_selection_input_sequence"
    against the earlier prompt. Underscores and glued words are dead weight in a
    scholarly-API query, so the prompt asks for prose and this enforces it.
    """
    text = re.sub(r"[_\-/]+", " ", query.strip())
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)  # split camelCase
    text = re.sub(r'["\']', " ", text)
    return re.sub(r"\s+", " ", text).strip()


async def expand(
    description: str,
    *,
    model: str = DEFAULT_MODEL,
    url: str = DEFAULT_URL,
    variants: int = DEFAULT_VARIANTS,
    timeout: float = 90.0,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[str], str | None]:
    """Return (queries, error). queries[0] is always the original description."""
    description = description.strip()
    if not description or variants <= 0:
        return [description], None

    owned = client is None
    client = client or httpx.AsyncClient(timeout=timeout)
    try:
        response = await client.post(
            f"{url.rstrip('/')}/api/generate",
            json={
                "model": model,
                "prompt": PROMPT.format(description=description, n=variants),
                "stream": False,
                "format": "json",
                # Qwen3 emits reasoning traces by default, which land in front of
                # the JSON body and break the parse. Harmless on models that
                # ignore the field.
                "think": False,
                "options": {"temperature": 0.7},
            },
            timeout=timeout,
        )
        response.raise_for_status()
        queries, error = _parse(response.json().get("response", ""), variants)
    except httpx.HTTPStatusError as exc:
        detail = "model not pulled? run: ollama pull " + model if exc.response.status_code == 404 else ""
        queries, error = [], f"ollama HTTP {exc.response.status_code} {detail}".strip()
    except httpx.TransportError as exc:
        queries, error = [], f"ollama unreachable at {url} ({type(exc).__name__}) — is it running?"
    except Exception as exc:  # never let the expander take down a run
        queries, error = [], f"{type(exc).__name__}: {exc}"
    finally:
        if owned:
            await client.aclose()

    # The description always leads. Variants that merely restate it add API calls
    # without adding coverage, so drop case-insensitive duplicates.
    seen = {description.lower()}
    out = [description]
    for q in queries:
        if q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out, error


if __name__ == "__main__":
    import sys

    topic = " ".join(sys.argv[1:]) or "transformer models for protein structure prediction"

    async def main():
        queries, error = await expand(topic)
        if error:
            print(f"expansion unavailable: {error}")
            print("falling back to the description alone\n")
        for i, q in enumerate(queries):
            print(f"  [{i}] {q}")

    asyncio.run(main())
