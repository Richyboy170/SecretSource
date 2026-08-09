"""Read the `manifest.jsonl` that the two source tools write beside their PDFs.

This is the seam between this tool and ../source-tool / ../source-semantic-tool.
Both append one JSON object per attempted download into `output_pdf/manifest.jsonl`,
and both record the query that produced it -- so indexing a folder can pick up
each paper's title and DOI, and searching it can reuse the original description
instead of making you retype it.

The two manifests differ, and this module reads either. source-tool writes five
keys (`query`, `status`, `path`, `detail`, `resolution`); source-semantic-tool
writes those plus `label`, `paper` and `run`. Only `paper` carries a title, so a
source-tool record falls back to the filename slug.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

MANIFEST_NAME = "manifest.jsonl"


@dataclass(frozen=True)
class ManifestEntry:
    """One successfully downloaded PDF, as the source tool described it."""

    path: Path
    query: str
    title: str
    doi: str | None

    def payload(self) -> dict[str, str | None]:
        """The subset carried into the vector store, for readable results."""
        return {"title": self.title, "doi": self.doi, "source_query": self.query}


def read_manifest(directory: Path) -> dict[Path, ManifestEntry]:
    """Map absolute PDF path -> entry, for every `status == "downloaded"` record.

    Keyed by resolved path so a caller that globbed the directory can look up
    what it found. Records for skipped, blocked, failed or unresolved downloads
    name no file on disk and are dropped.

    A missing or malformed manifest is not an error: the PDFs are still
    indexable, just without titles. Anything unparseable is skipped line by line
    so one bad append cannot hide the rest of the file.
    """
    path = directory / MANIFEST_NAME
    entries: dict[Path, ManifestEntry] = {}
    if not path.is_file():
        return entries

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") != "downloaded":
                continue
            raw_path = record.get("path")
            if not raw_path:
                continue

            # Manifests store a Windows path relative to the tool's own cwd
            # ("output_pdf\\foo.pdf"). Resolve against the directory we were
            # handed, so a manifest read from anywhere still finds its files.
            pdf_path = Path(str(raw_path).replace("\\", "/"))
            resolved = directory / pdf_path.name
            if not resolved.is_file():
                continue

            paper = record.get("paper") or {}
            resolution = record.get("resolution") or {}
            title = (
                paper.get("title")
                or resolution.get("matched_title")
                # source-tool records carry no title; the filename slug is the
                # only thing left, and it was built from the title anyway.
                or resolved.stem.rsplit("_", 1)[0].replace("-", " ")
            )
            entries[resolved.resolve()] = ManifestEntry(
                path=resolved.resolve(),
                query=str(record.get("query") or ""),
                title=str(title),
                doi=paper.get("doi") or resolution.get("doi"),
            )
    return entries


def distinct_queries(entries: dict[Path, ManifestEntry]) -> list[str]:
    """The descriptions represented in a manifest, in first-seen order."""
    seen: list[str] = []
    for entry in entries.values():
        if entry.query and entry.query not in seen:
            seen.append(entry.query)
    return seen


class AmbiguousQuery(Exception):
    """Raised when a manifest holds several descriptions and none was chosen.

    Not hypothetical: source-semantic-tool appends to one manifest across runs,
    and `output_pdf-o/manifest.jsonl` in this repo already spans four unrelated
    topics. Silently picking the first would rank a whole corpus against the
    wrong description and look like the model failing.
    """

    def __init__(self, queries: list[str]):
        self.queries = queries
        super().__init__(f"{len(queries)} distinct descriptions in manifest")


def resolve_query(entries: dict[Path, ManifestEntry], index: int | None = None) -> str:
    """The single description a manifest represents, or raise `AmbiguousQuery`.

    `index` is 1-based, matching what the error message prints back.
    """
    queries = distinct_queries(entries)
    if not queries:
        raise AmbiguousQuery([])
    if index is not None:
        if not 1 <= index <= len(queries):
            raise AmbiguousQuery(queries)
        return queries[index - 1]
    if len(queries) > 1:
        raise AmbiguousQuery(queries)
    return queries[0]
