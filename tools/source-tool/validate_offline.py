"""Offline validation harness — every check that needs no network.

    py validate_offline.py        # exits 0 if all cases pass, 1 otherwise

Covers the pure functions the resolver's correctness rests on: the allowlist guard,
batch-input grammar, title matching, identity scoring, candidate ranking and output
filename construction. The live end-to-end cases are recorded in VALIDATION.md and
are re-run with `py fetch_papers.py`; this file is what you can run in a second
without touching the six scholarly APIs.

Cases marked [F1]-[F4] are regression tests for defects found in the 2026-08-08
validation. They must not be relaxed without re-reading VALIDATION.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from fetch_papers import load_sources, is_allowed, parse_titles, slugify, MIN_PDF_BYTES
from paper_resolver import (FIND_PAPER_PDF_TOOL, TITLE_MATCH_THRESHOLD, Candidate,
                            Resolution, identity_score, location_doi, normalize_title,
                            pick_best, title_similarity)

HERE = Path(__file__).parent
PASSED = FAILED = 0


def check(name: str, got, want) -> None:
    global PASSED, FAILED
    if got == want:
        PASSED += 1
        print(f"  pass  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}\n          got  {got!r}\n          want {want!r}")


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# --------------------------------------------------------------------------- #
section("1. Allowlist parsing — sources.txt")

hosts = load_sources(HERE / "sources.txt")
check("hosts parsed from the shipped allowlist", len(hosts) > 0, True)
check("www. prefix normalized away", "nature.com" in hosts and "www.nature.com" not in hosts, True)
check("comments and blank lines skipped", any(h.startswith("#") for h in hosts), False)

# --------------------------------------------------------------------------- #
section("2. Allowlist guard — suffix-anchored host matching")

H = {"arxiv.org", "ncbi.nlm.nih.gov"}
for url, want, why in [
    ("https://arxiv.org/pdf/1706.03762",       True,  "exact host"),
    ("https://export.arxiv.org/pdf/x.pdf",     True,  "subdomain covered by parent"),
    ("https://www.arxiv.org/pdf/x.pdf",        True,  "www. normalized"),
    ("https://ftp.ncbi.nlm.nih.gov/pub/x.pdf", True,  "PMC ftp host via parent domain"),
    ("https://arxiv.org.evil.com/x.pdf",       False, "SECURITY lookalike suffix rejected"),
    ("https://notarxiv.org/x.pdf",             False, "SECURITY substring is not a subdomain"),
    ("https://sci-hub.se/x.pdf",               False, "unlisted host rejected"),
]:
    check(f"{why}: {url}", is_allowed(url, H), want)

# --------------------------------------------------------------------------- #
section("3. Batch input grammar — --from-file")

sample = HERE / "_validate_batch_tmp.txt"
sample.write_text("# comment\n\nAttention Is All You Need\n"
                  "doi:10.1371/journal.pmed.0020124\n10.1038/nature14539\n"
                  "Deep learning | 10.1038/nature14539\n", encoding="utf-8")
try:
    entries = parse_titles(sample)
finally:
    sample.unlink(missing_ok=True)

check("comment and blank lines skipped", len(entries), 4)
check("plain title", entries[0], ("Attention Is All You Need", None))
check("doi: prefix", entries[1], (None, "10.1371/journal.pmed.0020124"))
check("bare DOI autodetected", entries[2], (None, "10.1038/nature14539"))
check("Title | DOI", entries[3], ("Deep learning", "10.1038/nature14539"))

# --------------------------------------------------------------------------- #
section("4. Title matching")

check("normalize folds case and punctuation",
      normalize_title("Attention Is All You Need!"), "attention is all you need")
check("identical titles score 1.0",
      title_similarity("Attention Is All You Need", "attention is all you need!"), 1.0)
check("missing title scores 0.0", title_similarity("x", None), 0.0)
check("index truncation still matches (shorter found title contained in query)",
      title_similarity("Attention Is All You Need: A Transformer", "Attention Is All You Need"), 0.95)
check("short generic query does not match a longer distinct title",
      title_similarity("Deep learning", "Deep learning for image recognition") < TITLE_MATCH_THRESHOLD, True)

# [F2] a candidate that ADDS words is a different paper
check("[F2] prepended words rejected: 'Tensor Product Attention Is All You Need'",
      title_similarity("Attention Is All You Need",
                       "Tensor Product Attention Is All You Need") < TITLE_MATCH_THRESHOLD, True)
check("[F2] prepended words rejected: 'Not All Attention Is All You Need'",
      title_similarity("Attention Is All You Need",
                       "Not All Attention Is All You Need") < TITLE_MATCH_THRESHOLD, True)
check("[F2] appended words still rejected on their own merits",
      title_similarity("Attention Is All You Need",
                       "Attention Is All You Need For Speech Recognition") < TITLE_MATCH_THRESHOLD, True)

# --------------------------------------------------------------------------- #
section("5. Identity scoring")

DOI = "10.1038/nature14539"
exact = Candidate(source="s", title="Deep learning", doi=DOI, doi_keyed=True)
other = Candidate(source="s", title="Deep learning", doi="10.9999/other", doi_keyed=True)
silent = Candidate(source="s", title=None, doi=None, doi_keyed=True)
search_hit = Candidate(source="arxiv", title="Deep Learning", doi=None, doi_keyed=False)

check("matching DOI scores 1.0", identity_score(exact, None, DOI), 1.0)
check("DOI conflict with no title to fall back on scores 0.0",
      identity_score(other, None, DOI), 0.0)
check("DOI conflict on a DOI-keyed record defers to the title (preprint vs published)",
      identity_score(other, "Deep learning", DOI), 1.0)
check("DOI-keyed record that echoed no DOI scores 0.9",
      identity_score(silent, None, DOI), 0.9)
check("title-only query scores on the title alone",
      identity_score(search_hit, "Deep Learning", None), 1.0)

# [F1] the defect: a title-search hit standing in for a DOI-pinned query
check("[F1] title-search hit REJECTED against a DOI-pinned query",
      identity_score(search_hit, "Deep learning", DOI), 0.0)
check("[F1] and rejected even with a perfect title match",
      identity_score(search_hit, "Deep Learning", DOI) < TITLE_MATCH_THRESHOLD, True)
check("[F1] a DOI-less hit that was NOT fetched by DOI proves nothing",
      identity_score(Candidate(source="s", title=None, doi=None, doi_keyed=False), None, DOI), 0.0)

# --------------------------------------------------------------------------- #
section("6. Candidate ranking — identity outranks fetchability")

check("verified candidate keeps full confidence",
      Candidate(source="s", title_score=1.0, pdf_verified=True).confidence, 1.0)
check("unverified candidate is discounted to 0.7",
      Candidate(source="s", title_score=1.0, pdf_verified=False).confidence, 0.7)

right_paper = Candidate(source="openalex", title="Deep learning", doi=DOI, doi_keyed=True,
                        pdf_url="https://www.nature.com/x.pdf",
                        title_score=1.0, pdf_verified=False)
impostor = Candidate(source="arxiv", title="Deep Learning", doi=None,
                     pdf_url="https://arxiv.org/pdf/1807.07987v2",
                     title_score=0.95, pdf_verified=True)
check("[F1] exact match wins even unverified, over a verified weaker match",
      pick_best([impostor, right_paper]).source, "openalex")
check("[F1] ranking on confidence alone would have picked the wrong one",
      impostor.confidence > right_paper.confidence, True)

tie_unverified = Candidate(source="a", title_score=1.0, pdf_verified=False)
tie_verified = Candidate(source="b", title_score=1.0, pdf_verified=True)
check("within one identity tier, a fetchable PDF breaks the tie",
      pick_best([tie_unverified, tie_verified]).source, "b")
check("empty candidate list yields no best", pick_best([]), None)

# --------------------------------------------------------------------------- #
section("7. Copy identity — location DOI beats work-level DOI")

check("[F3] arXiv PDF URL yields the arXiv DOI, not the merged work DOI",
      location_doi("https://arxiv.org/pdf/1706.03762", "10.65215/2q58a426"),
      "10.48550/arXiv.1706.03762")
check("[F3] versioned arXiv URL resolves to the same submission",
      location_doi("https://arxiv.org/pdf/1706.03762v7", None), "10.48550/arXiv.1706.03762")
check("[F3] old-style arXiv id recognised",
      (location_doi("https://arxiv.org/abs/math.GT/0309136", None) or "")
      .startswith("10.48550/arXiv."), True)
check("non-arXiv host keeps the work DOI",
      location_doi("https://www.nature.com/articles/sdata201618.pdf", "10.1038/sdata.2016.18"),
      "10.1038/sdata.2016.18")
check("Candidate.identity prefers the location's own DOI",
      Candidate(source="s", doi="10.65215/2q58a426",
                location_doi="10.48550/arXiv.1706.03762").identity,
      "10.48550/arXiv.1706.03762")

# --------------------------------------------------------------------------- #
section("8. Output filename construction")

attention = Resolution(
    query_title="Attention Is All You Need", query_doi=None,
    best=Candidate(source="openalex", title="Attention Is All You Need",
                   doi="10.65215/2q58a426", location_doi="10.48550/arXiv.1706.03762",
                   pdf_url="https://arxiv.org/pdf/1706.03762"))
check("[F3] filename carries the copy's own DOI, not the mirror's",
      slugify(attention), "attention-is-all-you-need_10-48550-arxiv-1706-03762.pdf")

# [F1] the filename must never borrow the requested DOI
borrowed = Resolution(
    query_title="Deep learning", query_doi="10.1038/nature14539",
    best=Candidate(source="arxiv", title="Deep Learning", doi=None,
                   pdf_url="https://example.org/paper.pdf"))
name = slugify(borrowed)
check("[F1] requested DOI never appears in a filename it did not earn",
      "nature14539" in name, False)
check("[F1] an unproven identity falls back to a URL digest",
      name.startswith("deep-learning_") and name.endswith(".pdf"), True)

# [F4] the DOI slug keeps its head, not its tail
long_doi = Resolution(query_title="x", query_doi=None,
                      best=Candidate(source="s", title="AI driven web crawling",
                                     doi="10.1038/s41598-025-25616-x"))
check("[F4] registrant prefix survives truncation",
      slugify(long_doi).split("_")[1].startswith("10-1038"), True)

check("no best falls back to the query title",
      slugify(Resolution(query_title="No Match Here", query_doi=None, best=None)),
      "no-match-here.pdf")
check("title slug truncated to 80 characters",
      len(slugify(Resolution(query_title="x" * 200, query_doi=None, best=None))) - 4, 80)

# --------------------------------------------------------------------------- #
section("9. Declared agent-tool surface")

check("tool name", FIND_PAPER_PDF_TOOL["name"], "find_paper_pdf")
check("accepts title or doi", FIND_PAPER_PDF_TOOL["input_schema"]["anyOf"],
      [{"required": ["title"]}, {"required": ["doi"]}])
check("resolution reports the copy identity and the work identity separately",
      sorted(Resolution(query_title="t", query_doi=None,
                        best=Candidate(source="s")).as_dict()),
      ["alternatives", "confidence", "doi", "errors", "matched_title", "pdf_url",
       "source", "work_doi"])
check("a resolution with no match reports no DOI at all",
      Resolution(query_title="t", query_doi="10.1/x", best=None).as_dict()["doi"], None)

print(f"\nthresholds: TITLE_MATCH_THRESHOLD={TITLE_MATCH_THRESHOLD}  MIN_PDF_BYTES={MIN_PDF_BYTES}")
print(f"\n{'=' * 60}\n{PASSED} passed, {FAILED} failed\n{'=' * 60}")
raise SystemExit(1 if FAILED else 0)
