"""The labelled evaluation set and the tools that keep it honest.

Each answerable question carries
    source     the PDF it is answered from
    section    the section / regulation (human-readable)
    pages      PDF page(s) (1-based, as a viewer shows them) holding the passage
    anchor     a short verbatim phrase that contains the answer fact
    evidence   the longer verbatim passage the expected answer was read from

``python -m eval.eval_set`` re-checks every one of those claims against the ingested text, so an
expected answer can never drift away from the source or be "invented": a question that cannot be
located on the stated page fails verification.

``anchor_in_text`` is also the *hit* test of the chunking experiment: a retrieved chunk is a hit
if it comes from the right PDF and contains the anchor phrase.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

from src import config
from src.chunking import assemble_text, page_of
from src.ingest import Document, load_corpus

QUESTIONS_PATH = config.EVAL_DIR / "questions.json"

_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})
_EDGE_PUNCT = ".,;:!?()[]\"'—–"  # trimmed from word edges; inner hyphens/apostrophes stay


def _norm_token(token: str) -> str:
    """Lower-case, NFKC, straight quotes, no punctuation at the word edges."""
    return unicodedata.normalize("NFKC", token).translate(_QUOTES).lower().strip(_EDGE_PUNCT)


def normalise(text: str) -> str:
    """Whitespace / case / quote / edge-punctuation insensitive form used for comparisons."""
    return " ".join(t for t in (_norm_token(w) for w in text.split()) if t)


def load_questions(path: Path | str = QUESTIONS_PATH) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def answerable_questions(questions: list[dict] | None = None) -> list[dict]:
    return [q for q in (questions or load_questions()) if q["answerable"]]


def unanswerable_questions(questions: list[dict] | None = None) -> list[dict]:
    return [q for q in (questions or load_questions()) if not q["answerable"]]


def anchor_in_text(anchor: str, text: str) -> bool:
    """True if ``anchor`` occurs in ``text`` ignoring case, whitespace and quote style."""
    return normalise(anchor) in normalise(text)


def _find_spans(text: str, phrase: str) -> list[tuple[int, int]]:
    """All (start, end) character spans of ``phrase`` in ``text`` at word granularity."""
    words = [(m.start(), m.end(), _norm_token(m.group())) for m in re.finditer(r"\S+", text)]
    words = [w for w in words if w[2]]  # drop tokens that were pure punctuation
    target = [t for t in (_norm_token(t) for t in phrase.split()) if t]
    n = len(target)
    spans = []
    for i in range(len(words) - n + 1):
        if [w[2] for w in words[i : i + n]] == target:
            spans.append((words[i][0], words[i + n - 1][1]))
    return spans


def verify(questions: list[dict], docs: list[Document]) -> list[tuple[str, str, str]]:
    """Return ``(question id, status, detail)`` for every question; status is OK or FAIL."""
    by_source = {d.source: assemble_text(d) for d in docs}
    results = []
    for q in questions:
        qid = q["id"]
        if not q["answerable"]:
            results.append((qid, "OK", "unanswerable (no source passage expected)"))
            continue
        if q["source"] not in by_source:
            results.append((qid, "FAIL", f"unknown source {q['source']!r}"))
            continue
        text, page_starts = by_source[q["source"]]
        spans = _find_spans(text, q["anchor"])
        if len(spans) != 1:
            results.append((qid, "FAIL", f"anchor found {len(spans)} times (must be exactly 1)"))
            continue
        start, end = spans[0]
        found_pages = list(range(page_of(start, page_starts), page_of(end - 1, page_starts) + 1))
        if found_pages != q["pages"]:
            results.append((qid, "FAIL", f"anchor is on page(s) {found_pages}, question says {q['pages']}"))
            continue
        if not _find_spans(text, q["evidence"]):
            results.append((qid, "FAIL", "evidence passage not found verbatim"))
            continue
        if normalise(q["anchor"]) not in normalise(q["evidence"]):
            results.append((qid, "FAIL", "anchor is not part of the evidence passage"))
            continue
        results.append((qid, "OK", f"{q['section']} p.{'-'.join(map(str, q['pages']))}"))
    return results


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    questions = load_questions()
    results = verify(questions, load_corpus())
    for qid, status, detail in results:
        print(f"  {status:4} {qid:9} {detail}")
    failed = [r for r in results if r[1] != "OK"]
    n_ans = len(answerable_questions(questions))
    print(f"\n{len(results)} questions ({n_ans} answerable, {len(results) - n_ans} unanswerable); "
          f"{len(failed)} failed verification")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
