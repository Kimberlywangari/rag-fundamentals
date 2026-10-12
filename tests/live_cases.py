"""Fact patterns that define a *correct* answer for each evaluation question.

Each entry is a list of regular expressions that must ALL match (case-insensitive) in an answer.
Numbers accept word and digit forms, because a model may write "72 hours" or "seventy-two hours".
The facts themselves are the ones verified against the PDFs in eval/questions.json
(``python -m eval.eval_set``): this file only says how to recognise them in free text.
"""

from __future__ import annotations

import re


def n(word: str, digits: str | None = None) -> str:
    """Regex for a number written as a word or as digits (e.g. n("seventy[- ]two", "72"))."""
    return rf"(?:\b{word}\b|\b{digits}\b)" if digits else rf"\b{word}\b"


def money(word: str, digits: int, unit: str = "million") -> str:
    """'five million' | '5 million' | '5,000,000' | 'KES 5,000,000'."""
    scale = {"million": 1_000_000, "thousand": 1_000}[unit]
    return rf"(?:\b{word}\s+{unit}\b|\b{digits}\s*{unit}\b|{digits * scale:,})"


FACTS: dict[str, list[str]] = {
    "dpa-01": [n(r"seventy[- ]two", "72"), r"\bhours?\b"],
    "dpa-02": [r"\breasons?\b", r"\bdelay"],
    "dpa-03": [money("five", 5), r"(?:one per ?cent|1\s*(?:%|per ?cent)|1 percent)", r"\blower\b"],
    "dpa-04": [money("three", 3), n("ten", "10") + r"\s+years"],
    "dpa-05": [n("six", "6") + r"\s+years", r"(?:not|no)\b.{0,30}\bre-?appoint|single term|not eligible"],
    "dpa-06": [n("sixty", "60") + r"\s+days"],
    "dpa-07": [r"data controller|data processor|controller|processor"],
    "dpa-08": [r"\binform", r"\baccess", r"\bobject", r"\bcorrect", r"\bdelet|\berasure"],
    "dpa-09": [n("ninety", "90") + r"\s+days"],
    "reg-01": [n("seven", "7") + r"\s+days"],
    "reg-02": [n("fourteen", "14") + r"\s+days"],
    "reg-03": [rf"(?:\btwenty thousand\b|\b20,?000\b|\b20 thousand\b)", n("six", "6") + r"\s+months"],
    # the regulation says processing may commence AND the report is taken to be approved
    "reg-04": [n("sixty", "60") + r"\s+days", r"\bcommence|\bstart|\bbegin|\bproceed|\bapproved\b"],
    "reg-05": [r"transfers?\b.{0,40}personal data", r"outside kenya"],
    "reg-06": [r"non-?negotiable|terms and conditions"],
    "reg-07": [r"cabinet secretary"],
    "cmca-01": [n(r"twenty[- ]four", "24") + r"\s+hours"],
    "cmca-02": [money("twenty", 20), n("ten", "10") + r"\s+years"],
    "cmca-03": [money("fifty", 50)],
    "cmca-04": [n("four", "4") + r"\s+years"],
    "cmca-05": [money(r"twenty[- ]five", 25), n("twenty", "20") + r"\s+years"],
    # the question asks only "what fine", so only the fine amount is required
    "cmca-06": [r"(?:\bthree hundred thousand\b|\b300,?000\b|\b300 thousand\b)"],
}


def is_correct(qid: str, answer: str) -> bool:
    """True if every required fact pattern for ``qid`` occurs in ``answer``."""
    text = " ".join(answer.split())
    return all(re.search(p, text, flags=re.IGNORECASE) for p in FACTS[qid])


def missing_facts(qid: str, answer: str) -> list[str]:
    text = " ".join(answer.split())
    return [p for p in FACTS[qid] if not re.search(p, text, flags=re.IGNORECASE)]
