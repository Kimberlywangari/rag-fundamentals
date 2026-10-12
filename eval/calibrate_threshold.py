"""Calibrate the retrieval-score guard (config.MIN_SIMILARITY).

For answerable questions we look at the best chunk's cosine similarity; for questions the
documents cannot answer we do the same.  If the two groups are separable, a threshold between
them refuses obvious non-questions *without calling the LLM*.  The trade-off:

    threshold too high  -> refuses answerable questions phrased unusually (false refusals)
    threshold too low   -> lets off-topic questions through to the LLM, whose prompt rule is
                           then the only remaining defence

The calibration negatives below are deliberately NOT the unanswerable questions of the test set,
so the guard is not tuned on what it is later tested on.

Run:  python -m eval.calibrate_threshold
"""

from __future__ import annotations

import statistics
import sys

from eval.eval_set import answerable_questions, unanswerable_questions
from src import config
from src.retrieve import load_retriever

CALIBRATION_NEGATIVES = [
    "What is the VAT rate in Kenya?",
    "How do I register a motor vehicle in Kenya?",
    "What are the requirements for obtaining a Kenyan passport?",
    "What is the punishment for murder in Kenya?",
    "What does the Kenyan Constitution say about freedom of the press?",
    "How does the California Consumer Privacy Act define personal information?",
    "What is the penalty for a data breach under India's DPDP Act?",
    "How long is maternity leave in Kenya?",
    "What is the capital city of Australia?",
    "How do I bake sourdough bread?",
    "What licence does a bank need to operate in Kenya?",
    "What is the maximum interest rate banks may charge in Kenya?",
]


def top1(retriever, question: str) -> float:
    return retriever.retrieve(question, k=1)[0].score


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    retriever = load_retriever()
    pos = {q["id"]: top1(retriever, q["question"]) for q in answerable_questions()}
    neg_test = {q["id"]: top1(retriever, q["question"]) for q in unanswerable_questions()}
    neg_cal = {q: top1(retriever, q) for q in CALIBRATION_NEGATIVES}

    def describe(name: str, d: dict) -> None:
        v = sorted(d.values())
        print(f"{name:34} n={len(v):2d}  min={v[0]:.3f}  median={statistics.median(v):.3f}  max={v[-1]:.3f}")

    describe("answerable (top-1 score)", pos)
    describe("unanswerable, test set", neg_test)
    describe("unanswerable, calibration only", neg_cal)

    print("\nlowest answerable scores:")
    for qid, s in sorted(pos.items(), key=lambda kv: kv[1])[:5]:
        print(f"  {s:.3f}  {qid}")
    print("\nhighest calibration-negative scores:")
    for q, s in sorted(neg_cal.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {s:.3f}  {q}")

    print("\nthreshold sweep (calibration negatives only):")
    print("  threshold | answerable refused | calibration negatives blocked")
    for t in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55):
        refused = sum(s < t for s in pos.values())
        blocked = sum(s < t for s in neg_cal.values())
        print(f"  {t:9.2f} | {refused:2d}/{len(pos)}              | {blocked:2d}/{len(neg_cal)}")
    print(f"\nconfigured MIN_SIMILARITY = {config.MIN_SIMILARITY}")


if __name__ == "__main__":
    main()
