"""Live tests: real embedding model + real Gemini.   Run with:  pytest --live

Responses are cached in .llm_cache/ (git-ignored), so a second run is fast and repeatable and
spends no free-tier quota.  Delete that folder to force fresh model calls.

What these tests establish
  (a) a verifiable fact is answered correctly, and that fact is in the source PDF
  (b) retrieval matters: RAG answers correctly where base Gemini (no context) does not
  (c) unanswerable questions (tax law, employment law, other jurisdictions) get
      "insufficient evidence", not a fabricated answer
  (d) the retrieved chunks contain the expected passage
  (e) every number the answer states appears in the retrieved text
"""

from __future__ import annotations

import re

import pytest

from eval.eval_set import answerable_questions, anchor_in_text, unanswerable_questions
from src import config
from src.chunking import assemble_text
from src.generate import INSUFFICIENT_PREFIX
from src.ingest import load_corpus
from tests.live_cases import FACTS, is_correct, missing_facts

pytestmark = pytest.mark.live

ANSWERABLE = {q["id"]: q for q in answerable_questions()}
UNANSWERABLE = {q["id"]: q for q in unanswerable_questions()}

# Passages the chosen embedding model + chunking never places in the top 5 (see
# eval/chunking_results.md): a known, documented limitation of dense-only retrieval with no
# re-ranker.  strict=True: if one of these starts passing, the test fails loudly so the
# limitation note gets updated instead of silently going stale.
KNOWN_RETRIEVAL_MISSES = {
    "dpa-07": "paraphrase gap: 'who has to prove ... agreed' vs 'burden of proof ... consent'",
    "reg-05": "definition sits in a 'Part VII interpretation' regulation; ranks 7th of 284 chunks",
}


def maybe_xfail(qid: str):
    return pytest.param(
        qid,
        marks=pytest.mark.xfail(strict=True, reason=KNOWN_RETRIEVAL_MISSES[qid]),
        id=qid,
    ) if qid in KNOWN_RETRIEVAL_MISSES else pytest.param(qid, id=qid)


QIDS = [maybe_xfail(qid) for qid in ANSWERABLE]


@pytest.fixture(scope="module")
def source_text() -> dict[str, str]:
    """Cleaned full text of each PDF, to check facts against the documents themselves."""
    return {d.source: assemble_text(d)[0] for d in load_corpus()}


def chunk_rows(chunks, limit: int = 600) -> list[dict]:
    return [
        {"rank": c.rank, "score": c.score, "source": c.source, "pages": c.pages,
         "section": c.section, "text": " ".join(c.text.split())[:limit]}
        for c in chunks
    ]


def first_hit_rank(chunks, q) -> int | None:
    for c in chunks:
        if c.source == q["source"] and anchor_in_text(q["anchor"], c.text):
            return c.rank
    return None


# ---------------------------------------------------------------- (a) verifiable fact
@pytest.mark.parametrize("qid", QIDS)
def test_a_answer_is_correct_cited_and_verifiable_in_the_source(pipeline, recorder, source_text, qid):
    q = ANSWERABLE[qid]
    res = pipeline.ask(q["question"], k=config.TOP_K)
    a = res.answer
    ok = a.sufficient and is_correct(qid, a.text)
    recorder.append({
        "test": "a_verifiable_fact", "id": qid, "query": q["question"], "expected": q["expected_answer"],
        "source": q["source"], "section": q["section"], "pages": q["pages"],
        "rag_answer": a.text, "rag_model": a.model, "cited": a.cited, "guard": a.guard_triggered,
        "chunks": chunk_rows(res.chunks), "verdict": "PASS" if ok else "FAIL",
        "notes": "" if ok else f"missing fact patterns: {missing_facts(qid, a.text)}",
    })
    assert a.sufficient, f"model said insufficient evidence: {a.text}"
    assert is_correct(qid, a.text), f"missing {missing_facts(qid, a.text)} in: {a.text}"
    assert a.cited, "answer carries no citation"
    assert not a.invalid_citations, f"cites passages that were not supplied: {a.invalid_citations}"
    # the fact is really in the source document (not just in what the model wrote)
    assert anchor_in_text(q["anchor"], source_text[q["source"]])


# ------------------------------------------------------------- (b) retrieval matters
def test_b_rag_beats_base_gemini_on_at_least_one_question(pipeline, recorder):
    """For every answerable question run RAG and the no-context baseline, record both, and
    require at least one question where RAG is right and base Gemini is not.  The comparison
    is over ALL questions, so a favourable question cannot be cherry-picked."""
    rows = []
    for qid, q in ANSWERABLE.items():
        res = pipeline.ask(q["question"], k=config.TOP_K, with_base=True)
        rag_ok = res.answer.sufficient and is_correct(qid, res.answer.text)
        base_ok = is_correct(qid, res.base_answer.text)
        rows.append((qid, rag_ok, base_ok))
        recorder.append({
            "test": "b_rag_vs_base", "id": qid, "query": q["question"], "expected": q["expected_answer"],
            "source": q["source"], "section": q["section"], "pages": q["pages"],
            "rag_answer": res.answer.text, "rag_model": res.answer.model,
            "base_answer": res.base_answer.text, "base_model": res.base_answer.model,
            "cited": res.answer.cited, "chunks": chunk_rows(res.chunks),
            "rag_correct": rag_ok, "base_correct": base_ok,
            "verdict": ("RAG_BETTER" if rag_ok and not base_ok else
                        "BOTH_CORRECT" if rag_ok and base_ok else
                        "BASE_BETTER" if base_ok and not rag_ok else "BOTH_WRONG"),
        })
    rag_better = [qid for qid, r, b in rows if r and not b]
    base_better = [qid for qid, r, b in rows if b and not r]
    print(f"\nRAG better on {len(rag_better)}: {rag_better}")
    print(f"base better on {len(base_better)}: {base_better}")
    print(f"RAG correct {sum(r for _, r, _ in rows)}/{len(rows)}, "
          f"base correct {sum(b for _, _, b in rows)}/{len(rows)}")
    assert rag_better, "base Gemini matched RAG on every question; find more specific questions"
    assert sum(r for _, r, _ in rows) >= sum(b for _, _, b in rows)


# ----------------------------------------------------- (c) insufficient evidence
EXTRA_UNANSWERABLE = {
    "extra-tax": "What is the penalty for tax evasion in Kenya?",
    "extra-employment": "How much compensation can a court award for unfair dismissal in Kenya?",
}


@pytest.mark.parametrize(
    "qid, question",
    [(k, v["question"]) for k, v in UNANSWERABLE.items()] + list(EXTRA_UNANSWERABLE.items()),
    ids=list(UNANSWERABLE) + list(EXTRA_UNANSWERABLE),
)
def test_c_unanswerable_gets_insufficient_evidence_not_a_fabrication(pipeline, recorder, qid, question):
    res = pipeline.ask(question, k=config.TOP_K, with_base=True)
    a = res.answer
    recorder.append({
        "test": "c_insufficient_evidence", "id": qid, "query": question,
        "rag_answer": a.text, "rag_model": a.model, "guard": a.guard_triggered,
        "base_answer": res.base_answer.text, "base_model": res.base_answer.model,
        "top_score": res.chunks[0].score, "chunks": chunk_rows(res.chunks, 300),
        "verdict": "PASS" if (not a.sufficient and a.text.startswith(INSUFFICIENT_PREFIX)) else "FAIL",
        "notes": "refused by similarity guard" if a.guard_triggered else "refused by the model",
    })
    assert not a.sufficient
    assert a.text.startswith(INSUFFICIENT_PREFIX)
    if not a.guard_triggered:  # a model refusal must not smuggle in a number
        assert not re.search(r"\d", a.text), f"refusal contains a figure: {a.text}"


# ------------------------------------------------ (d) retrieved chunks hold the passage
@pytest.mark.parametrize("qid", QIDS)
def test_d_expected_passage_is_among_the_retrieved_chunks(pipeline, recorder, qid):
    q = ANSWERABLE[qid]
    chunks = pipeline.retriever.retrieve(q["question"], k=config.TOP_K)
    rank = first_hit_rank(chunks, q)
    recorder.append({
        "test": "d_passage_retrieved", "id": qid, "query": q["question"], "source": q["source"],
        "section": q["section"], "pages": q["pages"], "anchor": q["anchor"],
        "first_hit_rank": rank, "chunks": chunk_rows(chunks, 200),
        "verdict": "PASS" if rank else "MISS",
    })
    assert rank is not None, f"anchor passage not in the top {config.TOP_K}"


def test_d_overall_hit_rate_at_5_matches_the_experiment(pipeline):
    hits = sum(
        first_hit_rank(pipeline.retriever.retrieve(q["question"], k=5), q) is not None
        for q in ANSWERABLE.values()
    )
    assert hits / len(ANSWERABLE) >= 0.85  # experiment: 0.91 for fixed 1000/200


# ------------------------------------------- (e) answer facts appear in retrieved text
_NUMBER_WORDS = (
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fourteen|twenty|thirty|"
    r"forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million)"
)
_NUMBER_TOKEN = re.compile(rf"\b\d[\d,]*\b|\b{_NUMBER_WORDS}(?:[- ]{_NUMBER_WORDS})*\b", re.IGNORECASE)


def numbers_in(text: str) -> set[str]:
    """Numbers (digits or words) in ``text``, excluding the [n] citation markers."""
    text = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", " ", text)
    return {m.group(0).lower().replace("-", " ") for m in _NUMBER_TOKEN.finditer(text)}


@pytest.mark.parametrize("qid", QIDS)
def test_e_every_fact_in_the_answer_appears_in_the_retrieved_chunks(pipeline, recorder, qid):
    """Two groundedness checks that need no judgement call:
    1. every number the answer states (digits or words) occurs in the retrieved text;
    2. every fact the grader recognised in the answer (live_cases.FACTS) also occurs there.
    Non-numeric answers (lists, officials) are covered by check 2."""
    q = ANSWERABLE[qid]
    res = pipeline.ask(q["question"], k=config.TOP_K)
    context = " ".join(" ".join(c.text.split()) for c in res.chunks)
    context_nums = context.lower().replace("-", " ")
    answer = " ".join(res.answer.text.split())

    stated = numbers_in(answer)
    numbers_missing = sorted(n for n in stated if n not in context_nums)
    facts_in_answer = [p for p in FACTS[qid] if re.search(p, answer, flags=re.IGNORECASE)]
    facts_missing = [p for p in facts_in_answer if not re.search(p, context, flags=re.IGNORECASE)]
    ok = bool(facts_in_answer) and not numbers_missing and not facts_missing
    recorder.append({
        "test": "e_grounded_facts", "id": qid, "query": q["question"], "rag_answer": res.answer.text,
        "numbers_stated": sorted(stated), "numbers_not_in_chunks": numbers_missing,
        "facts_recognised": len(facts_in_answer), "facts_not_in_chunks": facts_missing,
        "verdict": "PASS" if ok else "FAIL",
    })
    assert facts_in_answer, f"no recognised fact in the answer: {res.answer.text}"
    assert not numbers_missing, f"numbers not found in the retrieved chunks: {numbers_missing}"
    assert not facts_missing, f"facts not supported by the retrieved chunks: {facts_missing}"
