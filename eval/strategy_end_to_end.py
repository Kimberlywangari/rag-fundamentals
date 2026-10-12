"""End-to-end check of the chunking choice: does the retrieval-level winner also give the best
*answers*?

The chunking experiment scores whether the answer phrase is present in a retrieved chunk.
That cannot see a chunk boundary removing the words that give a fact its scope.  Here the same
22 questions are answered by the full pipeline (retrieve top-5 -> Gemini) once with fixed
1000/200 chunks and once with structure-aware chunks, and the answers are graded with the same
fact patterns the live tests use.  Gemini answers are cached, so a rerun is free.

Run:  python -m eval.strategy_end_to_end      (needs GEMINI_API_KEY and network)
"""

from __future__ import annotations

import json
import sys
import warnings

import chromadb

from eval.eval_set import answerable_questions
from src import config
from src.chunking import chunk_fixed, chunk_structure
from src.embed_store import Embedder, build_index
from src.generate import Generator
from src.ingest import load_corpus
from src.retrieve import Retriever
from tests.live_cases import is_correct


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    docs = load_corpus()
    embedder = Embedder()
    generator = Generator()
    questions = answerable_questions()

    strategies = {
        "fixed 1000/200": [c for d in docs for c in chunk_fixed(d, 1000, 200)],
        f"structure max {config.STRUCTURE_MAX_SIZE}": [
            c for d in docs for c in chunk_structure(d, config.STRUCTURE_MAX_SIZE)
        ],
    }
    results: dict[str, dict[str, dict]] = {}
    for name, chunks in strategies.items():
        col = build_index(chunks, embedder, client=chromadb.EphemeralClient(),
                          name="e2e", reset=True)
        retriever = Retriever(embedder, col)
        rows = {}
        for q in questions:
            top = retriever.retrieve(q["question"], k=config.TOP_K)
            ans = generator.answer(q["question"], top)
            rows[q["id"]] = {
                "correct": ans.sufficient and is_correct(q["id"], ans.text),
                "declined": not ans.sufficient,
                "model": ans.model,
                "answer": ans.text,
            }
        results[name] = rows
        n_ok = sum(r["correct"] for r in rows.values())
        n_dec = sum(r["declined"] for r in rows.values())
        print(f"{name:22} correct {n_ok}/{len(rows)}   declined {n_dec}", flush=True)

    names = list(results)
    print("\nper-question differences:")
    for q in questions:
        a, b = (results[n][q["id"]]["correct"] for n in names)
        if a != b:
            print(f"  {q['id']:9} {names[0]}={'ok' if a else 'NO'}  {names[1]}={'ok' if b else 'NO'}")
    out = config.EVAL_DIR / "strategy_end_to_end.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
