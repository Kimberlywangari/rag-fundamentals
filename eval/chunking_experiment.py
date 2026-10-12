"""Chunking experiment: how do chunk size, overlap and strategy affect retrieval?

For every configuration the whole corpus is re-chunked, embedded and indexed (in memory), then
each answerable question is retrieved with and the *hit* test of eval_set is applied: a retrieved
chunk counts if it comes from the labelled PDF and contains the verbatim anchor phrase.

Metrics (higher is better)
    hit@k   share of questions with a hit among the top k results
    MRR     mean of 1/rank of the first hit (0 if no hit in the top 10)
Diagnostics
    chunks, median chars, and % of chunks over the embedding model's 256-token input limit
    (text beyond the limit is cut off before embedding, so it cannot influence retrieval).

Outputs: eval/chunking_results.md (table), .csv and .json (per-question first-hit ranks).
Run:  python -m eval.chunking_experiment
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass

import chromadb

from eval.eval_set import anchor_in_text, answerable_questions
from src import config
from src.chunking import Chunk, chunk_fixed, chunk_structure
from src.embed_store import Embedder, build_index
from src.ingest import load_corpus
from src.retrieve import Retriever

SIZES = [300, 600, 800, 1000, 1200, 1500]
OVERLAP_FRACTIONS = [0.0, 0.10, 0.20, 0.30]
STRUCTURE_MAX_SIZES = [800, 1200, 1500, 2000]
K_MAX = 10


@dataclass(frozen=True)
class Setup:
    strategy: str  # "fixed" | "structure"
    size: int  # window size (fixed) or max section size (structure)
    overlap: int = 0

    @property
    def label(self) -> str:
        if self.strategy == "fixed":
            return f"fixed {self.size}/{self.overlap}"
        return f"structure max {self.size}"


def all_setups() -> list[Setup]:
    setups = [
        Setup("fixed", size, round(size * frac))
        for size in SIZES
        for frac in OVERLAP_FRACTIONS
    ]
    setups += [Setup("structure", m) for m in STRUCTURE_MAX_SIZES]
    return setups


def chunk_with(docs, setup: Setup) -> list[Chunk]:
    if setup.strategy == "fixed":
        return [c for d in docs for c in chunk_fixed(d, setup.size, setup.overlap)]
    return [c for d in docs for c in chunk_structure(d, setup.size)]


def evaluate(setup: Setup, docs, questions, embedder) -> dict:
    chunks = chunk_with(docs, setup)
    client = chromadb.EphemeralClient()
    collection = build_index(chunks, embedder, client=client, name="exp", reset=True)
    retriever = Retriever(embedder, collection)

    first_hit: dict[str, int | None] = {}
    for q in questions:
        rank = None
        for r in retriever.retrieve(q["question"], k=K_MAX):
            if r.source == q["source"] and anchor_in_text(q["anchor"], r.text):
                rank = r.rank
                break
        first_hit[q["id"]] = rank

    n = len(questions)

    def hit_at(k: int) -> float:
        return sum(1 for r in first_hit.values() if r is not None and r <= k) / n

    # Coverage: is the answer phrase intact inside at least one chunk? If a boundary cuts it in
    # two, no retriever can ever return it whole, whatever the embedding model does.
    covered = sum(
        any(c.source == q["source"] and anchor_in_text(q["anchor"], c.text) for c in chunks)
        for q in questions
    )

    tokens = embedder.token_lengths([c.text for c in chunks])
    lens = [len(c.text) for c in chunks]
    return {
        "config": setup.label,
        "strategy": setup.strategy,
        "size": setup.size,
        "overlap": setup.overlap,
        "chunks": len(chunks),
        "median_chars": int(statistics.median(lens)),
        "median_tokens": int(statistics.median(tokens)),
        "pct_truncated": round(100 * sum(t > embedder.max_tokens for t in tokens) / len(tokens), 1),
        "coverage": round(covered / n, 3),
        "hit@1": round(hit_at(1), 3),
        "hit@3": round(hit_at(3), 3),
        "hit@5": round(hit_at(5), 3),
        "hit@10": round(hit_at(10), 3),
        "mrr": round(sum(1 / r for r in first_hit.values() if r) / n, 3),
        "first_hit": first_hit,
    }


def reciprocal_ranks(first_hit: dict[str, int | None]) -> list[float]:
    return [1 / r if r else 0.0 for r in first_hit.values()]


def bootstrap_mrr_diff(a: dict, b: dict, n: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """Paired bootstrap over questions: mean(MRR_a - MRR_b) with a 95% interval."""
    import random

    ra, rb = reciprocal_ranks(a["first_hit"]), reciprocal_ranks(b["first_hit"])
    diffs = [x - y for x, y in zip(ra, rb)]
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n)
    )
    return sum(diffs) / len(diffs), means[int(0.025 * n)], means[int(0.975 * n)]


def write_outputs(rows: list[dict], n_questions: int) -> None:
    config.EVAL_DIR.mkdir(exist_ok=True)
    cols = ["config", "chunks", "median_chars", "median_tokens", "pct_truncated", "coverage",
            "hit@1", "hit@3", "hit@5", "hit@10", "mrr"]
    with open(config.EVAL_DIR / "chunking_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])
    (config.EVAL_DIR / "chunking_results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    ranked = sorted(rows, key=lambda r: (-r["mrr"], -r["hit@1"], -r["hit@3"]))
    lines = [
        "# Chunking experiment results",
        "",
        f"{n_questions} answerable questions; embedding model `{config.EMBEDDING_MODEL}` "
        "(256-token input limit); cosine similarity; exact search.",
        "A hit = a top-k chunk from the labelled PDF that contains the verbatim anchor phrase.",
        "With so few questions one question is worth "
        f"{100 / n_questions:.1f} percentage points, so differences of one question are noise.",
        "",
        "Coverage = share of questions whose anchor phrase is intact inside at least one chunk "
        "(an upper bound on any hit@k).",
        "",
        "| config | chunks | median chars | median tokens | % > 256 tok | coverage | hit@1 | hit@3 | hit@5 | hit@10 | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['config']} | {r['chunks']} | {r['median_chars']} | {r['median_tokens']} | "
            f"{r['pct_truncated']} | {r['coverage']:.2f} | {r['hit@1']:.2f} | {r['hit@3']:.2f} | "
            f"{r['hit@5']:.2f} | {r['hit@10']:.2f} | {r['mrr']:.3f} |"
        )
    lines += ["", "## Ranked by MRR (then hit@1, hit@3)", ""]
    for i, r in enumerate(ranked[:5], start=1):
        lines.append(f"{i}. **{r['config']}**: MRR {r['mrr']:.3f}, hit@1 {r['hit@1']:.2f}, "
                     f"hit@3 {r['hit@3']:.2f}, hit@5 {r['hit@5']:.2f}")
    # Is the leader really better, or is it luck on a handful of questions?
    best = ranked[0]
    rivals = [
        next(r for r in rows if r["strategy"] == "fixed" and r["size"] == best["size"] and r["overlap"] == 0),
        max((r for r in rows if r["strategy"] == "structure"), key=lambda r: r["mrr"]),
        max((r for r in rows if r["strategy"] == "fixed" and r["size"] != best["size"]),
            key=lambda r: r["mrr"]),
    ] if best["strategy"] == "fixed" else [
        max((r for r in rows if r["strategy"] == "fixed"), key=lambda r: r["mrr"])
    ]
    lines += ["", f"## Is **{best['config']}** really better? (paired bootstrap over questions)", "",
              "| compared with | mean MRR difference | 95% interval | verdict |", "|---|---:|---:|---|"]
    for rival in rivals:
        mean, lo, hi = bootstrap_mrr_diff(best, rival)
        verdict = "clearly better" if lo > 0 else ("not distinguishable" if hi >= 0 else "worse")
        lines.append(f"| {rival['config']} | {mean:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {verdict} |")

    misses = {}
    for r in rows:
        for qid, rank in r["first_hit"].items():
            if rank is None or rank > 5:
                misses.setdefault(qid, []).append(r["config"])
    lines += ["", "## Questions missed from the top 5 (config count out of "
              f"{len(rows)})", ""]
    for qid, cfgs in sorted(misses.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"- `{qid}`: {len(cfgs)} configs")
    (config.EVAL_DIR / "chunking_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    docs = load_corpus()
    questions = answerable_questions()
    embedder = Embedder()
    rows = []
    for setup in all_setups():
        t0 = time.time()
        row = evaluate(setup, docs, questions, embedder)
        rows.append(row)
        print(f"{row['config']:22} chunks={row['chunks']:4d} trunc={row['pct_truncated']:5.1f}% "
              f"cov={row['coverage']:.2f} "
              f"hit@1={row['hit@1']:.2f} hit@3={row['hit@3']:.2f} hit@5={row['hit@5']:.2f} "
              f"MRR={row['mrr']:.3f}  ({time.time() - t0:.0f}s)")
    write_outputs(rows, len(questions))
    print(f"\nwrote {config.EVAL_DIR / 'chunking_results.md'}")


if __name__ == "__main__":
    main()
