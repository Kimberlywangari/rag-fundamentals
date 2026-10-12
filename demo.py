"""End-to-end demo: question -> ranked chunks (score, rank, source, page) -> grounded answer.

    python demo.py "Within how many hours must a data breach be notified?"
    python demo.py "..." --base          # also show what Gemini says with no retrieval
    python demo.py "..." -k 3 --rebuild  # change top-k / force re-indexing
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import warnings

warnings.filterwarnings("ignore")

from src import config  # noqa: E402
from src.pipeline import RAGPipeline, headings_in  # noqa: E402


def _wrap(text: str, indent: str = "    ", width: int = 100) -> str:
    return textwrap.fill(" ".join(text.split()), width=width, initial_indent=indent,
                         subsequent_indent=indent)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Ask a question about Kenyan data-protection law.")
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=config.TOP_K, help="chunks to retrieve")
    parser.add_argument("--base", action="store_true", help="also answer with no retrieval")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the vector index first")
    parser.add_argument("--no-cache", action="store_true", help="bypass the local response cache")
    parser.add_argument("--full", action="store_true", help="print full chunk text")
    args = parser.parse_args()

    pipeline = RAGPipeline()
    pipeline.ensure_index(rebuild=args.rebuild)
    result = pipeline.ask(args.question, k=args.k, with_base=args.base, use_cache=not args.no_cache)

    print("=" * 100)
    print(f"QUERY: {result.question}")
    print("=" * 100)
    print(f"\nRETRIEVED CHUNKS (top {len(result.chunks)} by cosine similarity)")
    for c in result.chunks:
        title = config.DOCUMENT_TITLES.get(c.source, c.source)
        print(f"\n  #{c.rank}  score={c.score:.4f}  {title}  |  page {c.pages}")
        heads = [c.section] if c.section else headings_in(c.text)
        if heads:
            print(f"      sections: {'; '.join(heads)}")
        print(_wrap(c.text if args.full else c.text[:420] + (" ..." if len(c.text) > 420 else "")))

    a = result.answer
    print("\n" + "-" * 100)
    if a.guard_triggered:
        origin = "similarity guard (LLM not called)"
    else:
        origin = f"{a.model}{', cached' if a.from_cache else ''}"
    print(f"RAG ANSWER  [{origin}]")
    print(_wrap(a.text, indent="  "))
    if a.cited:
        print("\n  Sources cited: " + ", ".join(
            f"[{n}] {config.DOCUMENT_TITLES.get(result.chunks[n-1].source, result.chunks[n-1].source)}"
            f" p.{result.chunks[n-1].pages}" for n in a.cited))
    if a.invalid_citations:
        print(f"  WARNING: answer cites passages that were not supplied: {a.invalid_citations}")
    if not a.sufficient:
        print("  (insufficient evidence: no answer was fabricated)")

    if result.base_answer is not None:
        b = result.base_answer
        print("\n" + "-" * 100)
        print(f"BASE LLM, NO RETRIEVAL  [{b.model}{', cached' if b.from_cache else ''}]")
        print(_wrap(b.text, indent="  "))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
