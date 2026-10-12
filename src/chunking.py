"""Chunking: turn a cleaned document into retrievable pieces, each with provenance metadata.

Why chunk at all?  An embedding model turns a *whole input* into one vector.  If the input is a
whole Act, the vector is a blur of everything in it; if it is one sentence, it loses context.
Chunk size is the knob between those extremes (and the embedding model also has a hard input
limit: all-MiniLM-L6-v2 silently truncates past 256 word-pieces).

This module holds the fixed-size strategy.  Sizes are in *characters* (deterministic, easy to
explain); the experiment in eval/ also reports the token cost of each size.
"""

from __future__ import annotations

import argparse
import re
import statistics
from bisect import bisect_right
from dataclasses import dataclass

from src import config
from src.ingest import Document, load_corpus

_WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # unique and stable: "<source-slug>-<strategy>-<index>"
    text: str
    source: str  # PDF filename
    page_start: int  # 1-based PDF page of the first character
    page_end: int  # 1-based PDF page of the last character (differs when a chunk spans a page)
    chunk_index: int  # position within its source document
    char_start: int  # offsets into the document's assembled text (for audit / overlap checks)
    char_end: int
    strategy: str  # "fixed" | "structure"
    section: str = ""  # e.g. "Section 25" / "Regulation 40" (structure strategy only)

    def to_metadata(self) -> dict:
        """Flat str/int dict, the only shape ChromaDB accepts for metadata."""
        return {
            "source": self.source,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "strategy": self.strategy,
            "section": self.section,
        }


def slugify(source: str) -> str:
    stem = re.sub(r"\.pdf$", "", source, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")


def assemble_text(doc: Document) -> tuple[str, list[tuple[int, int]]]:
    """Join the document's non-empty pages into one string.

    Returns ``(text, page_starts)`` where ``page_starts`` is a list of
    ``(char_offset, page_number)`` sorted by offset.  Chunking over the whole document (instead
    of page by page) means a clause that runs over a page break stays in one piece.
    """
    parts: list[str] = []
    starts: list[tuple[int, int]] = []
    pos = 0
    for page in doc.pages:
        text = page.text
        if not text:
            continue
        if parts:
            pos += 1  # the "\n" joining pages
        starts.append((pos, page.page))
        parts.append(text)
        pos += len(text)
    return "\n".join(parts), starts


def page_of(offset: int, page_starts: list[tuple[int, int]]) -> int:
    """PDF page containing character ``offset``."""
    idx = bisect_right([s for s, _ in page_starts], offset) - 1
    return page_starts[max(idx, 0)][1]


def chunk_fixed(doc: Document, size: int, overlap: int) -> list[Chunk]:
    """Split ``doc`` into windows of at most ``size`` characters, sharing ~``overlap`` characters.

    * A window ends at a word boundary (never mid-word), so ``size`` is an upper bound.
    * The next window starts ``overlap`` characters before the previous one ended, snapped
      forward to the next word start.  ``overlap=0`` means consecutive chunks share nothing:
      a sentence that straddles a boundary is cut in two and neither half carries the whole idea.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")

    text, page_starts = assemble_text(doc)
    words = [(m.start(), m.end()) for m in _WORD.finditer(text)]
    word_starts = [s for s, _ in words]
    slug = slugify(doc.source)
    chunks: list[Chunk] = []

    i = 0
    n = len(words)
    while i < n:
        start = words[i][0]
        j = i  # last word included
        while j + 1 < n and words[j + 1][1] - start <= size:
            j += 1
        end = words[j][1]
        chunks.append(
            Chunk(
                chunk_id=f"{slug}-fixed-{len(chunks):04d}",
                text=text[start:end],
                source=doc.source,
                page_start=page_of(start, page_starts),
                page_end=page_of(end - 1, page_starts),
                chunk_index=len(chunks),
                char_start=start,
                char_end=end,
                strategy="fixed",
            )
        )
        if j == n - 1:
            break
        # first word that starts at/after (end - overlap); always advance at least one word
        nxt = bisect_right(word_starts, end - overlap - 1)
        i = max(nxt, i + 1)
    return chunks


def chunk_documents_fixed(docs: list[Document], size: int, overlap: int) -> list[Chunk]:
    return [c for d in docs for c in chunk_fixed(d, size, overlap)]


# --- CLI: show stats, sample chunks and the effect of overlap ----------------------
def _describe(chunks: list[Chunk]) -> str:
    lens = [len(c.text) for c in chunks]
    return (
        f"{len(chunks):4d} chunks | chars min/median/max = "
        f"{min(lens)}/{int(statistics.median(lens))}/{max(lens)}"
    )


def _show_boundaries(chunks: list[Chunk], count: int, tail: int = 90) -> None:
    for a, b in list(zip(chunks, chunks[1:]))[10 : 10 + count]:
        shared = max(0, a.char_end - b.char_start)
        print(f"  [{a.chunk_index}->{b.chunk_index}] shared={shared:4d} chars")
        print(f"      ...{a.text[-tail:]!r}")
        print(f"      {b.text[:tail]!r}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-size chunking demo.")
    parser.add_argument("--size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--source", default="Data Protection Act.pdf")
    args = parser.parse_args()

    docs = load_corpus()
    print(f"\n=== fixed-size chunking: size={args.size} overlap={args.overlap} ===")
    for d in docs:
        print(f"  {d.source[:48]:48} {_describe(chunk_fixed(d, args.size, args.overlap))}")

    doc = next(d for d in docs if d.source == args.source)
    chunks = chunk_fixed(doc, args.size, args.overlap)
    print(f"\n--- {args.samples} sample chunks from {args.source} ---")
    for c in chunks[8 : 8 + args.samples]:
        print(f"\n[{c.chunk_id}] pages {c.page_start}-{c.page_end}, chars {c.char_start}-{c.char_end}")
        print(c.text)

    print(f"\n--- chunk boundaries with overlap={args.overlap} ---")
    _show_boundaries(chunks, 2)
    no_overlap = chunk_fixed(doc, args.size, 0)
    print("\n--- the same boundaries with overlap=0 (nothing shared) ---")
    _show_boundaries(no_overlap, 2)


if __name__ == "__main__":
    main()
