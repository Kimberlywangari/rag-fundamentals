"""Chunking: turn a cleaned document into retrievable pieces, each with provenance metadata.

Why chunk at all?  An embedding model turns a *whole input* into one vector.  If the input is a
whole Act, the vector is a blur of everything in it; if it is one sentence, it loses context.
Chunk size is the knob between those extremes (and the embedding model also has a hard input
limit: all-MiniLM-L6-v2 silently truncates past 256 word-pieces).

Two strategies live here:

* ``chunk_fixed``      - windows of N characters with an overlap, blind to meaning.
* ``chunk_structure``  - one chunk per legal section / regulation, split further only when a
                         section is too long.  Each chunk knows which section it belongs to.

Sizes are in *characters* (deterministic, easy to explain); the experiment in eval/ also
reports the token cost of each size.
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
    section: str = ""  # e.g. "Section 25: Principles of data protection" (structure only)

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


# =============================================================================
# Strategy 1: fixed-size windows with overlap
# =============================================================================
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


# =============================================================================
# Strategy 2: structure-aware (legal sections / regulations)
# =============================================================================
# "25. Principles of data protection"  /  "40. A data controller or ..."  /  "46A. Further orders"
SECTION_START = re.compile(r"^(\d{1,3})([A-Z]?)\.\s+(\S.*)$")
# "PART II - ENABLING THE RIGHTS" (Regs)  /  "Part II – ESTABLISHMENT OF THE OFFICE" (Acts)
PART_START = re.compile(r"^(PART\s+[IVXLC]+\b|Part\s+[IVXLC]+\s*[–—-])")
# "SCHEDULE", "SECOND SCHEDULE [s. 75.]"
SCHEDULE_START = re.compile(r"^(?:[A-Z]+\s+)?SCHEDULE\b")
# new "unit" inside an over-long section: subsection "(1)", or a defined term '"data" means'
UNIT_START = re.compile(r"^(\(\d{1,3}[A-Z]?\)\s|[\"“”][^\"“”]{1,80}[\"“”]\s)")
MAX_TITLE_CHARS = 100  # a row longer than this after "25." is body text, not a heading
SKIPPED_NUMBERS_ALLOWED = 3  # repealed / omitted sections can leave small gaps


@dataclass
class _Line:
    text: str
    kind: str
    start: int  # offset into assemble_text(doc)
    end: int


def _flatten(doc: Document) -> list[_Line]:
    """Document lines with offsets that match ``assemble_text`` exactly."""
    out: list[_Line] = []
    pos = 0
    started = False
    for page in doc.pages:
        if not page.lines:
            continue
        if started:
            pos += 1
        started = True
        for j, line in enumerate(page.lines):
            if j:
                pos += 1
            out.append(_Line(line.text, line.kind, pos, pos + len(line.text)))
            pos += len(line.text)
    return out


@dataclass
class _Segment:
    label: str
    first: int  # index of first line
    last: int  # index of last line (inclusive)
    header: int = 0  # how many leading lines are the heading (margin note + number row)


def _find_segments(lines: list[_Line], unit_word: str) -> list[_Segment]:
    """Cut the line list into sections.  Preamble and schedules become unnumbered segments."""
    segments: list[_Segment] = []
    current: _Segment | None = None
    last_num = 0
    in_schedule = False
    in_part_heading = False  # skipping the wrapped second line of a long Part title

    def close(upto: int) -> None:
        nonlocal current
        if current is not None:
            current.last = upto
            if current.last >= current.first:
                segments.append(current)
            current = None

    for idx, ln in enumerate(lines):
        if ln.kind == "note":
            continue  # a margin heading is picked up when its regulation row is reached
        if in_part_heading:
            if ln.text.isupper() and not SECTION_START.match(ln.text):
                continue  # e.g. "OF PERSONAL DATA", "BREACHES": tail of a wrapped Part title
            in_part_heading = False
        if not in_schedule and SCHEDULE_START.match(ln.text) and len(ln.text) < 60:
            close(idx - 1)
            current = _Segment(label=ln.text, first=idx, last=idx, header=1)
            in_schedule = True
            continue
        if not in_schedule and PART_START.match(ln.text):
            close(idx - 1)  # the Part heading itself belongs to no chunk
            current = None
            in_part_heading = True
            continue
        m = None if in_schedule else SECTION_START.match(ln.text)
        if m:
            n, suffix, rest = int(m.group(1)), m.group(2), m.group(3)
            ok = (n == last_num and bool(suffix)) or (last_num < n <= last_num + SKIPPED_NUMBERS_ALLOWED)
            if ok:
                has_note = idx > 0 and lines[idx - 1].kind == "note"
                if has_note:
                    title = lines[idx - 1].text
                elif unit_word == "Section" and len(rest) <= MAX_TITLE_CHARS:
                    title = rest  # Acts print the heading on the number's own row
                else:
                    title = ""  # a regulation row is body text, not a heading
                title = title.rstrip(".")
                close((idx - 1) - (1 if has_note else 0))
                label = f"{unit_word} {n}{suffix}" + (f": {title}" if title else "")
                current = _Segment(
                    label=label, first=idx - 1 if has_note else idx, last=idx,
                    header=2 if has_note else 1,
                )
                last_num = n
                continue
        if current is None:
            current = _Segment(label="", first=idx, last=idx)  # preamble / text after a Part head
    close(len(lines) - 1)
    return segments


def _pack(lines: list[_Line], max_size: int, header: int = 0) -> list[list[_Line]]:
    """Greedy-pack consecutive lines into groups of <= ~max_size characters.

    Units (subsections, definitions) are kept whole when they fit; an over-long unit falls back
    to line-level packing.  The first ``header`` lines (the section heading) are always glued
    to the first group, so a heading is never stranded alone.
    """
    head, body = lines[:header], lines[header:]
    units: list[list[_Line]] = []
    for ln in body:
        if not units or UNIT_START.match(ln.text):
            units.append([ln])
        else:
            units[-1].append(ln)

    def size_of(group: list[_Line]) -> int:
        return group[-1].end - group[0].start

    groups: list[list[_Line]] = []
    cur: list[_Line] = list(head)
    for unit in units:
        pieces = [unit] if size_of(unit) <= max_size else [[ln] for ln in unit]
        for piece in pieces:
            # `cur == head` means nothing but the heading is in the group: never flush that alone
            if cur and cur != head and size_of(cur + piece) > max_size:
                groups.append(cur)
                cur = []
            cur = cur + piece
    if cur:
        groups.append(cur)
    return groups


def chunk_structure(doc: Document, max_size: int = config.STRUCTURE_MAX_SIZE) -> list[Chunk]:
    """One chunk per section/regulation; sections longer than ``max_size`` are split on units."""
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    text, page_starts = assemble_text(doc)
    lines = _flatten(doc)
    unit_word = "Regulation" if "regulation" in doc.source.lower() else "Section"
    slug = slugify(doc.source)
    chunks: list[Chunk] = []

    def emit(group: list[_Line], label: str, continued: bool) -> None:
        body = text[group[0].start : group[-1].end]
        if continued and label:
            body = f"{label} (continued)\n{body}"
        chunks.append(
            Chunk(
                chunk_id=f"{slug}-structure-{len(chunks):04d}",
                text=body,
                source=doc.source,
                page_start=page_of(group[0].start, page_starts),
                page_end=page_of(group[-1].end - 1, page_starts),
                chunk_index=len(chunks),
                char_start=group[0].start,
                char_end=group[-1].end,
                strategy="structure",
                section=label,
            )
        )

    for seg in _find_segments(lines, unit_word):
        seg_lines = lines[seg.first : seg.last + 1]
        for k, group in enumerate(_pack(seg_lines, max_size, seg.header)):
            emit(group, seg.label, continued=k > 0)

    # A single line can still exceed max_size (e.g. a very long run-on definition): split by words.
    final: list[Chunk] = []
    for c in chunks:
        if len(c.text) <= max_size * 1.25:
            final.append(c)
            continue
        pseudo = [(m.start(), m.end()) for m in _WORD.finditer(c.text)]
        i = 0
        while i < len(pseudo):
            j = i
            while j + 1 < len(pseudo) and pseudo[j + 1][1] - pseudo[i][0] <= max_size:
                j += 1
            piece = c.text[pseudo[i][0] : pseudo[j][1]]
            if i > 0 and c.section:
                piece = f"{c.section} (continued)\n{piece}"
            final.append(
                Chunk(
                    chunk_id="", text=piece, source=c.source, page_start=c.page_start,
                    page_end=c.page_end, chunk_index=0, char_start=c.char_start,
                    char_end=c.char_end, strategy="structure", section=c.section,
                )
            )
            i = j + 1
    # re-number ids / indexes after any splitting
    return [
        Chunk(
            chunk_id=f"{slug}-structure-{k:04d}", text=c.text, source=c.source,
            page_start=c.page_start, page_end=c.page_end, chunk_index=k, char_start=c.char_start,
            char_end=c.char_end, strategy="structure", section=c.section,
        )
        for k, c in enumerate(final)
    ]


# =============================================================================
# Dispatcher + CLI
# =============================================================================
def chunk_corpus(
    docs: list[Document],
    strategy: str = config.CHUNK_STRATEGY,
    size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
    max_size: int = config.STRUCTURE_MAX_SIZE,
) -> list[Chunk]:
    """Chunk every document with the chosen strategy."""
    if strategy == "fixed":
        return [c for d in docs for c in chunk_fixed(d, size, overlap)]
    if strategy == "structure":
        return [c for d in docs for c in chunk_structure(d, max_size)]
    raise ValueError(f"unknown chunking strategy: {strategy!r}")


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


def _show_chunk(c: Chunk, limit: int = 700) -> None:
    where = f"p.{c.page_start}" if c.page_start == c.page_end else f"p.{c.page_start}-{c.page_end}"
    print(f"\n[{c.chunk_id}] {where} chars {c.char_start}-{c.char_end} ({len(c.text)} long)"
          + (f"  section={c.section!r}" if c.section else ""))
    print(c.text if len(c.text) <= limit else c.text[:limit] + " ...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunking demo.")
    parser.add_argument("--size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument("--max-size", type=int, default=config.STRUCTURE_MAX_SIZE)
    parser.add_argument("--source", default="Data Protection Act.pdf")
    parser.add_argument("--section", type=int, default=None,
                        help="show this section/regulation number, structure vs fixed side by side")
    args = parser.parse_args()

    docs = load_corpus()
    print(f"\n=== fixed: size={args.size} overlap={args.overlap} | "
          f"structure: max_size={args.max_size} ===")
    for d in docs:
        print(f"  {d.source[:44]:44}")
        print(f"      fixed     {_describe(chunk_fixed(d, args.size, args.overlap))}")
        print(f"      structure {_describe(chunk_structure(d, args.max_size))}")

    doc = next(d for d in docs if d.source == args.source)
    fixed = chunk_fixed(doc, args.size, args.overlap)
    struct = chunk_structure(doc, args.max_size)

    if args.section is not None:
        word = "Regulation" if "regulation" in doc.source.lower() else "Section"
        mine = [c for c in struct if re.match(rf"{word} {args.section}(?![0-9A-Z])", c.section)]
        print(f"\n##### STRUCTURE chunks for {word} {args.section} in {doc.source} #####")
        for c in mine:
            _show_chunk(c)
        if mine:
            lo, hi = mine[0].char_start, mine[-1].char_end
            print(f"\n##### FIXED chunks overlapping the same characters ({lo}-{hi}) #####")
            for c in fixed:
                if c.char_start < hi and c.char_end > lo:
                    _show_chunk(c)
        return

    print(f"\n--- fixed chunk boundaries (overlap={args.overlap}) ---")
    _show_boundaries(fixed, 2)
    print("\n--- the same boundaries with overlap=0 (nothing shared) ---")
    _show_boundaries(chunk_fixed(doc, args.size, 0), 2)
    print("\n--- first structure-aware chunk labels ---")
    for c in struct[:12]:
        print(f"  p.{c.page_start:<2} {len(c.text):5d} chars  {c.section or '(unlabelled)'}")


if __name__ == "__main__":
    main()
