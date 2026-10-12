"""PDF ingestion: page-level text with page number + source filename, cleaned of layout noise.

Why geometry and not just ``page.get_text()``?  The three PDFs were produced by two different
toolchains and each has its own quirks (found by inspecting them, see docs/TECHNICAL_NOTES.md):

* Acts (Apache FOP): a section number and its heading are separate text runs on the same row;
  a running header sits at the top, a page number at the bottom; table-of-contents pages use
  dot leaders ("1. Short title ........"); the first pages are cover / licence boilerplate;
  words contain ligature glyphs ("Ofﬁce").
* Regulations (Adobe PDF Library): a running header carries the *printed* page number
  (1979, 1980 ...); section titles live in a narrow right-hand margin column; and 19 lines on
  10 pages (mostly defined terms) are drawn as vector outlines, so NO text extractor can read
  them.  Only those line regions are cropped and OCR'd; all native text is kept as is.

Output is a list of ``Document`` objects, each with ``Page`` objects whose ``lines`` carry a
``kind`` ("body" or "note" = margin heading) so the structure-aware chunker can use the notes.

Run ``python -m src.ingest`` to see an extraction report.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from src import config

log = logging.getLogger(__name__)

# --- Tunables (all in PDF points; A4 = 595 x 842) -----------------------------
HEADER_BAND = 0.15  # top 15% of the page may contain a running header
FOOTER_BAND = 0.92  # below 92% of the page height may contain a footer / page number
REPEAT_FRACTION = 0.25  # a band line seen on >= 25% of pages is "running" boilerplate
ROW_TOLERANCE = 4.5  # top edges closer than this share a row (half the ~11 pt line pitch)
ROW_JOIN_GAP = 8.0  # outlined paths / native runs closer than this on one row belong together
MARGIN_GAP = 4.0  # a line starting this far right of the body column is a margin note
NOTE_ROW_TOLERANCE = 6.0  # margin note must sit within this of the regulation row it labels
NOTE_CLUSTER_GAP = 12.5  # consecutive margin lines closer than this are one note

DOT_LEADER = re.compile(r"(\.\s?){5,}")  # "Short title ........ 3"
BOILERPLATE = re.compile(r"Creative Commons|Share widely and freely", re.IGNORECASE)
TOC_ROW_NO_DOT = re.compile(r"^\d{1,3}\s+[A-Z]")  # Regulations contents: "49 Prior consultation."
TOC_HEADING = re.compile(r"^(ARRANGEMENT OF (REGULATIONS|SECTIONS)|Regulation|Section|Contents)$")
NOTE_CITATION = re.compile(r"\s*(L\.\s?N\.\s?No\.\s?\d+\s+of\s+\d{4}\.|No\.\s?\d+\s+of\s+\d{4}\.)\s*$")
NEXT_INSTRUMENT = re.compile(r"^LEGAL NOTICE NO\.\s*\d+")  # a different statutory instrument starts
REG_START = re.compile(r"^\d{1,3}\.\s")  # "40. A data controller ..."
PAGE_NUMBER = re.compile(r"^\d{1,4}$")


@dataclass
class RawLine:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class Line:
    text: str
    kind: str = "body"  # "body" | "note" (margin heading belonging to the next row)


@dataclass
class Page:
    source: str
    page: int  # 1-based PDF page index (what a PDF viewer shows, NOT the printed number)
    lines: list[Line] = field(default_factory=list)
    method: str = "text"  # "text" (embedded text layer) | "text+ocr" (outlined lines patched in)
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


@dataclass
class Document:
    source: str
    pages: list[Page]
    report: dict

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.pages if p.text)


# --- Raw line extraction ------------------------------------------------------
def _native_lines(page: pymupdf.Page) -> list[RawLine]:
    out: list[RawLine] = []
    for block in page.get_text("dict")["blocks"]:
        for ln in block.get("lines", []):  # image blocks have no "lines"
            text = "".join(span["text"] for span in ln["spans"])
            x0, y0, x1, y1 = ln["bbox"]
            out.append(RawLine(text, x0, y0, x1, y1))
    return out


def _outlined_rects(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """Bounding boxes of filled, text-line-sized vector paths made of curves.

    A glyph drawn as an outline (instead of a font reference) is a filled path of Bezier
    curves.  Text extractors only read font-based text, so such glyphs are invisible to them.
    """
    rects = []
    for drawing in page.get_drawings():
        rect = drawing["rect"]
        if drawing.get("fill") is None or rect.height >= 30 or rect.width < 6:
            continue
        if sum(1 for item in drawing["items"] if item[0] == "c") >= 6:
            rects.append(rect)
    return rects


_ocr_engine = None


def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr import RapidOCR  # imported lazily: heavy and only needed for a few lines

        _ocr_engine = RapidOCR(params={"Global.log_level": "warning"})
    return _ocr_engine


def _y_overlap(a, b) -> float:
    """Vertical overlap of two boxes as a fraction of the shorter one's height."""
    inter = min(a.y1, b.y1) - max(a.y0, b.y0)
    return max(inter, 0.0) / max(min(a.y1 - a.y0, b.y1 - b.y0), 1e-6)


def _outlined_bands(page: pymupdf.Page, natives: list[RawLine]) -> list[pymupdf.Rect]:
    """One box per text row that contains outlined glyphs.

    Outlined paths on the same row are merged, and the box is stretched to the right over the
    native text that continues the line (a line can be outlined only up to some letter, e.g.
    'identific' outlined + 'ation number' native), so the whole row can be OCR'd in one piece.
    """
    rects = sorted(_outlined_rects(page), key=lambda r: (r.y0, r.x0))
    merged: list[pymupdf.Rect] = []
    for r in rects:
        if merged and _y_overlap(merged[-1], r) > 0.5 and r.x0 - merged[-1].x1 <= ROW_JOIN_GAP:
            prev = merged[-1]
            merged[-1] = pymupdf.Rect(
                min(prev.x0, r.x0), min(prev.y0, r.y0), max(prev.x1, r.x1), max(prev.y1, r.y1)
            )
        else:
            merged.append(pymupdf.Rect(r))
    bands = []
    for r in merged:
        right = r.x1
        for n in sorted(natives, key=lambda n: n.x0):
            if (
                n.text.strip()
                and n.x0 >= r.x0 - 1
                and _y_overlap(n, r) > 0.5
                and n.x0 <= right + ROW_JOIN_GAP
            ):
                right = max(right, n.x1)
        bands.append(pymupdf.Rect(r.x0 - 0.5, r.y0, right, r.y1))
    return bands


def _words(text: str) -> list[str]:
    """Lower-case word / number tokens; apostrophes dropped so "subject's" -> "subjects"."""
    text = unicodedata.normalize("NFKC", text).lower()
    return re.findall(r"[a-z]+|\d+", re.sub(r"['’`]", "", text))


def _vocabulary(native_pages: list[list[RawLine]]) -> set[str]:
    """Tokens found in the PDF's own text layer: a free spell-checker for OCR output."""
    vocab = {"a", "i"}  # the only legitimate one-letter words
    for lines in native_pages:
        for ln in lines:
            vocab.update(w for w in _words(ln.text) if len(w) >= 2 or w.isdigit())
    return vocab


def _vocab_score(text: str, vocab: set[str]) -> float:
    """Fraction of the OCR output's words that the document itself uses (0..1).

    Stray one-letter tokens ("entity t that") count against the score.
    """
    words = _words(text)
    return sum(w in vocab for w in words) / len(words) if words else 0.0


def _ocr_once(
    page: pymupdf.Page, rect: pymupdf.Rect, dpi: int, vpad: float, border: int, hpad: float,
    left_abuts: bool,
) -> str:
    import numpy as np

    left = 0.0 if left_abuts else hpad  # never pull in the half-glyph of a native neighbour
    clip = pymupdf.Rect(rect.x0 - left, rect.y0 - vpad, rect.x1 + hpad, rect.y1 + vpad) & page.rect
    pix = page.get_pixmap(dpi=dpi, clip=clip)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    img = np.pad(img, ((border, border), (border, border), (0, 0)), constant_values=255)
    result = _get_ocr_engine()(img)
    boxes = result.boxes if result.boxes is not None else []
    pieces = sorted(
        (min(float(p[0]) for p in box), str(text)) for text, box in zip(result.txts or [], boxes)
    )
    return " ".join(t for _, t in pieces).strip()


# (dpi, vertical pad pt, white border px, horizontal pad pt).  The engine is sensitive to
# these, so every band is read under all of them and the outputs vote (see _ocr_rect).
OCR_SETTINGS = [(200, 2, 60, 4), (200, 2, 60, 0), (150, 2, 60, 4), (300, 2, 60, 4), (200, 6, 24, 4)]


def _quote_key(text: str) -> str:
    """Comparison key that ignores which kind of quote mark the OCR chose."""
    return re.sub(r"[“”‘’\"'`]+", '"', text)


def _ocr_rect(
    pdf_hash: str,
    page: pymupdf.Page,
    page_no: int,
    rect: pymupdf.Rect,
    vocab: set[str],
    left_abuts: bool = False,
) -> str:
    """OCR one row band (cached on disk).

    Cropping to the damaged row is far more accurate than OCR of a whole page, and leaves all
    the native text untouched.  The engine is brittle (one stray letter, a doubled quote), so
    the band is read under several crop settings and the answer is chosen by vote:
    keep the outputs with the best vocabulary score, then take the most common one.
    """
    cache = (
        config.LLM_CACHE_DIR / "ocr"
        / f"{pdf_hash}_p{page_no}_b{round(rect.x0)}_{round(rect.y0)}_{round(rect.x1)}.txt"
    )
    if cache.exists():
        return cache.read_text(encoding="utf-8")

    outputs: list[str] = []
    scores: list[float] = []
    for dpi, vpad, border, hpad in OCR_SETTINGS:
        raw = _ocr_once(page, rect, dpi, vpad, border, hpad, left_abuts)
        text = re.sub(r'(["”])["”]+', r"\1", " ".join(raw.split()))  # OCR doubles closing quotes
        outputs.append(text)
        scores.append(_vocab_score(text, vocab))
        agree = sum(
            1 for o, s in zip(outputs, scores) if s == 1.0 and _quote_key(o) == _quote_key(text)
        )
        if scores[-1] == 1.0 and agree >= 2:
            break  # two independent reads agree and every word is a known word: done
    finalists = [t for t, s in zip(outputs, scores) if s == max(scores)]
    votes = Counter(_quote_key(t) for t in finalists)
    top_key = max(votes, key=votes.get)  # ties: first finalist in settings order
    best_text = next(t for t in finalists if _quote_key(t) == top_key)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(best_text, encoding="utf-8")
    return best_text


def _overlap_fraction(ln: RawLine, rect: pymupdf.Rect) -> float:
    """Share of the text line's box that lies inside ``rect``."""
    w = min(ln.x1, rect.x1) - max(ln.x0, rect.x0)
    h = min(ln.y1, rect.y1) - max(ln.y0, rect.y0)
    area = max(ln.x1 - ln.x0, 1e-6) * max(ln.y1 - ln.y0, 1e-6)
    return max(w, 0) * max(h, 0) / area


# --- Cleaning helpers ---------------------------------------------------------
def _normalise(text: str) -> str:
    """NFKC turns ligature glyphs (U+FB01 'ﬁ') and no-break spaces into plain characters."""
    text = unicodedata.normalize("NFKC", text).replace("­", "")
    return " ".join(text.split())


def _key(text: str) -> str:
    """Letters-only key so 'Kenya Subsidiary L egislation, 2021' matches across pages."""
    return re.sub(r"[\d\W_]+", "", text.lower())


def _merge_rows(lines: list[RawLine]) -> list[RawLine]:
    """Join text runs that share a visual row (e.g. '8.' + 'Functions of the Office')."""
    rows: list[list[RawLine]] = []
    for ln in sorted(lines, key=lambda r: (r.y0, r.x0)):
        if rows and abs(ln.y0 - rows[-1][0].y0) <= ROW_TOLERANCE:
            rows[-1].append(ln)
        else:
            rows.append([ln])
    merged = []
    for row in rows:
        row.sort(key=lambda r: r.x0)
        # Fragments that touch (gap < 1 pt) are one word cut in two by an outline/native seam,
        # e.g. native "subjec" + recovered "t's": join without a space.
        text = row[0].text
        for prev, cur in zip(row, row[1:]):
            sep = "" if cur.x0 - prev.x1 < 1.0 and prev.text[-1:].isalpha() and cur.text[:1].isalpha() else " "
            text += sep + cur.text
        merged.append(
            RawLine(
                text,
                min(r.x0 for r in row),
                min(r.y0 for r in row),
                max(r.x1 for r in row),
                max(r.y1 for r in row),
            )
        )
    return merged


def _cluster_notes(notes: list[RawLine]) -> list[RawLine]:
    """Consecutive margin lines form one note, e.g. 'General principles' / 'for transfers of'.

    Trailing legal citations ("L. N. No.196 of 2020.", "No. 24 of 2019.") are cross-references,
    not part of the heading, and are stripped.
    """
    clusters: list[list[RawLine]] = []
    for ln in sorted(notes, key=lambda r: r.y0):
        if clusters and ln.y0 - clusters[-1][-1].y0 <= NOTE_CLUSTER_GAP:
            clusters[-1].append(ln)
        else:
            clusters.append([ln])
    out = []
    for c in clusters:
        text = " ".join(r.text for r in c)
        while True:
            stripped = NOTE_CITATION.sub("", text).strip()
            if stripped == text:
                break
            text = stripped
        out.append(RawLine(text, c[0].x0, c[0].y0, c[-1].x1, c[-1].y1))
    return out


def _body_right_edge(lines: list[RawLine], fallback: float = float("inf")) -> float:
    """Right edge of the main text column = 80th percentile of x1 over long lines.

    Margin notes are short, so they never enter the sample.  Computed per page because the
    Regulations mix a narrow body column (to ~406 pt) with full-width forms on later pages.
    """
    xs = sorted(ln.x1 for ln in lines if len(ln.text) >= 40)
    if len(xs) < 5:
        return fallback
    return xs[int(0.8 * (len(xs) - 1))]


def _join_hyphenated(lines: list[Line]) -> list[Line]:
    """'sub-' / 'regulation (1)' -> 'sub-regulation (1)' (lowercase continuation only)."""
    out: list[Line] = []
    for line in lines:
        prev = out[-1] if out else None
        if (
            prev is not None
            and prev.kind == "body"
            and line.kind == "body"
            and re.search(r"[A-Za-z]-$", prev.text)
            and line.text[:1].islower()
        ):
            out[-1] = Line(prev.text + line.text, "body")
        else:
            out.append(line)
    return out


# --- Main entry point ---------------------------------------------------------
def ingest_pdf(pdf_path: Path, ocr: bool = True) -> Document:
    pdf_path = Path(pdf_path)
    pdf_hash = hashlib.sha1(pdf_path.read_bytes()).hexdigest()[:16]
    doc = pymupdf.open(str(pdf_path))
    n_pages = len(doc)

    # Pass 1: native text lines per page; OCR only the outlined (non-text) line regions.
    raw: list[tuple[Page, float, list[RawLine]]] = []
    ocr_available = True
    native_pages = [_native_lines(pg) for pg in doc]
    vocab = _vocabulary(native_pages)
    for i, (pg, lines) in enumerate(zip(doc, native_pages), start=1):
        page = Page(source=pdf_path.name, page=i)
        rects = _outlined_bands(pg, lines)
        if rects:
            recovered: list[RawLine] = []
            if ocr and ocr_available:
                try:
                    for rect in rects:
                        abuts = any(
                            n.text.strip() and abs(n.x1 - rect.x0) <= 1.5 and _y_overlap(n, rect) > 0.5
                            for n in lines
                        )
                        text = _ocr_rect(pdf_hash, pg, i, rect, vocab, abuts)
                        if text:
                            recovered.append(RawLine(text, rect.x0, rect.y0, rect.x1, rect.y1))
                except ImportError:
                    ocr_available = False
                    recovered = []
            if recovered:
                # drop native fragments that belong to an outlined line (e.g. a stray "nal")
                lines = [
                    ln for ln in lines if not any(_overlap_fraction(ln, r) > 0.5 for r in rects)
                ] + recovered
                page.method = "text+ocr"
                page.warnings.append(f"{len(recovered)} outlined line(s) recovered by OCR")
            else:
                page.warnings.append(
                    f"{len(rects)} outlined-glyph path(s) are NOT extractable "
                    "(OCR disabled or unavailable): some words are missing"
                )
        raw.append((page, pg.rect.height, lines))

    doc_right_edge = _body_right_edge([ln for _, _, lines in raw for ln in lines])

    # Pass 2: which short band lines repeat on many pages? (running headers / footers)
    band_counts: Counter[str] = Counter()
    for _, height, lines in raw:
        keys = {
            _key(_normalise(ln.text))
            for ln in lines
            if ln.y0 < HEADER_BAND * height or ln.y0 > FOOTER_BAND * height
        }
        band_counts.update(k for k in keys if k)
    repeat_threshold = max(2, math.ceil(REPEAT_FRACTION * n_pages))
    running = {k for k, c in band_counts.items() if c >= repeat_threshold}

    stats: Counter[str] = Counter()
    for page, height, lines in raw:
        cleaned: list[RawLine] = []
        for ln in lines:
            text = _normalise(ln.text)
            if not text:
                continue
            ln = RawLine(text, ln.x0, ln.y0, ln.x1, ln.y1)
            in_band = ln.y0 < HEADER_BAND * height or ln.y0 > FOOTER_BAND * height
            if in_band and (PAGE_NUMBER.match(text) or _key(text) in running):
                stats["header_footer_lines_removed"] += 1
            elif DOT_LEADER.search(text):
                stats["dot_leader_lines_removed"] += 1  # TOC rows and blank form fields
            else:
                cleaned.append(ln)

        right_edge = _body_right_edge(cleaned, fallback=doc_right_edge)
        body = [ln for ln in cleaned if ln.x0 <= right_edge + MARGIN_GAP]
        notes = [ln for ln in cleaned if ln.x0 > right_edge + MARGIN_GAP]

        rows = []
        page_toc_rows = 0
        for row in _merge_rows(body):
            # Regulations contents row, e.g. "51 Prior consultation" (real body rows read "51. ").
            if TOC_ROW_NO_DOT.match(row.text) and row.x0 < 140 and len(row.text) < 110:
                page_toc_rows += 1
            elif TOC_HEADING.match(row.text):
                page_toc_rows += 1
            else:
                rows.append(row)
        stats["toc_rows_removed"] += page_toc_rows
        if page_toc_rows >= 5 and sum(len(r.text) for r in rows) < 450:
            # what is left of a contents page is only its Part headings: drop the page text
            page.warnings.append("contents page dropped")
            stats["contents_pages_dropped"] += 1
            page.lines = []
            continue
        if BOILERPLATE.search(" ".join(r.text for r in rows)):
            page.warnings.append("licence / cover boilerplate page dropped")
            stats["boilerplate_pages_dropped"] += 1
            page.lines = []
            continue

        # Attach each margin note to the regulation row it labels (nearest start-of-regulation).
        attach: dict[int, str] = {}
        for note in _cluster_notes(notes):
            best, best_d = None, NOTE_ROW_TOLERANCE + 1
            for idx, row in enumerate(rows):
                d = abs(row.y0 - note.y0)
                if REG_START.match(row.text) and d < best_d:
                    best, best_d = idx, d
            if not note.text:
                stats["margin_notes_dropped"] += 1
            elif best is not None and best_d <= NOTE_ROW_TOLERANCE and best not in attach:
                attach[best] = note.text
                stats["margin_notes_attached"] += 1
            else:
                stats["margin_notes_dropped"] += 1  # cross-references like "No. 24 of 2019."

        assembled: list[Line] = []
        for idx, row in enumerate(rows):
            if idx in attach:
                assembled.append(Line(attach[idx], kind="note"))
            assembled.append(Line(row.text, kind="body"))
        page.lines = _join_hyphenated(assembled)

    pages = [p for p, _, _ in raw]

    # A file can end with the *start of the next instrument* (the Regulations PDF ends with the
    # contents list of Legal Notice No. 264). Everything from there on is not this document.
    cut = False
    for page in pages[1:]:  # page 1 legitimately carries this document's own Legal Notice number
        if not cut:
            for idx, line in enumerate(page.lines):
                if NEXT_INSTRUMENT.match(line.text):
                    stats["next_instrument_lines_dropped"] += len(page.lines) - idx
                    page.lines = page.lines[:idx]
                    page.warnings.append("tail of a different instrument dropped")
                    cut = True
                    break
        else:
            stats["next_instrument_lines_dropped"] += len(page.lines)
            page.lines = []

    report = {
        "source": pdf_path.name,
        "pages": n_pages,
        "chars": sum(len(p.text) for p in pages),
        "empty_pages": [p.page for p in pages if not p.text],
        "ocr_patched_pages": [p.page for p in pages if p.method == "text+ocr"],
        "unrepaired_pages": [
            p.page for p in pages if any("NOT extractable" in w for w in p.warnings)
        ],
        "body_right_edge_pt": round(doc_right_edge, 1),
        **stats,
    }
    return Document(source=pdf_path.name, pages=pages, report=report)


def load_corpus(raw_dir: Path | None = None, ocr: bool = True) -> list[Document]:
    raw_dir = Path(raw_dir) if raw_dir else config.RAW_DATA_DIR
    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {raw_dir}")
    return [ingest_pdf(p, ocr=ocr) for p in pdfs]


# --- CLI report ---------------------------------------------------------------
def _print_report(docs: list[Document]) -> None:
    for d in docs:
        print("=" * 78)
        for key, value in d.report.items():
            print(f"  {key:30}: {value}")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the PDFs and print an extraction report.")
    parser.add_argument("--no-ocr", action="store_true", help="skip the OCR fallback")
    parser.add_argument("--show", metavar="SOURCE:PAGE", help='print one cleaned page, e.g. "Data Protection Act.pdf:10"')
    args = parser.parse_args()

    docs = load_corpus(ocr=not args.no_ocr)
    _print_report(docs)
    if args.show:
        src, _, num = args.show.rpartition(":")
        for d in docs:
            if d.source == src:
                page = d.pages[int(num) - 1]
                print(f"\n--- {src} page {page.page} ({page.method}) ---")
                for line in page.lines:
                    print(("  [NOTE] " if line.kind == "note" else "  ") + line.text)


if __name__ == "__main__":
    main()
