"""Offline tests for fixed-size and structure-aware chunking."""

from __future__ import annotations

import re

import pytest

from src.chunking import assemble_text, chunk_fixed, chunk_structure
from tests.conftest import make_doc, words_text


# --------------------------------------------------------------------------- fixed-size
@pytest.fixture
def doc():
    # 300 words over two pages: a chunk will have to straddle the page break
    return make_doc([words_text(150), words_text(150, start=150)])


def test_chunks_never_exceed_size(doc):
    for size, overlap in [(100, 0), (100, 30), (250, 50)]:
        assert all(len(c.text) <= size for c in chunk_fixed(doc, size, overlap))


def test_chunk_text_is_exact_slice_of_document(doc):
    text, _ = assemble_text(doc)
    for c in chunk_fixed(doc, 120, 40):
        assert text[c.char_start : c.char_end] == c.text


def test_chunks_never_cut_a_word_in_half(doc):
    all_words = set(re.findall(r"\S+", assemble_text(doc)[0]))
    for c in chunk_fixed(doc, 97, 31):  # awkward numbers on purpose
        assert set(c.text.split()) <= all_words


def test_overlap_zero_shares_nothing_and_loses_nothing(doc):
    chunks = chunk_fixed(doc, 100, 0)
    for a, b in zip(chunks, chunks[1:]):
        assert b.char_start >= a.char_end  # no shared characters
    covered = [w for c in chunks for w in c.text.split()]
    assert covered == re.findall(r"\S+", assemble_text(doc)[0])  # every word exactly once


def test_overlap_is_shared_and_bounded(doc):
    overlap = 40
    chunks = chunk_fixed(doc, 120, overlap)
    for a, b in zip(chunks, chunks[1:]):
        shared = a.char_end - b.char_start
        assert 0 < shared <= overlap  # snapped to a word start, never more than requested
        assert shared > overlap - 10  # words are 6 chars: snapping costs at most one word
        assert a.text.endswith(b.text.split()[0]) or b.text.split()[0] in a.text


def test_overlap_makes_more_chunks_than_none(doc):
    assert len(chunk_fixed(doc, 100, 40)) > len(chunk_fixed(doc, 100, 0))


def test_every_word_is_covered_with_overlap(doc):
    text, _ = assemble_text(doc)
    chunks = chunk_fixed(doc, 100, 40)
    covered = set()
    for c in chunks:
        covered.update(m.start() + c.char_start for m in re.finditer(r"\S+", c.text))
    assert covered == {m.start() for m in re.finditer(r"\S+", text)}


@pytest.mark.parametrize(
    "size, overlap", [(0, 0), (-5, 0), (100, -1), (100, 100), (100, 150)]
)
def test_invalid_parameters_raise(doc, size, overlap):
    with pytest.raises(ValueError):
        chunk_fixed(doc, size, overlap)


def test_metadata_source_pages_and_ids(doc):
    chunks = chunk_fixed(doc, 150, 30)
    assert {c.source for c in chunks} == {"Test Act.pdf"}
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 2
    # at least one chunk straddles the page break
    assert any(c.page_start == 1 and c.page_end == 2 for c in chunks)
    # pages are monotonic
    assert all(a.page_start <= b.page_start for a, b in zip(chunks, chunks[1:]))


def test_to_metadata_is_chroma_compatible(doc):
    meta = chunk_fixed(doc, 150, 30)[0].to_metadata()
    assert all(isinstance(v, (str, int, float, bool)) for v in meta.values())


def test_chunking_is_deterministic(doc):
    assert chunk_fixed(doc, 120, 40) == chunk_fixed(doc, 120, 40)


def test_empty_pages_are_skipped_and_page_numbers_stay_true():
    doc = make_doc([["Cover page text here."], [], ["Real content on page three."]])
    chunks = chunk_fixed(doc, 500, 0)
    assert len(chunks) == 1
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 3)


# ------------------------------------------------------------------- structure-aware
ACT_LINES = [
    "THE TEST ACT",
    "Part I – PRELIMINARY",
    "1. Short title",
    "This Act may be cited as the Test Act.",
    "2. Interpretation",
    "In this Act—",
    '"data" means information;',
    '"person" means a natural person;',
    "Part II – THE OFFICE",
    "3. Establishment of the Office",
    "(1) There is established an Office.",
    "(2) The Office is independent.",
    "FIRST SCHEDULE",
    "1. The oath of office.",  # a list item in a schedule, NOT section 1 again
    "2. The second item.",
]


def labels(chunks):
    return [c.section for c in chunks]


def test_sections_are_detected_and_labelled():
    chunks = chunk_structure(make_doc([ACT_LINES]), max_size=2000)
    assert labels(chunks) == [
        "",
        "Section 1: Short title",
        "Section 2: Interpretation",
        "Section 3: Establishment of the Office",
        "FIRST SCHEDULE",
    ]


def test_schedule_list_items_do_not_restart_section_numbering():
    chunks = chunk_structure(make_doc([ACT_LINES]), max_size=2000)
    assert labels(chunks).count("Section 1: Short title") == 1
    schedule = next(c for c in chunks if c.section == "FIRST SCHEDULE")
    assert "The second item." in schedule.text


def test_part_headings_belong_to_no_chunk():
    chunks = chunk_structure(make_doc([ACT_LINES]), max_size=2000)
    assert not any("PRELIMINARY" in c.text or "THE OFFICE" in c.text for c in chunks)


def test_chunk_holds_exactly_its_section():
    chunks = chunk_structure(make_doc([ACT_LINES]), max_size=2000)
    sec2 = next(c for c in chunks if c.section.startswith("Section 2"))
    assert '"data" means information;' in sec2.text
    assert "Establishment" not in sec2.text
    assert "Short title" not in sec2.text


def test_oversized_section_is_split_but_heading_is_never_stranded():
    items = [f"({i}) Provision number {i} says something fairly long indeed, filler words here."
             for i in range(1, 21)]
    doc = make_doc([["1. Big section", *items]])
    chunks = chunk_structure(doc, max_size=300)
    assert len(chunks) > 1
    assert all(c.section == "Section 1: Big section" for c in chunks)
    assert chunks[0].text.startswith("1. Big section")
    assert len(chunks[0].text) > len("1. Big section") + 20  # heading kept with a body unit
    assert all(c.text.startswith("Section 1: Big section (continued)") for c in chunks[1:])
    assert all(len(c.text) <= 300 * 1.25 for c in chunks)


def test_gaps_for_repealed_sections_and_lettered_insertions_are_accepted():
    doc = make_doc([["1. One", "text", "2. Two", "text", "4. Four", "text", "4A. Four-A", "text"]])
    assert labels(chunk_structure(doc, 1000)) == [
        "Section 1: One", "Section 2: Two", "Section 4: Four", "Section 4A: Four-A",
    ]


def test_regulation_margin_note_becomes_the_title():
    doc = make_doc(
        [[("Citation.", "note"), "1. These Regulations may be cited as the Test Regulations.",
          ("Interpretation.", "note"), "2. In these Regulations, unless the context requires—"]],
        source="Test Regulations.pdf",
    )
    chunks = chunk_structure(doc, 1000)
    assert labels(chunks) == ["Regulation 1: Citation", "Regulation 2: Interpretation"]
    assert chunks[0].text.startswith("Citation.")  # the heading travels with the text


def test_structure_chunks_carry_pages():
    doc = make_doc([["1. First", "page one text"], ["more of section one", "2. Second", "text"]])
    chunks = chunk_structure(doc, 1000)
    first = next(c for c in chunks if c.section.startswith("Section 1"))
    second = next(c for c in chunks if c.section.startswith("Section 2"))
    assert (first.page_start, first.page_end) == (1, 2)
    assert (second.page_start, second.page_end) == (2, 2)
