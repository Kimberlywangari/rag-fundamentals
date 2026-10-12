"""Offline tests for ingestion helpers and for the real Data Protection Act PDF (local file)."""

from __future__ import annotations

import pytest

from src import config
from src.ingest import (
    Line,
    RawLine,
    _cluster_notes,
    _join_hyphenated,
    _merge_rows,
    _normalise,
    _vocab_score,
    _vocabulary,
    ingest_pdf,
)


def raw(text, x0, y0, x1=None):
    return RawLine(text, x0, y0, x1 if x1 is not None else x0 + 5 * len(text), y0 + 9)


def test_ligatures_and_spaces_are_normalised():
    assert _normalise("Ofﬁce  of the ﬁrst") == "Office of the first"


def test_split_text_runs_on_one_row_are_joined_left_to_right():
    rows = _merge_rows([raw("Functions of the Office", 98, 98.4), raw("8.", 80, 98.4)])
    assert [r.text for r in rows] == ["8. Functions of the Office"]


def test_different_rows_stay_separate():
    rows = _merge_rows([raw("first", 70, 100), raw("second", 70, 112)])
    assert [r.text for r in rows] == ["first", "second"]


def test_touching_fragments_are_joined_without_a_space():
    a = RawLine("the data subjec", 160, 412, 239.0, 421)
    b = RawLine("t's account", 238.6, 416, 320, 424)
    assert _merge_rows([a, b])[0].text == "the data subject's account"


def test_hyphenated_line_break_is_rejoined_only_before_lowercase():
    lines = [Line("a transfer in reliance on sub-"), Line("regulation (1)"),
             Line("the relevant two-"), Line("Part III applies")]
    assert [l.text for l in _join_hyphenated(lines)] == [
        "a transfer in reliance on sub-regulation (1)", "the relevant two-", "Part III applies",
    ]


def test_margin_note_citations_are_stripped():
    notes = [raw("Exemption.", 418, 351), raw("L. N. No.196 of 2020.", 418, 359)]
    assert [n.text for n in _cluster_notes(notes)] == ["Exemption."]
    assert [n.text for n in _cluster_notes([raw("No. 24 of 2019.", 418, 320)])] == [""]


def test_ocr_vocabulary_check_penalises_stray_letters_and_garbage():
    vocab = _vocabulary([[RawLine("an entity that transfers personal data", 0, 0, 1, 1)]])
    assert _vocab_score("an entity that transfers personal data", vocab) == 1.0
    assert _vocab_score("an entity t that transfers personal data", vocab) < 1.0
    assert _vocab_score("rnnsrd sers tes ns n nnos", vocab) == 0.0


# ------------------------------------------------- the real PDF (local file, no OCR needed)
@pytest.fixture(scope="module")
def dpa():
    return ingest_pdf(config.RAW_DATA_DIR / "Data Protection Act.pdf", ocr=False)


def test_real_pdf_page_count_and_front_matter(dpa):
    assert len(dpa.pages) == 32
    assert dpa.pages[1].text == ""  # licence boilerplate page dropped
    assert dpa.report["empty_pages"] == [2, 3, 4, 5]  # licence + contents pages


def test_real_pdf_section_text_and_page_numbers(dpa):
    page22 = dpa.pages[21]
    assert page22.page == 22
    assert "within seventy-two hours of becoming aware of" in page22.text
    assert "Office" in dpa.pages[9].text  # 'Ofﬁce' ligature normalised to 'Office'


def test_real_pdf_headers_and_page_numbers_are_removed(dpa):
    for page in dpa.pages[5:]:
        assert "Data Protection Act (Cap. 411C)" not in page.text
        assert not any(line.text.strip().isdigit() for line in page.lines)
