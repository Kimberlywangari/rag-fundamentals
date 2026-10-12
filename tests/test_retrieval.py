"""Offline tests for embedding storage and top-k retrieval (real Chroma, fake embeddings)."""

from __future__ import annotations

import uuid

import chromadb
import numpy as np
import pytest

from src.chunking import chunk_fixed
from src.embed_store import build_index, get_client, open_collection
from src.retrieve import Retriever
from tests.conftest import make_doc

DOC_A = make_doc(
    [[
        "Breach notification. A data controller shall notify the Data Commissioner within "
        "seventy-two hours of becoming aware of a personal data breach.",
        "Penalties. A person who commits an offence is liable to a fine not exceeding "
        "three million shillings or imprisonment for ten years.",
        "Registration. No person shall act as a data controller unless registered with "
        "the Commissioner and holding a valid certificate.",
    ]],
    source="Data Protection Act.pdf",
)
DOC_B = make_doc(
    [[
        "Cyber harassment. A person who wilfully communicates with another person in a "
        "manner likely to cause fear of violence commits an offence.",
        "Phishing. A person who creates a website to induce a recipient to disclose "
        "personal information for an unlawful purpose commits an offence.",
    ]],
    source="Computer Misuse and Cybercrimes.pdf",
)


@pytest.fixture
def retriever(fake_embedder):
    chunks = chunk_fixed(DOC_A, 160, 30) + chunk_fixed(DOC_B, 160, 30)
    # chunk ids repeat across documents only if slugs collide; they must not
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    col = build_index(
        chunks, fake_embedder, client=chromadb.EphemeralClient(),
        name=f"t_{uuid.uuid4().hex}", reset=True,
    )
    return Retriever(fake_embedder, col)


def test_results_are_ranked_best_first(retriever):
    res = retriever.retrieve("notify the Data Commissioner of a personal data breach", k=4)
    assert [r.rank for r in res] == [1, 2, 3, 4]
    scores = [r.score for r in res]
    assert scores == sorted(scores, reverse=True)


def test_relevant_chunk_ranks_first(retriever):
    top = retriever.retrieve("within how many hours must the Data Commissioner be notified", k=3)[0]
    assert "seventy-two hours" in top.text
    assert top.source == "Data Protection Act.pdf"
    top = retriever.retrieve("website used to induce a recipient to disclose personal information", k=3)[0]
    assert "Phishing" in top.text
    assert top.source == "Computer Misuse and Cybercrimes.pdf"


def test_score_is_one_minus_cosine_distance_and_matches_numpy(retriever, fake_embedder):
    query = "fine not exceeding three million shillings"
    res = retriever.retrieve(query, k=3)
    q = fake_embedder.encode([query])[0]
    for r in res:
        v = fake_embedder.encode([r.text])[0]
        cosine = float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v)))
        assert r.score == pytest.approx(cosine, abs=1e-3)
        assert r.score == pytest.approx(1 - r.distance, abs=1e-3)


def test_exact_text_scores_about_one(retriever):
    chunk_text = retriever.collection.get(limit=1, include=["documents"])["documents"][0]
    best = retriever.retrieve(chunk_text, k=1)[0]
    assert best.score == pytest.approx(1.0, abs=1e-3)
    assert best.text == chunk_text


def test_provenance_metadata_is_returned(retriever):
    r = retriever.retrieve("registered with the Commissioner certificate", k=1)[0]
    assert r.source == "Data Protection Act.pdf"
    assert (r.page_start, r.page_end) == (1, 1)
    assert r.pages == "1"
    assert r.chunk_id.startswith("data-protection-act-fixed-")


def test_k_larger_than_collection_returns_everything(retriever):
    total = retriever.collection.count()
    assert len(retriever.retrieve("data", k=total + 50)) == total


def test_source_filter_restricts_results(retriever):
    res = retriever.retrieve("person commits an offence", k=10, source="Computer Misuse and Cybercrimes.pdf")
    assert res and {r.source for r in res} == {"Computer Misuse and Cybercrimes.pdf"}


def test_bad_arguments_raise(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("   ", k=3)
    with pytest.raises(ValueError):
        retriever.retrieve("data", k=0)


def test_persistent_index_survives_reopening(fake_embedder, tmp_path):
    chunks = chunk_fixed(DOC_A, 160, 30)
    build_index(chunks, fake_embedder, client=get_client(tmp_path), name="persist", reset=True)

    reopened = open_collection(get_client(tmp_path), "persist")  # a brand-new client object
    assert reopened.count() == len(chunks)
    top = Retriever(fake_embedder, reopened).retrieve("seventy-two hours breach", k=1)[0]
    assert "seventy-two hours" in top.text


def test_rebuilding_with_reset_does_not_duplicate(fake_embedder, tmp_path):
    chunks = chunk_fixed(DOC_A, 160, 30)
    client = get_client(tmp_path)
    build_index(chunks, fake_embedder, client=client, name="rebuild", reset=True)
    col = build_index(chunks, fake_embedder, client=client, name="rebuild", reset=True)
    assert col.count() == len(chunks)
