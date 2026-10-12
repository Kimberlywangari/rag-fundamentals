"""Shared fixtures.

Offline tests (the default) need no network and download nothing.  Tests marked ``live`` call
Gemini and load the real embedding model; they are skipped unless ``pytest --live`` is given.
"""

from __future__ import annotations

import json
import re
import zlib
from datetime import datetime, timezone

import numpy as np
import pytest

from src import config
from src.ingest import Document, Line, Page


# --------------------------------------------------------------------- live-test plumbing
def pytest_addoption(parser):
    parser.addoption(
        "--live", action="store_true", default=False,
        help="also run live tests (Gemini API + embedding model; needs network and GEMINI_API_KEY)",
    )


def pytest_collection_modifyitems(config, items):  # noqa: D103 - pytest hook
    if config.getoption("--live"):
        return
    skip = pytest.mark.skip(reason="live test: run with `pytest --live` (needs network + GEMINI_API_KEY)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def pipeline():
    """The real pipeline over the persistent index (built on first use)."""
    try:
        config.get_gemini_api_key()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    from src.pipeline import RAGPipeline

    pipe = RAGPipeline()
    pipe.ensure_index()
    return pipe


@pytest.fixture(scope="session")
def recorder():
    """Collects one evidence record per live scenario and writes eval/live_results.json,
    which docs/RESULTS.md is generated from (so the results document cannot drift from what
    the tests actually saw)."""
    records: list[dict] = []
    yield records
    if records:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "gemini_model_preferred": config.GEMINI_MODEL,
            "chunking": {"strategy": config.CHUNK_STRATEGY, "size": config.CHUNK_SIZE,
                         "overlap": config.CHUNK_OVERLAP},
            "top_k": config.TOP_K,
            "min_similarity": config.MIN_SIMILARITY,
            "records": records,
        }
        config.EVAL_DIR.mkdir(exist_ok=True)
        (config.EVAL_DIR / "live_results.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )


class FakeEmbedder:
    """Deterministic bag-of-words embedder: hashes each word into one of 256 buckets and
    L2-normalises.  Two texts that share words get a high cosine, unrelated texts ~0, which
    is all the retrieval logic needs to be tested.  (crc32, not hash(): stable across runs.)"""

    dim = 256

    def encode(self, texts, batch_size: int = 64) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                out[i, zlib.crc32(word.encode()) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


def make_doc(pages, source: str = "Test Act.pdf") -> Document:
    """Build a Document from ``pages``: a list of pages, each a list of lines.  A line is a
    string, or a ``(text, "note")`` tuple for a margin heading."""
    built = []
    for number, lines in enumerate(pages, start=1):
        page = Page(source=source, page=number)
        for line in lines:
            text, kind = line if isinstance(line, tuple) else (line, "body")
            page.lines.append(Line(text, kind))
        built.append(page)
    return Document(source=source, pages=built, report={})


def words_text(n_words: int, per_line: int = 8, start: int = 0) -> list[str]:
    """``n_words`` distinct, short pseudo-words laid out ``per_line`` per line."""
    words = [f"w{start + i:04d}x" for i in range(n_words)]
    return [" ".join(words[i : i + per_line]) for i in range(0, n_words, per_line)]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
