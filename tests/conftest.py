"""Shared fixtures for the offline test-suite (no network, no model downloads)."""

from __future__ import annotations

import re
import zlib

import numpy as np
import pytest

from src.ingest import Document, Line, Page


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
