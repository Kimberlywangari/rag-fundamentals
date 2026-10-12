"""The whole RAG pipeline in one place:

    PDFs -> ingest -> chunk -> embed -> Chroma          (build, once)
    question -> embed -> top-k chunks -> prompt -> Gemini -> grounded answer   (ask, per query)

``RAGPipeline.ask`` returns everything the demo and the tests need (the ranked chunks with
scores, the answer, and optionally the no-context baseline answer) so each stage can be
inspected rather than trusted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src import config
from src.chunking import chunk_corpus
from src.embed_store import Embedder, build_index, get_client, open_collection
from src.generate import Answer, Generator
from src.ingest import load_corpus
from src.retrieve import RetrievedChunk, Retriever

_HEADING = re.compile(r"^(\d{1,3}[A-Z]?)\.\s+([A-Z][^\n]{2,90})$", re.MULTILINE)


def index_signature() -> dict:
    """The settings that determine what is in the index; stored on the collection."""
    return {
        "embedding_model": config.EMBEDDING_MODEL,
        "chunk_strategy": config.CHUNK_STRATEGY,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "structure_max_size": config.STRUCTURE_MAX_SIZE,
    }


def headings_in(text: str) -> list[str]:
    """Section headings that appear inside a chunk (a display aid for fixed-size chunks,
    which carry no section label of their own)."""
    return [f"{num}. {title.strip()}" for num, title in _HEADING.findall(text)]


@dataclass
class PipelineResult:
    question: str
    chunks: list[RetrievedChunk]
    answer: Answer
    base_answer: Answer | None = None


class RAGPipeline:
    def __init__(self, embedder: Embedder | None = None, generator: Generator | None = None):
        self._embedder = embedder
        self.generator = generator or Generator()
        self._retriever: Retriever | None = None

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    # -- index ---------------------------------------------------------------------
    def build_index(self) -> int:
        """Ingest, chunk, embed and store everything. Returns the number of chunks."""
        docs = load_corpus()
        chunks = chunk_corpus(docs)
        collection = build_index(
            chunks, self.embedder, client=get_client(), metadata=index_signature()
        )
        self._retriever = Retriever(self.embedder, collection)
        return collection.count()

    def ensure_index(self, rebuild: bool = False) -> None:
        """Use the persisted index if it exists and matches the current config, else rebuild."""
        if not rebuild:
            try:
                collection = open_collection(get_client())
                stored = {k: (collection.metadata or {}).get(k) for k in index_signature()}
                if collection.count() > 0 and stored == index_signature():
                    self._retriever = Retriever(self.embedder, collection)
                    return
            except Exception:  # noqa: BLE001 - no collection yet
                pass
        self.build_index()

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self.ensure_index()
        return self._retriever  # type: ignore[return-value]

    # -- question answering -----------------------------------------------------------
    def ask(
        self,
        question: str,
        k: int = config.TOP_K,
        with_base: bool = False,
        use_cache: bool = True,
        min_score: float | None = None,
    ) -> PipelineResult:
        chunks = self.retriever.retrieve(question, k=k)
        answer = self.generator.answer(question, chunks, min_score=min_score, use_cache=use_cache)
        base = self.generator.answer_without_context(question, use_cache=use_cache) if with_base else None
        return PipelineResult(question=question, chunks=chunks, answer=answer, base_answer=base)
