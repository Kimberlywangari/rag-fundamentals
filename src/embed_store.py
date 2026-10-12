"""Embedding + vector storage.

What an embedding is
    A neural network (here all-MiniLM-L6-v2, a small sentence-transformer) reads a piece of text
    and outputs a fixed-length list of numbers: 384 of them.  Texts that *mean* similar things
    land close together in that 384-dimensional space; unrelated texts land far apart.  The
    numbers have no individual meaning; only distances / angles between vectors do.

Cosine similarity
    cos(a, b) = (a . b) / (|a| |b|)  is the cosine of the angle between two vectors: 1 = same
    direction (same meaning), 0 = unrelated, -1 = opposite.  We L2-normalise every vector so
    |a| = |b| = 1 and cosine reduces to a plain dot product.

How ChromaDB reports it
    With ``space="cosine"`` Chroma returns a *distance* = 1 - cosine_similarity (0 = identical,
    1 = orthogonal, 2 = opposite).  Our score is therefore  ``similarity = 1 - distance``.
    ``python -m src.embed_store --explain`` proves this numerically.

Run ``python -m src.embed_store --build`` to (re)build the persistent index.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Protocol

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import chromadb  # noqa: E402
from chromadb.config import Settings  # noqa: E402

from src import config  # noqa: E402
from src.chunking import Chunk, chunk_corpus  # noqa: E402
from src.ingest import load_corpus  # noqa: E402


class SupportsEncode(Protocol):
    """Anything with ``encode(list[str]) -> ndarray``; lets tests inject a fake offline model."""

    def encode(self, texts: list[str]) -> np.ndarray: ...


class Embedder:
    """Thin wrapper over a local sentence-transformers model that returns unit-length vectors."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer  # heavy import, kept lazy

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        getter = getattr(self.model, "get_embedding_dimension", None) or (
            self.model.get_sentence_embedding_dimension
        )
        return int(getter())

    @property
    def max_tokens(self) -> int:
        """Inputs longer than this many word-pieces are silently truncated by the model."""
        return int(self.model.max_seq_length)

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # unit length => cosine == dot product
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def token_lengths(self, texts: list[str]) -> list[int]:
        """Word-piece count of each text (to see which chunks the model would truncate)."""
        tok = self.model.tokenizer
        return [len(ids) for ids in tok(texts, add_special_tokens=True, truncation=False)["input_ids"]]


# --- ChromaDB -------------------------------------------------------------------
def get_client(path: Path | str | None = None) -> chromadb.ClientAPI:
    """Persistent local client; ``path=None`` uses config.CHROMA_DIR (git-ignored)."""
    target = Path(path) if path else config.CHROMA_DIR
    target.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(target), settings=Settings(anonymized_telemetry=False))


def create_collection(client: chromadb.ClientAPI, name: str = config.COLLECTION_NAME, reset: bool = True):
    """Create a cosine-space collection. ``embedding_function=None``: we always supply vectors,
    so Chroma can never silently embed with a different model."""
    if reset:
        try:
            client.delete_collection(name)
        except Exception:  # noqa: BLE001 - collection did not exist yet
            pass
    return client.get_or_create_collection(
        name=name,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )


def open_collection(client: chromadb.ClientAPI, name: str = config.COLLECTION_NAME):
    return client.get_collection(name=name, embedding_function=None)


def add_chunks(collection, chunks: list[Chunk], embedder: SupportsEncode, batch_size: int = 128) -> None:
    """Embed ``chunks`` and store text + vector + provenance metadata under the chunk id."""
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectors = embedder.encode([c.text for c in batch])
        collection.add(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            embeddings=vectors.tolist(),
            metadatas=[c.to_metadata() for c in batch],
        )


def build_index(
    chunks: list[Chunk],
    embedder: SupportsEncode,
    client: chromadb.ClientAPI | None = None,
    name: str = config.COLLECTION_NAME,
    reset: bool = True,
):
    client = client or get_client()
    collection = create_collection(client, name, reset=reset)
    add_chunks(collection, chunks, embedder)
    return collection


# --- CLI ------------------------------------------------------------------------
def explain() -> None:
    """Worked example: embeddings, cosine by hand, and Chroma's distance for the same pair."""
    emb = Embedder()
    a = "The data controller must notify the Data Commissioner when personal data is breached."
    b = "Organisations have to tell the regulator if people's personal information is leaked."
    c = "Elephants migrate across the Serengeti plains every year."
    va, vb, vc = emb.encode([a, b, c])

    print(f"model            : {emb.model_name}")
    print(f"dimension        : {emb.dimension} numbers per text")
    print(f"max input        : {emb.max_tokens} word-pieces (longer text is truncated)")
    print(f"\nA = {a!r}\nB = {b!r}\nC = {c!r}")
    print(f"\nfirst 6 numbers of A : {[round(float(x), 4) for x in va[:6]]}  ... ({len(va)} total)")
    print(f"length |A|           : {np.linalg.norm(va):.4f}  (normalised to 1)")

    def cosine(x, y):
        return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))

    print("\ncosine(A, B) = (A . B) / (|A| |B|)")
    print(f"             = {float(np.dot(va, vb)):.4f} / ({np.linalg.norm(va):.2f} x {np.linalg.norm(vb):.2f})"
          f" = {cosine(va, vb):.4f}   <- same meaning, almost no shared words")
    print(f"cosine(A, C) = {cosine(va, vc):.4f}   <- unrelated topic")

    # Prove Chroma's distance = 1 - cosine: store B and C, query with A.
    col = chromadb.EphemeralClient().create_collection(
        "explain", configuration={"hnsw": {"space": "cosine"}}, embedding_function=None
    )
    col.add(ids=["B", "C"], embeddings=[vb.tolist(), vc.tolist()], documents=[b, c])
    res = col.query(query_embeddings=[va.tolist()], n_results=2)
    print("\nChroma query with A:")
    for id_, dist in zip(res["ids"][0], res["distances"][0]):
        print(f"  {id_}: distance={dist:.4f}  -> similarity = 1 - distance = {1 - dist:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunks and store them in ChromaDB.")
    parser.add_argument("--explain", action="store_true", help="worked embedding / cosine example")
    parser.add_argument("--build", action="store_true", help="(re)build the persistent index")
    parser.add_argument("--strategy", default=config.CHUNK_STRATEGY, choices=["fixed", "structure"])
    parser.add_argument("--size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument("--max-size", type=int, default=config.STRUCTURE_MAX_SIZE)
    args = parser.parse_args()

    if args.explain:
        explain()
    if args.build:
        docs = load_corpus()
        chunks = chunk_corpus(docs, args.strategy, args.size, args.overlap, args.max_size)
        t0 = time.time()
        emb = Embedder()
        col = build_index(chunks, emb)
        lens = emb.token_lengths([c.text for c in chunks])
        over = sum(1 for n in lens if n > emb.max_tokens)
        print(f"\nstored {col.count()} chunks in {config.CHROMA_DIR} ({time.time() - t0:.1f}s)")
        print(f"strategy={args.strategy}  size={args.size}  overlap={args.overlap}  "
              f"max_size={args.max_size}")
        print(f"chunks longer than the model's {emb.max_tokens}-token limit (truncated when "
              f"embedded): {over}/{len(chunks)}")
    if not (args.explain or args.build):
        parser.print_help()


if __name__ == "__main__":
    main()
