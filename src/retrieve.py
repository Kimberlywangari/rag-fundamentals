"""Retrieval: query -> top-k most similar chunks, each with score, rank and provenance.

How ranking works
    1. The query is embedded with the *same* model that embedded the chunks (a different model
       would put the query in a different space and distances would be meaningless).
    2. ChromaDB finds the nearest stored vectors by cosine distance.  For a few hundred chunks
       the HNSW index is effectively exact; at large scale it is approximate nearest neighbour.
    3. Results are sorted by distance ascending, i.e. similarity (1 - distance) descending.
       Rank 1 = most similar.  The score is cosine similarity, roughly in [0, 1] for text.

What k trades off
    Small k  -> precise, cheap, short prompt, but if the answer sits in rank k+1 it is lost
                (a recall failure the generator can never repair).
    Large k  -> more chance the right chunk is present, but more irrelevant text: a longer,
                costlier prompt and more distractors the LLM may latch onto.
    k is a fixed number here, so a question needing three scattered provisions can still be
    under-served and an easy one still pays for k chunks.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from src import config
from src.embed_store import Embedder, SupportsEncode, get_client, open_collection


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int  # 1 = best match
    score: float  # cosine similarity = 1 - Chroma distance
    distance: float  # raw Chroma cosine distance
    text: str
    source: str
    page_start: int
    page_end: int
    section: str
    chunk_id: str

    @property
    def pages(self) -> str:
        if self.page_start == self.page_end:
            return str(self.page_start)
        return f"{self.page_start}-{self.page_end}"


class Retriever:
    def __init__(self, embedder: SupportsEncode, collection):
        self.embedder = embedder
        self.collection = collection

    def retrieve(self, query: str, k: int = config.TOP_K, source: str | None = None) -> list[RetrievedChunk]:
        """Return up to ``k`` chunks ranked by cosine similarity to ``query``.

        ``source`` optionally restricts the search to one PDF (metadata filter).
        """
        if not query.strip():
            raise ValueError("query must not be empty")
        if k <= 0:
            raise ValueError("k must be positive")
        n = self.collection.count()
        if n == 0:
            return []
        vector = self.embedder.encode([query])[0]
        res = self.collection.query(
            query_embeddings=[vector.tolist()],
            n_results=min(k, n),
            where={"source": source} if source else None,
            include=["documents", "metadatas", "distances"],
        )
        out: list[RetrievedChunk] = []
        for rank, (cid, doc, meta, dist) in enumerate(
            zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]),
            start=1,
        ):
            out.append(
                RetrievedChunk(
                    rank=rank,
                    score=round(1.0 - float(dist), 4),
                    distance=round(float(dist), 4),
                    text=doc,
                    source=meta["source"],
                    page_start=int(meta["page_start"]),
                    page_end=int(meta["page_end"]),
                    section=meta.get("section", ""),
                    chunk_id=cid,
                )
            )
        return out


def load_retriever(embedder: SupportsEncode | None = None, name: str = config.COLLECTION_NAME) -> Retriever:
    """Open the persistent index built by ``python -m src.embed_store --build``."""
    return Retriever(embedder or Embedder(), open_collection(get_client(), name))


def format_results(results: list[RetrievedChunk], snippet: int = 260) -> str:
    lines = []
    for r in results:
        where = f"{r.source}  p.{r.pages}" + (f"  [{r.section}]" if r.section else "")
        text = " ".join(r.text.split())
        lines.append(f"#{r.rank}  score={r.score:.4f}  {where}")
        lines.append(f"    {text[:snippet]}{'...' if len(text) > snippet else ''}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve the top-k chunks for a query.")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=config.TOP_K)
    parser.add_argument("--source", default=None, help="restrict to one PDF filename")
    args = parser.parse_args()
    retriever = load_retriever()
    print(format_results(retriever.retrieve(args.query, args.k, args.source)))


if __name__ == "__main__":
    main()
