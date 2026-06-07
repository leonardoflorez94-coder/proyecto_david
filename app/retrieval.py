from dataclasses import dataclass

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.config import Settings


@dataclass(frozen=True)
class SearchResult:
    text: str
    source: str
    page: int | None
    chunk: int
    score: float | None


class Retriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def search(self, query: str, limit: int | None = None) -> list[SearchResult]:
        if self.count() == 0:
            return []
        response = self.collection.query(
            query_texts=[query],
            n_results=limit or self.settings.retrieval_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results: list[SearchResult] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            results.append(
                SearchResult(
                    text=document,
                    source=str(metadata.get("source", "unknown")),
                    page=metadata.get("page"),
                    chunk=int(metadata.get("chunk", 0)),
                    score=None if distance is None else round(1 - float(distance), 4),
                )
            )
        return results

    def count(self) -> int:
        return self.collection.count()
