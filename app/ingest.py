from app.config import get_settings
from app.document_loader import DocumentChunk, chunk_pages, load_document_pages
from app.retrieval import Retriever


def ingest_documents(reset: bool = True) -> tuple[int, int]:
    settings = get_settings()
    pages = load_document_pages(settings.docs_dir)
    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)

    retriever = Retriever(settings)
    if reset:
        try:
            retriever.client.delete_collection(settings.collection_name)
        except ValueError:
            pass
        retriever = Retriever(settings)

    if not chunks:
        return len(pages), 0

    for start in range(0, len(chunks), 100):
        batch = chunks[start : start + 100]
        retriever.collection.upsert(
            ids=[chunk.id for chunk in batch],
            documents=[chunk.text for chunk in batch],
            metadatas=[_metadata(chunk) for chunk in batch],
        )
    return len(pages), len(chunks)


def _metadata(chunk: DocumentChunk) -> dict[str, str | int]:
    metadata: dict[str, str | int] = {"source": chunk.source, "chunk": chunk.chunk}
    if chunk.page is not None:
        metadata["page"] = chunk.page
    return metadata


if __name__ == "__main__":
    documents, chunks = ingest_documents(reset=True)
    print(f"Indexed {documents} document pages into {chunks} chunks")
