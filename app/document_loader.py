import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass(frozen=True)
class DocumentPage:
    source: str
    text: str
    page: int | None = None


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    text: str
    source: str
    page: int | None
    chunk: int


def load_document_pages(docs_dir: Path) -> list[DocumentPage]:
    pages: list[DocumentPage] = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.suffix.lower() == ".pdf":
            pages.extend(_load_pdf(path, docs_dir))
        else:
            pages.append(DocumentPage(source=_relative_source(path, docs_dir), text=path.read_text(encoding="utf-8")))
    return [page for page in pages if normalize_text(page.text)]


def chunk_pages(pages: list[DocumentPage], chunk_size: int, chunk_overlap: int) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for page in pages:
        for index, text in enumerate(chunk_text(page.text, chunk_size, chunk_overlap)):
            raw_id = f"{page.source}:{page.page}:{index}:{text}"
            chunk_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    text=text,
                    source=page.source,
                    page=page.page,
                    chunk=index,
                )
            )
    return chunks


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    clean = normalize_text(text)
    if not clean:
        return []
    if len(clean) <= chunk_size:
        return [clean]

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)
    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        if end < len(clean):
            boundary = clean.rfind(". ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        chunks.append(clean[start:end].strip())
        start += step
    return [chunk for chunk in chunks if chunk]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _load_pdf(path: Path, docs_dir: Path) -> list[DocumentPage]:
    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    for index, page in enumerate(reader.pages, start=1):
        pages.append(
            DocumentPage(
                source=_relative_source(path, docs_dir),
                text=page.extract_text() or "",
                page=index,
            )
        )
    return pages


def _relative_source(path: Path, docs_dir: Path) -> str:
    return str(path.relative_to(docs_dir)).replace("\\", "/")
