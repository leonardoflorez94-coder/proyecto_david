from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    docs_dir: Path = Path("data/documents")
    storage_dir: Path = Path("storage")
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2:3b"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    memory_max_messages: int = 8
    retrieval_k: int = 5
    chunk_size: int = 900
    chunk_overlap: int = 150
    max_context_chars: int = 6500
    collection_name: str = "iam_documents"

    @property
    def chroma_dir(self) -> Path:
        return self.storage_dir / "chroma"

    @property
    def sqlite_path(self) -> Path:
        return self.storage_dir / "memory.sqlite3"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return settings
