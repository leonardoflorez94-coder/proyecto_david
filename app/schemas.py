from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: str | None = None


class Source(BaseModel):
    source: str
    page: int | None = None
    chunk: int
    score: float | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: list[Source]


class IngestRequest(BaseModel):
    reset: bool = True


class IngestResponse(BaseModel):
    documents_indexed: int
    chunks_indexed: int


class Message(BaseModel):
    role: str
    content: str
    created_at: str


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
    model: str
