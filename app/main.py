from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.ingest import ingest_documents
from app.llm import OllamaClient
from app.memory import MemoryStore
from app.prompt import build_messages
from app.retrieval import Retriever
from app.schemas import ChatRequest, ChatResponse, HealthResponse, IngestRequest, IngestResponse, Message, Source

settings = get_settings()
memory = MemoryStore(settings.sqlite_path)
retriever = Retriever(settings)
llm = OllamaClient(settings)

app = FastAPI(title="MeLi IAM Bot", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    memory.init()


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse("app/static/index.html")


app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", indexed_chunks=retriever.count(), model=settings.ollama_model)


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    documents, chunks = ingest_documents(reset=request.reset)
    global retriever
    retriever = Retriever(settings)
    return IngestResponse(documents_indexed=documents, chunks_indexed=chunks)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    conversation_id = request.conversation_id or str(uuid4())
    results = retriever.search(request.message)

    if not results:
        answer = "No encontré información relevante en los documentos indexados para responder esa pregunta."
        memory.add_message(conversation_id, "user", request.message)
        memory.add_message(conversation_id, "assistant", answer)
        return ChatResponse(conversation_id=conversation_id, answer=answer, sources=[])

    history = memory.recent_messages(conversation_id, settings.memory_max_messages)
    messages = build_messages(request.message, results, history, settings.max_context_chars)
    try:
        answer = llm.generate(messages)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    memory.add_message(conversation_id, "user", request.message)
    memory.add_message(conversation_id, "assistant", answer)
    return ChatResponse(
        conversation_id=conversation_id,
        answer=answer,
        sources=[
            Source(source=result.source, page=result.page, chunk=result.chunk, score=result.score)
            for result in results
        ],
    )


@app.get("/conversations/{conversation_id}/messages", response_model=list[Message])
def conversation_messages(conversation_id: str) -> list[Message]:
    return [Message(role=item.role, content=item.content, created_at=item.created_at) for item in memory.all_messages(conversation_id)]
