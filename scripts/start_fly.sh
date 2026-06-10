#!/usr/bin/env sh
set -eu

mkdir -p "$STORAGE_DIR" "$DOCS_DIR" "${OLLAMA_MODELS:-/data/ollama}"

ollama serve &
OLLAMA_PID=$!

echo "Waiting for Ollama..."
for i in $(seq 1 60); do
  if curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

echo "Pulling model $OLLAMA_MODEL..."
ollama pull "$OLLAMA_MODEL"

if [ "${AUTO_INGEST:-false}" = "true" ]; then
  echo "Indexing documents from $DOCS_DIR..."
  python -m app.ingest || echo "Document ingestion failed; API will still start."
fi

echo "Starting API..."
python -m uvicorn app.main:app --host "${APP_HOST:-0.0.0.0}" --port "${APP_PORT:-8000}"

kill "$OLLAMA_PID" 2>/dev/null || true
