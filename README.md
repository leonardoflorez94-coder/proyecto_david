# MeLi IAM Bot

Bot conversacional para responder preguntas sobre IAM usando documentos como fuente de verdad.

La solución implementa una API en Python con FastAPI, RAG sobre documentos locales, memoria conversacional propia en SQLite y un LLM open source servido con Ollama.

## Arquitectura

- `FastAPI`: expone `/chat`, `/ingest`, `/health` y el historial de conversaciones.
- `Ollama`: ejecuta un modelo open source, por defecto `llama3.2:3b`.
- `ChromaDB`: guarda embeddings de los documentos para recuperación semántica.
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`: embeddings multilingues para documentos en español/inglés.
- `SQLite`: memoria conversacional implementada por la API, sin memoria de LangChain ni del LLM.
- Frontend simple tipo chat disponible en `/`.

## Requisitos

- Python 3.11+
- Ollama instalado: https://ollama.com
- Documentos IAM descargados desde el Drive del challenge

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Descargar el modelo open source:

```bash
ollama pull llama3.2:3b
```

Copiar los documentos del challenge en:

```text
data/documents/
```

Formatos soportados: PDF, TXT y MD.

## Ingesta De Documentos

Ejecutar:

```bash
python -m app.ingest
```

O con la API levantada:

```bash
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d "{\"reset\": true}"
```

La ingesta divide los documentos en chunks, genera embeddings y los persiste en `storage/chroma`.

## Ejecutar La API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Abrir el chat web:

```text
http://localhost:8000
```

Consultar la API:

```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"message\": \"Que significa AAA?\"}"
```

Respuesta esperada:

```json
{
  "conversation_id": "...",
  "answer": "...",
  "sources": [
    {
      "source": "documento.pdf",
      "page": 3,
      "chunk": 0,
      "score": 0.72
    }
  ]
}
```

Para continuar una conversación, reenviar el mismo `conversation_id`.

## Docker

Primero levantar Ollama en el host y descargar el modelo:

```bash
ollama pull llama3.2:3b
```

Luego:

```bash
docker compose up --build
```

El contenedor usa `host.docker.internal:11434` para llamar a Ollama desde Docker.

## Memoria

La memoria está implementada en `app/memory.py` con SQLite.

La API guarda mensajes `user` y `assistant` por `conversation_id` y recupera los últimos `MEMORY_MAX_MESSAGES` para dar continuidad. El prompt aclara que el historial se usa solo para continuidad y que las respuestas deben basarse en los documentos recuperados.

## Decisiones De Diseño

- Se usa RAG para mantener respuestas actualizadas respecto de los documentos sin depender de conocimiento preentrenado del modelo.
- El prompt fuerza al bot a rechazar preguntas sin contexto documental suficiente.
- La memoria no depende del LLM ni de librerías con memoria integrada.
- Se devuelve la lista de fuentes usadas para facilitar auditoría de respuestas.
- Se incluye un script de fine-tuning LoRA opcional para cubrir el requerimiento de entrenamiento cuando haya hardware/tiempo disponible.

## Fine-Tuning LoRA Opcional

El runtime recomendado es RAG, pero se incluye un entrenamiento LoRA liviano sobre chunks documentales.

```bash
pip install -r training/requirements-train.txt
python training/train_lora.py
```

El adapter queda en:

```text
training/output/iam-lora
```

Base model usada: `Qwen/Qwen2.5-0.5B-Instruct`. Se eligió por ser open source y más viable para una prueba local que modelos más grandes.

## Pruebas Conversacionales Sugeridas

Después de indexar los documentos del challenge, probar:

- `Que tipos de metodos de autenticacion existen?`
- `Por que no es recomendable tener token de sesion con fecha de expiracion grande?`
- `Que significa AAA?`
- `De lo anterior, dame un resumen en 3 bullets`
- `Cual es la capital de Francia?`

La última pregunta debería ser rechazada o respondida indicando que no hay información en los documentos.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Variables De Entorno

- `OLLAMA_BASE_URL`: URL de Ollama. Default: `http://localhost:11434`.
- `OLLAMA_MODEL`: modelo open source. Default: `llama3.2:3b`.
- `DOCS_DIR`: carpeta de documentos. Default: `data/documents`.
- `STORAGE_DIR`: carpeta de ChromaDB y SQLite. Default: `storage`.
- `MEMORY_MAX_MESSAGES`: cantidad de mensajes recientes usados como memoria. Default: `8`.
- `RETRIEVAL_K`: cantidad de chunks recuperados por pregunta. Default: `5`.
