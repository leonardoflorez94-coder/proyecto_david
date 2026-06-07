from app.memory import StoredMessage
from app.retrieval import SearchResult


SYSTEM_PROMPT = """
Sos un asistente de IAM para Mercado Libre.
Respondé únicamente con información respaldada por los documentos provistos en el contexto.
Usá el historial solo para mantener continuidad conversacional, no como fuente de verdad.
Si el contexto no contiene información suficiente, decí claramente que no encontraste esa información en los documentos.
Sé preciso, actualizado respecto del documento y evitá inventar datos.
Incluí una explicación breve y accionable cuando corresponda.
""".strip()


def build_messages(question: str, context: list[SearchResult], history: list[StoredMessage], max_context_chars: int) -> list[dict[str, str]]:
    formatted_context = format_context(context, max_context_chars)
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for message in history:
        messages.append({"role": message.role, "content": message.content})
    messages.append(
        {
            "role": "user",
            "content": (
                "Contexto documental disponible:\n"
                f"{formatted_context}\n\n"
                "Pregunta del usuario:\n"
                f"{question}"
            ),
        }
    )
    return messages


def format_context(results: list[SearchResult], max_chars: int) -> str:
    blocks: list[str] = []
    used = 0
    for result in results:
        page = f", pagina {result.page}" if result.page is not None else ""
        header = f"[Fuente: {result.source}{page}, chunk {result.chunk}]"
        block = f"{header}\n{result.text}"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks) if blocks else "Sin contexto relevante."
