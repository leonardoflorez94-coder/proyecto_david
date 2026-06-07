import httpx

from app.config import Settings


class OllamaClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def generate(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.settings.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
            },
        }
        try:
            with httpx.Client(timeout=90) as client:
                response = client.post(f"{self.settings.ollama_base_url}/api/chat", json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                "No se pudo contactar a Ollama. Verificá que esté corriendo y que el modelo exista."
            ) from exc
        data = response.json()
        return data.get("message", {}).get("content", "").strip()
