import httpx
from src.config import get_settings


async def complete(prompt: str, system: str = "") -> str:
    settings = get_settings()

    if settings.openrouter_api_key:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/zimpele/ZimRadar",
                },
                json={"model": settings.openrouter_model, "messages": messages},
                timeout=60.0,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    # Ollama fallback
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{settings.ollama_url}/api/generate",
            json={"model": settings.ollama_model, "prompt": full_prompt, "stream": False},
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()["response"]
