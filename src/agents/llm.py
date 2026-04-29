import asyncio
import httpx
from src.config import get_settings


async def complete(prompt: str, system: str = "") -> str:
    settings = get_settings()
    provider = settings.llm_provider

    if provider == "auto":
        provider = "openrouter" if settings.openrouter_api_key else "ollama"

    if provider == "openrouter":
        return await _openrouter(prompt, system, settings)
    if provider == "ollama":
        return await _ollama(prompt, system, settings)
    raise ValueError(f"Unknown LLM provider: {provider!r}. Use 'openrouter', 'ollama', or 'auto'.")


async def _openrouter(prompt: str, system: str, settings) -> str:
    if not settings.openrouter_api_key:
        raise ValueError("LLM_PROVIDER=openrouter but OPENROUTER_API_KEY is not set")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(5):
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
        if resp.status_code == 429:
            wait = 2 ** attempt
            await asyncio.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def _ollama(prompt: str, system: str, settings) -> str:
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{settings.ollama_url}/api/generate",
            json={"model": settings.ollama_model, "prompt": full_prompt, "stream": False},
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()["response"]
