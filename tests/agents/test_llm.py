import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.agents.llm import complete


@pytest.mark.asyncio
async def test_complete_uses_openrouter_when_key_set(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    monkeypatch.setenv("LLM_PROVIDER", "auto")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"choices": [{"message": {"content": "test response"}}]}

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        from src.config import get_settings

        get_settings.cache_clear()

        result = await complete("hello", system="be helpful")
        assert result == "test response"
        call_kwargs = mock_client.post.call_args
        assert "openrouter.ai" in call_kwargs[0][0]


@pytest.mark.asyncio
async def test_complete_falls_back_to_ollama_when_no_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER", "auto")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"response": "ollama reply"}

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        from src.config import get_settings

        get_settings.cache_clear()

        result = await complete("hello")
        assert result == "ollama reply"
        call_kwargs = mock_client.post.call_args
        assert "ollama" in call_kwargs[0][0] or "11434" in call_kwargs[0][0]


@pytest.mark.asyncio
async def test_complete_explicit_ollama_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-ignored")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"response": "local model reply"}

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        from src.config import get_settings

        get_settings.cache_clear()

        result = await complete("hello")
        assert result == "local model reply"
        call_kwargs = mock_client.post.call_args
        assert "ollama" in call_kwargs[0][0] or "11434" in call_kwargs[0][0]


@pytest.mark.asyncio
async def test_complete_explicit_openrouter_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-real")
    monkeypatch.setenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"choices": [{"message": {"content": "openrouter reply"}}]}

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        from src.config import get_settings

        get_settings.cache_clear()

        result = await complete("hello")
        assert result == "openrouter reply"
        call_kwargs = mock_client.post.call_args
        assert "openrouter.ai" in call_kwargs[0][0]
