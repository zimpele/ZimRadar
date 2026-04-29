from src.config import Settings


def test_openrouter_fields_have_defaults():
    s = Settings()
    assert s.openrouter_api_key == ""
    assert s.openrouter_model == "meta-llama/llama-3.3-70b-instruct:free"


def test_openrouter_fields_read_from_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemma-3-27b-it:free")
    s = Settings()
    assert s.openrouter_api_key == "sk-test-key"
    assert s.openrouter_model == "google/gemma-3-27b-it:free"
