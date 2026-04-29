import uuid
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient


def _get_app():
    # Re-import to avoid module-level graph singleton issues in tests
    import importlib
    import src.api.main as m

    importlib.reload(m)
    return m.app


def test_assess_returns_report_id():
    report_id = str(uuid.uuid4())
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "final_report": "Risk is high.",
            "report_id": report_id,
            "low_confidence": False,
        }
    )

    with patch("src.api.main.build_graph", return_value=mock_graph):
        from src.api.main import app

        client = TestClient(app)
        resp = client.post("/assess", json={"region": "Harris County, TX"})

    assert resp.status_code == 200
    assert resp.json()["report_id"] == report_id


def test_assess_requires_auth_when_api_key_set(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    from src.config import get_settings

    get_settings.cache_clear()

    from src.api.main import app

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/assess", json={"region": "Test"}, headers={})
    assert resp.status_code == 401
    get_settings.cache_clear()


def test_get_report_not_found():
    rid = str(uuid.uuid4())
    mock_session = AsyncMock()
    mock_session.get = AsyncMock(return_value=None)

    with patch("src.api.main.get_async_session") as mock_ctx:
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

        from src.api.main import app

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get(f"/report/{rid}")

    assert resp.status_code == 404


def test_health_endpoint():
    from src.api.main import app

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
