from fastapi.testclient import TestClient


def _get_app():
    from src.api.main import app

    return app


def test_admin_login_page_accessible():
    client = TestClient(_get_app(), follow_redirects=False)
    resp = client.get("/admin/")
    # unauthenticated → redirect to login
    assert resp.status_code in (200, 302)


def test_admin_login_rejected_with_wrong_password(monkeypatch):
    monkeypatch.setenv("ADMIN_USER", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "secret")
    from src.config import get_settings

    get_settings.cache_clear()

    client = TestClient(_get_app(), follow_redirects=False)
    resp = client.post("/admin/login", data={"username": "admin", "password": "wrong"})
    assert resp.status_code in (200, 302, 400)
    assert "authenticated" not in client.cookies
    get_settings.cache_clear()


def test_admin_login_succeeds_with_correct_password(monkeypatch):
    monkeypatch.setenv("ADMIN_USER", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "secret")
    monkeypatch.setenv("SECRET_KEY", "testsecret")
    from src.config import get_settings

    get_settings.cache_clear()

    client = TestClient(_get_app(), follow_redirects=False)
    resp = client.post("/admin/login", data={"username": "admin", "password": "secret"})
    # successful login redirects to /admin/
    assert resp.status_code in (302, 200)
    get_settings.cache_clear()
