import types


def test_authorize_logs_in_with_id_token(monkeypatch):
    # Import app and test client
    import app as app_module

    client = app_module.app.test_client()

    # Monkeypatch oauth methods to bypass real Google calls
    monkeypatch.setattr(
        app_module.oauth.google,
        "authorize_access_token",
        lambda: {"access_token": "dummy", "id_token": "dummy"},
    )
    monkeypatch.setattr(
        app_module.oauth.google,
        "parse_id_token",
        lambda token: {"sub": "user-123", "name": "Test User", "email": "t@example.com"},
    )

    # Hit the callback directly
    resp = client.get("/authorize", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/chat")

    # Now protected route should be accessible
    r2 = client.get("/chat")
    assert r2.status_code == 200


def test_authorize_fallbacks_to_userinfo(monkeypatch):
    import app as app_module

    client = app_module.app.test_client()

    # authorize_access_token returns a token, but parse_id_token raises
    monkeypatch.setattr(
        app_module.oauth.google,
        "authorize_access_token",
        lambda: {"access_token": "dummy"},
    )

    def raise_parse(_):
        raise RuntimeError("bad id token")

    monkeypatch.setattr(app_module.oauth.google, "parse_id_token", raise_parse)

    # Simulate userinfo endpoint response
    class DummyResp:
        def json(self):
            return {"sub": "user-456", "name": "Alt User", "email": "alt@example.com"}

    monkeypatch.setattr(app_module.oauth.google, "get", lambda path: DummyResp())

    resp = client.get("/authorize", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/chat")

    r2 = client.get("/chat")
    assert r2.status_code == 200

