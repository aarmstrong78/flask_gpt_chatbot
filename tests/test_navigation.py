import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import app


@pytest.fixture(autouse=True)
def _patch_external(monkeypatch):
    monkeypatch.setattr(app, "get_gpt_response", lambda *args, **kwargs: "mocked")
    monkeypatch.setattr(app, "get_gpt_response_stream", lambda *args, **kwargs: iter(["mocked"]))
    monkeypatch.setattr(app, "rebuild_faiss", lambda *args, **kwargs: None)


def create_client():
    app.app.config["TESTING"] = False
    return app.app.test_client()


def test_upload_page_has_chat_link():
    client = create_client()
    response = client.get("/upload_page")
    assert response.status_code == 200
    assert b'href="/chat"' in response.data
