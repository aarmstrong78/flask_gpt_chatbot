import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Set required environment variable before importing app
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import app


# Patch functions that would hit external services
@pytest.fixture(autouse=True)
def _patch_external(monkeypatch):
    monkeypatch.setattr(app, "get_gpt_response", lambda *args, **kwargs: "mocked")
    monkeypatch.setattr(app, "get_gpt_response_stream", lambda *args, **kwargs: iter(["mocked"]))
    # do not rebuild FAISS during tests
    monkeypatch.setattr(app, "rebuild_faiss", lambda *args, **kwargs: None)


@pytest.fixture(scope="module", autouse=True)
def _add_error_route():
    @app.app.route("/trigger_error")
    def trigger_error():
        raise Exception("boom")


def create_client():
    app.app.config["TESTING"] = False
    return app.app.test_client()


def test_404_handler():
    client = create_client()
    response = client.get("/nonexistent")
    assert response.status_code == 404
    assert b"404 - Page Not Found" in response.data


def test_internal_error_handler():
    client = create_client()
    response = client.get("/trigger_error")
    assert response.status_code == 500
    assert b"500 - Internal Server Error" in response.data


def test_get_response_missing_message():
    client = create_client()
    response = client.post("/get_response", json={})
    assert response.status_code == 400
    assert response.get_json()["error"] == "No input provided"


def test_stream_response_missing_message():
    client = create_client()
    response = client.post("/stream_response", json={})
    assert response.status_code == 400
    assert response.get_json()["error"] == "No input provided"


def test_remove_document_missing_id():
    client = create_client()
    response = client.post("/remove_document", json={})
    assert response.status_code == 400
    assert response.get_json()["error"] == "No document_id provided"


def test_remove_document_not_found(monkeypatch):
    def fake_remove(doc_id):
        raise ValueError("Document ID not found.")

    monkeypatch.setattr(app, "remove_document", fake_remove)
    client = create_client()
    response = client.post("/remove_document", json={"document_id": "abc"})
    assert response.status_code == 404
    assert response.get_json()["error"] == "Document ID not found."
