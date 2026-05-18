from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_ask_requires_document_id():
    response = client.post("/ask", json={"question": "What is this about?"})
    assert response.status_code == 422


def test_ask_rejects_empty_document_id():
    response = client.post(
        "/ask",
        json={"question": "What is this about?", "document_id": ""},
    )
    assert response.status_code == 422

