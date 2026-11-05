"""Basic API tests. /api/ask answering is covered in test_ask_api.py."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_returns_welcome_message() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]


def test_health_returns_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 2001},
        {},
        {"question": None},
    ],
)
def test_ask_rejects_invalid_questions(payload: dict) -> None:
    response = client.post("/api/ask", json=payload)
    assert response.status_code == 422
