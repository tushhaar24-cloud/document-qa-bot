"""CORS settings the React frontend relies on.

Document listing, chat history and evaluation report APIs are tested in
test_documents_db_api.py, test_conversations_api.py and test_evaluation_api.py.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.mark.parametrize("method", ["GET", "POST", "PATCH", "DELETE"])
def test_cors_allows_the_frontend_dev_server(method: str) -> None:
    response = client.options(
        "/api/conversations",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": method},
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_cors_rejects_unknown_origins() -> None:
    response = client.get("/health", headers={"Origin": "http://evil.example"})

    assert "access-control-allow-origin" not in response.headers
