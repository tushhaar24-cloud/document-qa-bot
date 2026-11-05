"""Database initialization when the API starts (init_database itself is faked here;
test_init_db.py runs it against real PostgreSQL)."""

import dataclasses
import logging

import pytest
from fastapi.testclient import TestClient

from app import main
from app.db.init_db import InitResult
from app.db.session import DatabaseError


@pytest.fixture
def startup(monkeypatch: pytest.MonkeyPatch):
    """Configure startup; never load the real embedding model or touch the real database."""

    def configure(db_init_on_startup: bool = True, init=None):
        calls = []

        def fake_init() -> InitResult:
            calls.append(True)
            if init is not None:
                return init()
            return InitResult(database_created=False, created_tables=["documents"], existing_tables=[])

        monkeypatch.setattr(main, "init_database", fake_init)
        monkeypatch.setattr(
            main, "settings",
            dataclasses.replace(main.settings, embedding_preload=False, db_init_on_startup=db_init_on_startup),
        )
        return calls

    return configure


def test_startup_creates_the_tables(startup, caplog: pytest.LogCaptureFixture) -> None:
    calls = startup()
    caplog.set_level(logging.INFO, logger="app.main")

    with TestClient(main.app) as client:  # "with" runs the startup and shutdown hooks
        assert client.get("/health").status_code == 200

    assert calls == [True]
    assert "Created database tables: documents" in caplog.text


def test_startup_continues_when_the_database_is_down(startup, caplog: pytest.LogCaptureFixture) -> None:
    def unavailable() -> InitResult:
        raise DatabaseError("The database is unavailable.", 503)

    startup(init=unavailable)

    with TestClient(main.app) as client:
        assert client.get("/health").json() == {"status": "ok"}

    assert "Database initialization skipped: The database is unavailable." in caplog.text


def test_startup_initialization_can_be_turned_off(startup) -> None:
    calls = startup(db_init_on_startup=False)

    with TestClient(main.app) as client:
        client.get("/health")

    assert calls == []
