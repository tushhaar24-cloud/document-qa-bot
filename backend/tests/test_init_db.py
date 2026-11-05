"""Database and table creation (init-db command and API startup) against a real PostgreSQL server.

Run with:  pytest -m postgres
Needs TEST_DATABASE_URL (in the environment or backend/.env) pointing at a disposable
database, e.g. postgresql+psycopg://postgres:...@localhost:5432/document_qa_bot_test.
Every table in that database is dropped.
"""

import dataclasses
import os
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url

from app.db import cli, init_db
from app.db import session as db_session
from app.db.base import Base

pytestmark = pytest.mark.postgres

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "")


@pytest.fixture
def postgres(monkeypatch: pytest.MonkeyPatch):
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL is not set")
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)
    cli.get_settings.cache_clear()
    engine = create_engine(TEST_DATABASE_URL)
    Base.metadata.drop_all(engine)
    db_session.set_engine(None)  # init_db builds its engine from DATABASE_URL
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()
    db_session.dispose_engine()
    cli.get_settings.cache_clear()


def test_init_db_creates_every_table(postgres, capsys) -> None:
    assert cli.main(["init-db"]) == 0

    assert set(Base.metadata.tables) <= set(inspect(postgres).get_table_names())
    assert f"Created {len(Base.metadata.tables)} tables" in capsys.readouterr().out


def test_init_db_is_safe_to_run_again_and_keeps_data(postgres, capsys) -> None:
    cli.main(["init-db"])
    with postgres.begin() as connection:
        connection.execute(text(
            "INSERT INTO documents (id, filename, status, created_at, updated_at) "
            "VALUES ('abc', 'a.pdf', 'indexed', now(), now())"
        ))

    assert cli.main(["init-db"]) == 0

    assert "Created 0 tables" in capsys.readouterr().out
    with postgres.connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM documents")) == 1


def test_postgres_enforces_constraints_and_cascades(postgres) -> None:
    cli.main(["init-db"])

    with postgres.begin() as connection:
        connection.execute(text(
            "INSERT INTO conversations (id, title, created_at, updated_at) "
            "VALUES ('11111111-1111-1111-1111-111111111111', 't', now(), now())"
        ))
        connection.execute(text(
            "INSERT INTO chat_messages (id, conversation_id, position, role, content, created_at) "
            "VALUES ('22222222-2222-2222-2222-222222222222', '11111111-1111-1111-1111-111111111111', 1, 'user', 'q', now())"
        ))
        connection.execute(text("DELETE FROM conversations"))
        assert connection.scalar(text("SELECT count(*) FROM chat_messages")) == 0

    with pytest.raises(Exception, match="ck_documents_status"):
        with postgres.begin() as connection:
            connection.execute(text(
                "INSERT INTO documents (id, filename, status, created_at, updated_at) "
                "VALUES ('x', 'a.pdf', 'bogus', now(), now())"
            ))


def test_create_database_if_missing(postgres) -> None:
    url = make_url(TEST_DATABASE_URL).set(database="document_qa_bot_init_check")
    server = create_engine(make_url(TEST_DATABASE_URL).set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with server.connect() as connection:
            connection.execute(text("DROP DATABASE IF EXISTS document_qa_bot_init_check"))

        assert init_db.create_database_if_missing(url) is True
        assert init_db.create_database_if_missing(url) is False  # already there
    finally:
        with server.connect() as connection:
            connection.execute(text("DROP DATABASE IF EXISTS document_qa_bot_init_check"))
        server.dispose()


def test_api_startup_creates_missing_tables(postgres, monkeypatch: pytest.MonkeyPatch) -> None:
    from app import main

    # Startup reads the module-level settings; skip loading the real embedding model.
    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, embedding_preload=False, db_init_on_startup=True))
    assert inspect(postgres).get_table_names() == []

    with TestClient(main.app) as client:  # "with" runs the startup and shutdown hooks
        for _ in range(100):
            if set(Base.metadata.tables) <= set(inspect(postgres).get_table_names()):
                break
            time.sleep(0.1)
        assert client.get("/api/conversations").json() == {"total": 0, "conversations": []}

    assert set(Base.metadata.tables) <= set(inspect(postgres).get_table_names())
