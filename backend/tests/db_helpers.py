"""Helpers for tests that need the database to fail."""

import pytest
from sqlalchemy import create_engine

from app.db import session as db_session


def make_unreachable_engine():
    # Nothing listens on port 1, so every connection attempt fails at once.
    return create_engine(
        "postgresql+psycopg://user:secret@127.0.0.1:1/missing", connect_args={"connect_timeout": 1}
    )


@pytest.fixture
def database_down():
    """Point the app at a PostgreSQL server that is not running."""
    db_session.set_engine(make_unreachable_engine())
    yield
    # isolated_database (conftest) puts its own engine back for the next test.
