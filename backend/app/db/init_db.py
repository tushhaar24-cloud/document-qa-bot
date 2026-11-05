"""Create the PostgreSQL database and tables from the models in app/db/models.py.

Used by `python -m app.db.cli init-db` and, unless DB_INIT_ON_STARTUP=false, when the
API starts. Only missing objects are created: existing tables and their data are
never changed or dropped. After changing a column in models.py, apply that change to
existing tables yourself (ALTER TABLE), or drop the table and run this again (its
data is lost).
"""

from dataclasses import dataclass

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import get_settings
from app.db import models  # noqa: F401  (registers the tables on Base.metadata)
from app.db.base import Base
from app.db.session import get_engine, to_database_error


@dataclass(frozen=True)
class InitResult:
    database_created: bool
    created_tables: list[str]
    existing_tables: list[str]


def create_database_if_missing(url: URL) -> bool:
    """Create the PostgreSQL database named in `url`; returns True when it was created."""
    if url.get_backend_name() != "postgresql" or not url.database:
        return False
    # CREATE DATABASE cannot run inside a transaction, and needs a connection to another database.
    server = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT", connect_args={"connect_timeout": 5})
    try:
        with server.connect() as connection:
            exists = connection.scalar(text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": url.database})
            if exists:
                return False
            quoted = server.dialect.identifier_preparer.quote(url.database)
            connection.execute(text(f"CREATE DATABASE {quoted}"))
            return True
    finally:
        server.dispose()


def init_database() -> InitResult:
    """Create the database if needed, then every table that does not exist yet.

    Raises DatabaseError (503 when the server cannot be reached).
    """
    try:
        database_created = create_database_if_missing(make_url(get_settings().database_url))
        engine = get_engine()
        existing = set(inspect(engine).get_table_names())
        # checkfirst (the default) skips tables that already exist.
        Base.metadata.create_all(engine)
    except SQLAlchemyError as exc:
        raise to_database_error(exc) from exc
    return InitResult(
        database_created=database_created,
        created_tables=[name for name in Base.metadata.tables if name not in existing],
        existing_tables=[name for name in Base.metadata.tables if name in existing],
    )
