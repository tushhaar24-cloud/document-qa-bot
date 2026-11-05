"""Engine and session management.

The engine is created on first use, so the API still starts (and the RAG
endpoints still work) while PostgreSQL is down; only database work fails, with a 503.

Two ways to get a session:
- `get_db`: a FastAPI dependency for plain `def` endpoints (FastAPI runs those in a thread).
  The endpoint commits; the dependency rolls back on errors and always closes the session.
- `session_scope()`: a context manager for code already running in a worker thread,
  e.g. a function passed to `asyncio.to_thread` from an `async` endpoint. It commits on
  success, rolls back on any exception, closes the session and raises `DatabaseError`.
SQLAlchemy errors that escape an endpoint are turned into 503/500 responses in `app.main`.
"""

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.exc import DBAPIError, OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None
_lock = threading.Lock()


class DatabaseError(Exception):
    """A database failure that is safe to show to API clients."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def get_engine() -> Engine:
    global _engine, _session_factory
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = create_engine(
                    get_settings().database_url,
                    pool_pre_ping=True,  # drop connections the server closed instead of failing a request
                    connect_args={"connect_timeout": 5},
                )
                # expire_on_commit=False: objects stay readable after the session closes.
                _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def set_engine(engine: Engine | None) -> None:
    """Use a different engine (tests use SQLite); None goes back to the configured one."""
    global _engine, _session_factory
    with _lock:
        if _engine is not None and _engine is not engine:
            _engine.dispose()
        _engine = engine
        _session_factory = sessionmaker(bind=engine, expire_on_commit=False) if engine is not None else None


def dispose_engine() -> None:
    global _engine, _session_factory
    with _lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None
        _session_factory = None


def to_database_error(exc: SQLAlchemyError) -> DatabaseError:
    """Map SQLAlchemy errors to safe messages; details only go to the log."""
    logger.error("Database error: %s: %s", type(exc).__name__, getattr(exc, "orig", exc))
    is_connection_problem = isinstance(exc, OperationalError) or (
        isinstance(exc, DBAPIError) and exc.connection_invalidated
    )
    if is_connection_problem:
        return DatabaseError("The database is unavailable.", 503)
    return DatabaseError("A database error occurred.", 500)


@contextmanager
def session_scope() -> Iterator[Session]:
    get_engine()
    assert _session_factory is not None
    session = _session_factory()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        raise to_database_error(exc) from exc
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency. Endpoints call `db.commit()` themselves before returning,
    so a failed commit becomes an error response instead of happening after a 200."""
    get_engine()
    assert _session_factory is not None
    session = _session_factory()
    try:
        yield session
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()
