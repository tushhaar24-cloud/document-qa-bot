"""FastAPI application entry point.

Run from the backend folder with either:
    python -m app.main
    uvicorn app.main:app --reload --port 8030
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.api.conversations import router as conversations_router
from app.api.documents import router as documents_router
from app.api.evaluation import router as evaluation_router
from app.api.questions import router as questions_router
from app.api.search import router as search_router
from app.core.config import get_settings
from app.db.init_db import init_database
from app.db.session import DatabaseError, dispose_engine, to_database_error
from app.services.embeddings import EmbeddingError, get_embedding_model
from app.services.generation import close_groq_client
from app.services.reranking import close_cohere_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


async def preload_embedding_model() -> None:
    try:
        await asyncio.to_thread(get_embedding_model)
    except EmbeddingError:
        # Already logged; uploads will retry loading and return 503 if it still fails.
        logger.warning("Embedding model preload failed")


async def initialize_database() -> None:
    try:
        result = await asyncio.to_thread(init_database)
    except DatabaseError as exc:
        # The API still serves questions; database features return errors until it is reachable.
        # Restart the API (or run: python -m app.db.cli init-db) once PostgreSQL is up.
        logger.warning("Database initialization skipped: %s", exc.message)
        return
    if result.database_created:
        logger.info("Created the database")
    if result.created_tables:
        logger.info("Created database tables: %s", ", ".join(result.created_tables))
    else:
        logger.info("Database tables are present")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Slow startup work runs in the background so the server starts immediately.
    preload_task = asyncio.create_task(preload_embedding_model()) if settings.embedding_preload else None
    db_task = asyncio.create_task(initialize_database()) if settings.db_init_on_startup else None
    yield
    for task in (preload_task, db_task):
        if task is not None and not task.done():
            task.cancel()
    await close_groq_client()
    await close_cohere_client()
    dispose_engine()


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.include_router(questions_router)
app.include_router(documents_router)
app.include_router(search_router)
app.include_router(evaluation_router)
app.include_router(conversations_router)

# Lets the React frontend (a different origin) call the API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(DatabaseError)
async def database_error_handler(request: Request, exc: DatabaseError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_error_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    # Never send SQL or driver messages to clients; to_database_error logs them.
    error = to_database_error(exc)
    return JSONResponse(status_code=error.status_code, content={"detail": error.message})


@app.get("/", tags=["meta"])
async def root() -> dict[str, str]:
    return {"message": f"Welcome to the {settings.app_name} API. See /docs for usage."}


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=settings.port, reload=True)
