# Evaluated Document QA Bot — backend

FastAPI service for PDF question answering with RAG (Groq, ChromaDB, hybrid search, optional Cohere reranking) and Ragas evaluation.

## Storage

| Store | Holds |
|---|---|
| ChromaDB (`chroma_db/`) | Chunk text, embeddings and chunk metadata. Used for retrieval. |
| PostgreSQL | Document metadata and processing status, chat conversations with messages and citations, and evaluation results. Never used for retrieval. |

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt
copy .env.example .env           # then fill in GROQ_API_KEY and the DB_* values
```

The API creates the database (if it does not exist) and any missing tables when it
starts, in the background. To do this without starting the API, for example to see
the result or set up a new server, run:

```bash
python -m app.db.cli init-db
```

Set `DB_INIT_ON_STARTUP=false` to turn off the startup step. If PostgreSQL is down at
startup, the API logs a warning and starts anyway. Restart it, or run `init-db`, once
PostgreSQL is up.

If you used the app before PostgreSQL was added, add rows for the documents already
in ChromaDB, and load the saved evaluation reports:

```bash
python -m app.db.cli sync-documents
python -m app.db.cli import-reports
```

## Run

```bash
uvicorn app.main:app --reload --port 8030
```

API docs: http://localhost:8030/docs

The API starts even when PostgreSQL is down. Asking questions keeps working, but answers
are not saved (`history_saved: false`). Uploads and the history and evaluation endpoints
return 503.

## Main endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/documents/upload` | Upload and index a PDF; its status is recorded as `processing`, then `indexed` or `failed` |
| GET | `/api/documents` | All documents with status, newest change first |
| GET / DELETE | `/api/documents/{id}` | Document details / remove its chunks and record |
| POST | `/api/ask` | Answer a question with citations; saved to `conversation_id` or a new conversation |
| GET | `/api/conversations` | Saved conversations (`limit`, `offset`) |
| GET / PATCH / DELETE | `/api/conversations/{id}` | Messages and citations / rename / delete |
| GET | `/api/evaluation/reports` | Stored evaluation runs |
| GET / DELETE | `/api/evaluation/reports/{id}` | One run's metrics per retrieval configuration / remove it |
| POST | `/api/evaluation/reports/import` | Store report files from `evaluation/reports/` that are not in the database |

## Evaluation

```bash
python -m evaluation.evaluate_rag --configs dense,hybrid,hybrid+rerank
```

Writes JSON, CSV and Markdown reports to `evaluation/reports/` and saves the results in PostgreSQL.

## Changing the database schema

`init-db` creates missing tables and never changes existing ones. After editing
`app/db/models.py`:

- **New table:** run `python -m app.db.cli init-db` again.
- **Changed column in an existing table:** apply the change yourself, e.g.
  `ALTER TABLE documents ADD COLUMN language VARCHAR(20);`. Alternatively, drop the
  table and run `init-db` again, which loses that table's data.

## Tests

```bash
pytest                    # unit and API tests (in-memory SQLite, no PostgreSQL needed)
pytest -m integration     # uses the real embedding model
pytest -m postgres        # init-db against TEST_DATABASE_URL (all tables in it are dropped)
```
