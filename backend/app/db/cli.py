"""Database maintenance commands. Run from the backend folder:

    python -m app.db.cli init-db          # create the database (if missing) and its tables
                                          # (the API also does this when it starts)
    python -m app.db.cli sync-documents   # add rows for documents already in ChromaDB
    python -m app.db.cli import-reports   # store evaluation/reports/*.json in PostgreSQL

All are safe to run repeatedly.
"""

import argparse
import sys

from sqlalchemy.engine import make_url

from app.core.config import get_settings
from app.db.init_db import init_database
from app.db.session import DatabaseError, session_scope
from app.repositories import documents as document_repo
from app.services.evaluation_reports import REPORTS_DIR, import_report_files
from app.services.vector_store import VectorStoreError, list_documents


def init_db() -> int:
    result = init_database()
    if result.database_created:
        print(f"Created database {make_url(get_settings().database_url).database}.")
    created = result.created_tables
    print(f"Created {len(created)} tables" + (f": {', '.join(created)}" if created else "") + ".")
    if result.existing_tables:
        print(f"Already present (left unchanged): {', '.join(result.existing_tables)}.")
    return 0


def sync_documents() -> int:
    """Create (or repair) a database row for every document found in ChromaDB."""
    stored = list_documents()
    created = 0
    with session_scope() as db:
        for doc in stored:
            created += document_repo.upsert_indexed_from_vector_store(
                db, doc.document_id, doc.filename, doc.chunk_count, len(doc.page_numbers)
            )
    print(f"{len(stored)} documents in ChromaDB; {created} new database records created.")
    # Rows still "processing" for documents ChromaDB does not have are left alone: they
    # are either being uploaded right now or failed in a way that left no chunks.
    return 0


def import_reports(replace: bool) -> int:
    with session_scope() as db:
        result = import_report_files(db, REPORTS_DIR, replace=replace)
    print(f"Imported {len(result.imported)}, already stored {len(result.skipped)}, unreadable {len(result.failed)}.")
    for report_id in result.failed:
        print(f"  could not read {report_id}.json")
    return 1 if result.failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.db.cli", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-db", help="create the database (if missing) and its tables")
    commands.add_parser("sync-documents", help="add database rows for documents already in ChromaDB")
    imports = commands.add_parser("import-reports", help="store evaluation/reports/*.json in the database")
    imports.add_argument("--replace", action="store_true", help="re-import reports that are already stored")
    args = parser.parse_args(argv)
    try:
        if args.command == "init-db":
            return init_db()
        if args.command == "sync-documents":
            return sync_documents()
        return import_reports(args.replace)
    except (DatabaseError, VectorStoreError) as exc:
        print(f"Error: {exc.message}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
