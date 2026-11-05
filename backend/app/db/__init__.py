"""PostgreSQL persistence (SQLAlchemy): document metadata, chat history and evaluation results.

Vectors and chunk text stay in ChromaDB (`app.services.vector_store`); nothing here
is used for retrieval.
"""
