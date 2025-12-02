"""
Convenience exports for the personal assistant service.

The FastAPI app and other scripts can import from this module to access the
refactored helpers without dealing with deep package paths.
"""

from backend.config import AppConfig
from backend.document_processing import process_pdf
from backend.embeddings import build_semantic_chunker, chunk_pages, init_embedding_model
from backend.helpers import collection_name_from_document
from backend.pipelines import ingest_document, run_qa
from backend.retrieval import LimitedSemanticCache, build_hybrid_retriever, build_qa_chain
from backend.vectorstores import (
    get_existing_qdrant_vectorstore,
    init_qdrant_client,
    init_qdrant_vectorstore,
    init_sparse_embeddings,
)

__all__ = [
    "AppConfig",
    "process_pdf",
    "init_embedding_model",
    "build_semantic_chunker",
    "chunk_pages",
    "init_qdrant_client",
    "init_sparse_embeddings",
    "init_qdrant_vectorstore",
    "get_existing_qdrant_vectorstore",
    "LimitedSemanticCache",
    "build_hybrid_retriever",
    "build_qa_chain",
    "collection_name_from_document",
    "ingest_document",
    "run_qa",
]
