from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
from uuid import uuid4

from langchain_qdrant import QdrantVectorStore

from backend.config import PDFProcessingConfig, VectorStoreConfig
from backend.document_processing.pdf_pipeline import process_pdf
from backend.embeddings.factory import chunk_pages
from backend.vectorstores.qdrant_store import init_qdrant_vectorstore


def ingest_document(
    file_path: Path,
    collection_name: str,
    embedding_model,
    chunker,
    qdrant_client,
    sparse_embeddings,
    pdf_config: PDFProcessingConfig,
    vector_config: VectorStoreConfig,
) -> Tuple[Dict, QdrantVectorStore]:
    pdf_output = process_pdf(
        str(file_path),
        dpi=pdf_config.dpi,
        max_pages=pdf_config.max_pages,
        max_workers=pdf_config.max_workers,
        fast_mode=pdf_config.fast_mode,
    )
    documents = chunk_pages(pdf_output["pages"], chunker, str(file_path))
    vectorstore = init_qdrant_vectorstore(
        client=qdrant_client,
        collection_name=collection_name,
        embeddings=embedding_model,
        sparse_embeddings=sparse_embeddings,
        vector_size=vector_config.vector_size,
        distance_metric=vector_config.distance_metric,
    )
    ids = vectorstore.add_documents(documents, ids=[str(uuid4()) for _ in documents])
    result = {
        "collection_name": collection_name,
        "file": str(file_path),
        "processing_summary": pdf_output["summary"],
        "pages": len(pdf_output["pages"]),
        "chunks_ingested": len(ids),
    }
    return result, vectorstore


__all__ = ["ingest_document"]
