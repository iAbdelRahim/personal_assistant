from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache(maxsize=4)
def init_embedding_model(
    model_name: str = "google/embeddinggemma-300m",
    device: str = "cpu",
    provider: str = "huggingface",
) -> Embeddings:
    """Initialize and memoize the embedding model for the requested provider."""
    provider = provider.lower()
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required when OPEN_MODE is false or provider is 'gemini'.")
        return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device, "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_semantic_chunker(
    embedding_model: Embeddings,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: int = 95,
) -> SemanticChunker:
    return SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
    )


def chunk_pages(
    pages: Sequence[dict],
    chunker: SemanticChunker,
    source_path: str,
) -> List[Document]:
    documents: List[Document] = []
    for page in pages:
        if not page.get("final_text"):
            continue
        metadata = {
            "pageno": page.get("pageno"),
            "source": source_path,
            "extraction_method": page.get("extraction_method"),
        }
        page_doc = [Document(page_content=page["final_text"], metadata=metadata)]
        documents.extend(chunker.split_documents(page_doc))
    return documents


__all__ = ["init_embedding_model", "build_semantic_chunker", "chunk_pages"]
