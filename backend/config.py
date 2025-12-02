from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PDFProcessingConfig:
    dpi: int = 150
    max_workers: int = 4
    fast_mode: bool = True
    max_pages: Optional[int] = None


@dataclass
class ChunkingConfig:
    breakpoint_threshold_type: str = "percentile"
    breakpoint_threshold_amount: int = 95


@dataclass
class VectorStoreConfig:
    url: str = "http://localhost:6333"
    api_key: Optional[str] = None
    collection_name: str = "documents"
    vector_size: int = 768
    distance_metric: str = "COSINE"
    sparse_model_name: str = "Qdrant/bm25"


@dataclass
class CacheConfig:
    max_size: int = 100
    similarity_threshold: float = 0.80


@dataclass
class LLMConfig:
    provider: str = "ollama"
    model_name: str = "llama3.2"
    temperature: float = 0.1
    base_url: Optional[str] = None


@dataclass
class AppConfig:
    pdf: PDFProcessingConfig = field(default_factory=PDFProcessingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding_model: str = "google/embeddinggemma-300m"
    embedding_device: str = "cpu"
    default_collection_prefix: str = "doc"

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables with sane defaults."""
        pdf_cfg = PDFProcessingConfig(
            dpi=int(os.getenv("PDF_DPI", 150)),
            max_workers=int(os.getenv("PDF_MAX_WORKERS", 4)),
            fast_mode=os.getenv("PDF_FAST_MODE", "true").lower() == "true",
            max_pages=int(os.getenv("PDF_MAX_PAGES")) if os.getenv("PDF_MAX_PAGES") else None,
        )
        chunk_cfg = ChunkingConfig(
            breakpoint_threshold_type=os.getenv("CHUNK_THRESHOLD_TYPE", "percentile"),
            breakpoint_threshold_amount=int(os.getenv("CHUNK_THRESHOLD_AMOUNT", 95)),
        )
        vector_cfg = VectorStoreConfig(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY") or None,
            collection_name=os.getenv("QDRANT_COLLECTION", "documents"),
            vector_size=int(os.getenv("QDRANT_VECTOR_SIZE", 768)),
            distance_metric=os.getenv("QDRANT_DISTANCE", "COSINE").upper(),
            sparse_model_name=os.getenv("QDRANT_SPARSE_MODEL", "Qdrant/bm25"),
        )
        cache_cfg = CacheConfig(
            max_size=int(os.getenv("CACHE_MAX_SIZE", 100)),
            similarity_threshold=float(os.getenv("CACHE_SIMILARITY", 0.80)),
        )
        llm_cfg = LLMConfig(
            provider=os.getenv("LLM_PROVIDER", "ollama").lower(),
            model_name=os.getenv("LLM_MODEL", "llama3.2"),
            temperature=float(os.getenv("LLM_TEMPERATURE", 0.1)),
            base_url=os.getenv("LLM_BASE_URL") or None,
        )

        return cls(
            pdf=pdf_cfg,
            chunking=chunk_cfg,
            vectorstore=vector_cfg,
            cache=cache_cfg,
            llm=llm_cfg,
            embedding_model=os.getenv("EMBEDDING_MODEL", "google/embeddinggemma-300m"),
            embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
            default_collection_prefix=os.getenv("DEFAULT_COLLECTION_PREFIX", "doc"),
        )


__all__ = [
    "AppConfig",
    "PDFProcessingConfig",
    "ChunkingConfig",
    "VectorStoreConfig",
    "CacheConfig",
    "LLMConfig",
]
