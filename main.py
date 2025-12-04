from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from backend.config import AppConfig, PDFProcessingConfig, VectorStoreConfig
from backend.embeddings import build_semantic_chunker, init_embedding_model
from backend.helpers import collection_name_from_document
from backend.pipelines import ingest_document, run_qa
from backend.retrieval import LimitedSemanticCache, build_hybrid_retriever, build_qa_chain
from backend.vectorstores import (
    get_existing_qdrant_vectorstore,
    init_qdrant_client,
    init_qdrant_vectorstore,
    init_sparse_embeddings,
    list_collections,
)

load_dotenv()

app = FastAPI(title="Personal Assist API", version="0.1.0")


class IngestRequest(BaseModel):
    file_path: str = Field(..., description="Absolute path to the PDF file to ingest.")
    collection_name: Optional[str] = Field(
        None, description="Override name for the Qdrant collection. Defaults to document-derived slug."
    )
    fast_mode: Optional[bool] = Field(None, description="Override fast mode for PDF processing.")
    max_pages: Optional[int] = Field(None, description="Limit number of pages to process (debugging).")


class IngestResponse(BaseModel):
    collection_name: str
    file: str
    pages: int
    chunks_ingested: int
    processing_summary: Dict[str, Any]


class QueryRequest(BaseModel):
    query: str
    collection_name: Optional[str] = Field(None, description="Collection to query. Defaults to last ingested.")
    top_k: int = Field(10, ge=1, le=50)
    use_cache: bool = Field(True, description="Toggle semantic cache usage.")
    search_all_collections: bool = Field(
        False, description="When true, query every available collection in parallel."
    )


class QueryResponse(BaseModel):
    query: str
    collection_name: str
    answer: str
    latency_seconds: float
    cache_hit: bool
    cache_stats: Dict[str, Any]


class CollectionsResponse(BaseModel):
    collections: List[str]


def _init_llm(config: AppConfig):
    provider = config.llm.provider
    if provider == "ollama":
        kwargs = {"model": config.llm.model_name, "temperature": config.llm.temperature}
        if config.llm.base_url:
            kwargs["base_url"] = config.llm.base_url
        return ChatOllama(**kwargs)
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required when using the Gemini provider.")
        return ChatGoogleGenerativeAI(
            model=config.llm.model_name,
            temperature=config.llm.temperature,
            google_api_key=api_key,
        )
    raise ValueError(f"Unsupported LLM provider: {config.llm.provider}")


def _pdf_config_with_overrides(base: PDFProcessingConfig, request: IngestRequest) -> PDFProcessingConfig:
    return PDFProcessingConfig(
        dpi=base.dpi,
        max_workers=base.max_workers,
        fast_mode=base.fast_mode if request.fast_mode is None else request.fast_mode,
        max_pages=base.max_pages if request.max_pages is None else request.max_pages,
    )


def _vector_config(base: VectorStoreConfig) -> VectorStoreConfig:
    return VectorStoreConfig(
        url=base.url,
        api_key=base.api_key,
        collection_name=base.collection_name,
        vector_size=base.vector_size,
        distance_metric=base.distance_metric,
        sparse_model_name=base.sparse_model_name,
    )


@app.on_event("startup")
def startup_event():
    settings = AppConfig.from_env()
    if not settings.open_mode:
        settings.llm.provider = "gemini"
        settings.llm.model_name = os.getenv("GEMINI_LLM_MODEL", settings.llm.model_name or "gemini-2.5-flash")
        settings.embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    app.state.config = settings
    embedding_provider = "huggingface" if settings.open_mode else "gemini"
    app.state.embedding_model = init_embedding_model(settings.embedding_model, settings.embedding_device, embedding_provider)
    app.state.semantic_chunker = build_semantic_chunker(
        app.state.embedding_model,
        breakpoint_threshold_type=settings.chunking.breakpoint_threshold_type,
        breakpoint_threshold_amount=settings.chunking.breakpoint_threshold_amount,
    )
    app.state.qdrant_client = init_qdrant_client(settings.vectorstore.url, settings.vectorstore.api_key)
    app.state.sparse_embeddings = init_sparse_embeddings(settings.vectorstore.sparse_model_name)
    app.state.llm = _init_llm(settings)
    app.state.cache = LimitedSemanticCache(
        embeddings=app.state.embedding_model,
        max_size=settings.cache.max_size,
        similarity_threshold=settings.cache.similarity_threshold,
    )
    app.state.vectorstores: Dict[str, Any] = {}
    app.state.retrievers: Dict[str, Any] = {}
    app.state.last_collection: Optional[str] = None
    app.state.open_mode = settings.open_mode
    try:
        app.state.collections = list_collections(app.state.qdrant_client)
    except Exception:
        app.state.collections = []


def _resolve_collection_name(requested: Optional[str], file_path: Path) -> str:
    prefix = app.state.config.default_collection_prefix
    return requested or collection_name_from_document(file_path.name, prefix=prefix)


def _get_vectorstore(collection_name: str):
    if collection_name in app.state.vectorstores:
        return app.state.vectorstores[collection_name]
    vectorstore = get_existing_qdrant_vectorstore(
        client=app.state.qdrant_client,
        collection_name=collection_name,
        embeddings=app.state.embedding_model,
        sparse_embeddings=app.state.sparse_embeddings,
    )
    app.state.vectorstores[collection_name] = vectorstore
    return vectorstore


def _get_retriever(collection_name: str, top_k: int):
    cache_key = (collection_name, top_k)
    if cache_key in app.state.retrievers:
        return app.state.retrievers[cache_key]
    vectorstore = _get_vectorstore(collection_name)
    retriever = build_hybrid_retriever(vectorstore, k=max(top_k, 20))
    app.state.retrievers[cache_key] = retriever
    return retriever


def _build_multi_collection_retriever(collections: List[str], top_k: int):
    per_collection_k = max(top_k, 5)

    class MultiCollectionRetriever:
        def invoke(self, query: str):
            docs = []
            max_workers = max(1, min(len(collections), os.cpu_count() or 4))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self._retrieve_for_collection, collection, query)
                    for collection in collections
                ]
                for future in as_completed(futures):
                    docs.extend(future.result())
            return docs[:top_k]

        def _retrieve_for_collection(self, collection_name: str, query: str):
            retriever = _get_retriever(collection_name, per_collection_k)
            results = retriever.invoke(query)[:per_collection_k]
            for doc in results:
                doc.metadata.setdefault("collection", collection_name)
            return results

    return MultiCollectionRetriever()


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(request: IngestRequest):
    file_path = Path(request.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    if not file_path.is_file():
        raise HTTPException(status_code=400, detail="Provided path is not a file.")

    collection_name = _resolve_collection_name(request.collection_name, file_path)
    existing_collections = await run_in_threadpool(list_collections, app.state.qdrant_client)
    if collection_name in existing_collections:
        raise HTTPException(
            status_code=409,
            detail=f"Collection '{collection_name}' already exists. Delete it or choose a different name before re-uploading.",
        )
    pdf_config = _pdf_config_with_overrides(app.state.config.pdf, request)
    vector_config = _vector_config(app.state.config.vectorstore)

    result, vectorstore = await run_in_threadpool(
        ingest_document,
        file_path,
        collection_name,
        app.state.embedding_model,
        app.state.semantic_chunker,
        app.state.qdrant_client,
        app.state.sparse_embeddings,
        pdf_config,
        vector_config,
    )

    app.state.vectorstores[collection_name] = vectorstore
    app.state.last_collection = collection_name
    app.state.collections = sorted(set(existing_collections + [collection_name]))
    # invalidate retrievers using this collection
    app.state.retrievers = {k: v for k, v in app.state.retrievers.items() if k[0] != collection_name}

    return IngestResponse(**result)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    if request.search_all_collections:
        collections = await run_in_threadpool(list_collections, app.state.qdrant_client)
        collections = sorted(collections)
        if not collections:
            raise HTTPException(status_code=400, detail="No collections available to search.")
        retriever = _build_multi_collection_retriever(collections, request.top_k)
        collection_label = "all"
    else:
        collection_name = request.collection_name or app.state.last_collection
        if not collection_name:
            raise HTTPException(status_code=400, detail="No collection available. Ingest a document first.")
        retriever = _get_retriever(collection_name, request.top_k)
        collection_label = collection_name

    cache = app.state.cache if request.use_cache else None
    qa_chain = build_qa_chain(retriever=retriever, llm=app.state.llm, cache=cache, top_k=request.top_k)
    qa_result = await run_in_threadpool(run_qa, request.query, qa_chain)
    cache_hit = cache.last_hit() if cache else False

    return QueryResponse(
        query=qa_result["query"],
        collection_name=collection_label,
        answer=qa_result["answer"],
        latency_seconds=qa_result["latency_seconds"],
        cache_hit=cache_hit,
        cache_stats=cache.stats() if cache else {},
    )


@app.get("/collections", response_model=CollectionsResponse)
async def collections_endpoint():
    collections = await run_in_threadpool(list_collections, app.state.qdrant_client)
    app.state.collections = collections
    return CollectionsResponse(collections=collections)


@app.get("/health")
def health_check():
    return {"status": "ok", "collection": app.state.last_collection, "total_collections": len(app.state.collections)}
