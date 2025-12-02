from __future__ import annotations

from typing import Optional

from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import SparseVectorParams, VectorParams


def init_qdrant_client(url: str, api_key: Optional[str] = None) -> QdrantClient:
    return QdrantClient(url=url, api_key=api_key)


def init_sparse_embeddings(model_name: str = "Qdrant/bm25") -> FastEmbedSparse:
    return FastEmbedSparse(model_name=model_name)


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    distance_metric: str = "COSINE",
) -> None:
    distance = getattr(models.Distance, distance_metric.upper(), models.Distance.COSINE)
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"dense": VectorParams(size=vector_size, distance=distance, on_disk=True)},
            sparse_vectors_config={"sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))},
            quantization_config={"scalar": {"type": "int8", "quantile": 0.99}},
        )
    except UnexpectedResponse as exc:
        if "already exists" not in str(exc):
            raise


def init_qdrant_vectorstore(
    client: QdrantClient,
    collection_name: str,
    embeddings,
    sparse_embeddings: FastEmbedSparse,
    vector_size: int = 768,
    distance_metric: str = "COSINE",
) -> QdrantVectorStore:
    ensure_collection(client, collection_name, vector_size, distance_metric)
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )


def get_existing_qdrant_vectorstore(
    client: QdrantClient,
    collection_name: str,
    embeddings,
    sparse_embeddings: FastEmbedSparse,
) -> QdrantVectorStore:
    return QdrantVectorStore.from_existing_collection(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )


def list_collections(client: QdrantClient) -> list[str]:
    response = client.get_collections()
    if not response.collections:
        return []
    return [collection.name for collection in response.collections]


__all__ = [
    "init_qdrant_client",
    "init_sparse_embeddings",
    "init_qdrant_vectorstore",
    "get_existing_qdrant_vectorstore",
    "list_collections",
]
