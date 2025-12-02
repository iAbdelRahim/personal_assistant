from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np


class LimitedSemanticCache:
    """Simple LRU-like semantic cache keyed by query embeddings."""

    def __init__(self, embeddings, max_size: int = 100, similarity_threshold: float = 0.80):
        self.embeddings = embeddings
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.last_lookup_hit = False

    def _similarity(self, query_vec, cached_vec) -> float:
        return float(
            np.dot(query_vec, cached_vec)
            / (np.linalg.norm(query_vec) * np.linalg.norm(cached_vec) + 1e-10)
        )

    def get(self, query: str) -> Optional[List[Any]]:
        query_embedding = self.embeddings.embed_query(query)
        for cached_query, cached_data in self.cache.items():
            if self._similarity(query_embedding, cached_data["embedding"]) >= self.similarity_threshold:
                self.hits += 1
                self.last_lookup_hit = True
                # Move to end to mark as recently used
                self.cache.move_to_end(cached_query)
                return cached_data["results"]
        self.misses += 1
        self.last_lookup_hit = False
        return None

    def put(self, query: str, results: List[Any]) -> None:
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        embedding = self.embeddings.embed_query(query)
        self.cache[query] = {"embedding": embedding, "results": results}

    def stats(self) -> Dict[str, float]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total else 0.0
        return {"hits": self.hits, "misses": self.misses, "hit_rate": hit_rate}

    def last_hit(self) -> bool:
        return self.last_lookup_hit


__all__ = ["LimitedSemanticCache"]
