from __future__ import annotations

import time
from typing import Dict


def run_qa(query: str, qa_chain) -> Dict:
    start = time.perf_counter()
    answer = qa_chain.invoke(query)
    duration = time.perf_counter() - start
    return {"query": query, "answer": answer, "latency_seconds": duration}


__all__ = ["run_qa"]
