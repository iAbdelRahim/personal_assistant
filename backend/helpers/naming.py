from __future__ import annotations

import re
from pathlib import Path


def collection_name_from_document(document_path: str, prefix: str = "doc") -> str:
    stem = Path(document_path).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    slug = slug or "collection"
    combined = f"{prefix}-{slug}"
    return combined[:63]


__all__ = ["collection_name_from_document"]
