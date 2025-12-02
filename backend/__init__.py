"""
Core package for the personal assistant service.

The package exposes document-processing utilities, embeddings helpers,
vector-store management, and retrieval/QA pipelines that power the FastAPI app.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("backend")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0"

__all__ = ["__version__"]
