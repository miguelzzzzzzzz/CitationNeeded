"""HTTP service for Citation Needed (optional extra ``api``: FastAPI + uvicorn).

``rag_engine.api.service`` is framework-free; only ``rag_engine.api.app`` imports
FastAPI, so the rest of the package never needs the ``api`` extra.
"""

from rag_engine.api.service import (
    DependencyMissingError,
    GenerationError,
    IndexNotLoadedError,
    InvalidRequestError,
    RagService,
    ServedIndex,
)

__all__ = [
    "DependencyMissingError",
    "GenerationError",
    "IndexNotLoadedError",
    "InvalidRequestError",
    "RagService",
    "ServedIndex",
]
