"""FastAPI service endpoints, ASGI application instance, and Pydantic schemas."""

from __future__ import annotations

from typing import Any, List, Optional

from src.api.schemas import BatchPredictionResponse, HealthResponse, PredictionResponse

try:
    from src.api.app import app, create_app
except ImportError:
    app: Optional[Any] = None  # type: ignore[no-redef]
    create_app: Optional[Any] = None  # type: ignore[no-redef]

__all__: list[str] = [
    "app",
    "create_app",
    "HealthResponse",
    "PredictionResponse",
    "BatchPredictionResponse",
]

