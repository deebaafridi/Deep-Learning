"""Pydantic request and response validation schemas for the DualCNN FastAPI service.

Defines schemas for:
1. `HealthResponse`: System status, device, and model initialization state.
2. `PredictionResponse`: Single-image inference results including dual probabilities.
3. `BatchPredictionResponse`: Aggregate inference results across multiple uploaded images.
"""

from __future__ import annotations

from typing import List

try:
    from pydantic import BaseModel, Field
except ImportError:
    # Graceful fallback for environments where pydantic is not yet installed
    class BaseModel:  # type: ignore
        """Minimal fallback class replicating basic attribute assignment."""
        def __init__(self, **kwargs) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def Field(*args, **kwargs):  # type: ignore
        return None


class HealthResponse(BaseModel):
    """Health check response schema.

    Attributes:
        status: Service operational health string ("healthy").
        model_loaded: Boolean indicating if weights were loaded from checkpoint.
        device: Compute hardware identifier ("cuda" or "cpu").
        version: API semantic version string.
    """

    status: str = "healthy"
    model_loaded: bool = False
    device: str = "cpu"
    version: str = "1.0.0"


class PredictionResponse(BaseModel):
    """Single-image inference prediction response schema.

    Attributes:
        filename: Name of the evaluated image file.
        cube_probability: Estimated probability of cube presence in range [0, 1].
        sphere_probability: Estimated probability of sphere presence in range [0, 1].
        joint_probability: Combined composite probability P(cube * sphere).
        prediction: Binary classification label (1 = positive, 0 = negative).
        class_name: Human-readable textual class description.
        inference_time_ms: Wall-clock latency for preprocessing and forward inference in milliseconds.
    """

    filename: str
    cube_probability: float
    sphere_probability: float
    joint_probability: float
    prediction: int
    class_name: str
    inference_time_ms: float


class BatchPredictionResponse(BaseModel):
    """Batch prediction response schema.

    Attributes:
        total_images: Total number of evaluated images.
        positive_count: Number of images classified as Class 1.
        negative_count: Number of images classified as Class 0.
        predictions: List of individual `PredictionResponse` objects.
    """

    total_images: int
    positive_count: int
    negative_count: int
    predictions: List[PredictionResponse]
