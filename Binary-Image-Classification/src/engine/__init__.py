"""Training loop execution, validation metrics evaluation, and test-time augmentation inference."""

from __future__ import annotations

from src.engine.inference import generate_predictions, predict_tta
from src.engine.trainer import _val, train, validate

__all__: list[str] = [
    "train",
    "validate",
    "_val",
    "predict_tta",
    "generate_predictions",
]

