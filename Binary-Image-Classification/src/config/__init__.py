"""Configuration dataclasses, default hyperparameters, and system execution constants."""

from __future__ import annotations

from src.config.config import (
    ALLOWED_EXTENSIONS,
    BATCH_SIZE,
    CKPT,
    DEVICE,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    INPUT_SIZE,
    LR,
    NUM_WORKERS,
    RANDOM_SEED,
    VAL_PER_CLASS,
    WEIGHT_DECAY,
    Config,
)

__all__: list[str] = [
    "Config",
    "ALLOWED_EXTENSIONS",
    "INPUT_SIZE",
    "VAL_PER_CLASS",
    "BATCH_SIZE",
    "EPOCHS",
    "EARLY_STOP_PATIENCE",
    "LR",
    "WEIGHT_DECAY",
    "NUM_WORKERS",
    "RANDOM_SEED",
    "CKPT",
    "DEVICE",
]

