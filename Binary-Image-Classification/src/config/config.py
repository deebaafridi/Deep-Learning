"""Configuration specifications and default hyperparameters for DualCNN.

Defines the centralized `Config` dataclass and exposes module-level constants
preserving full backward compatibility with legacy scripts and functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Set, Union

try:
    import torch
    DEFAULT_DEVICE: Any = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    torch = None
    DEFAULT_DEVICE = "cuda"

# ==============================================================================
# Global Hyperparameter Constants (Backward Compatibility)
# ==============================================================================

ALLOWED_EXTENSIONS: Set[str] = {".png", ".jpg", ".jpeg", ".bmp"}
"""Allowed file extensions for image discovery."""

INPUT_SIZE: int = 96
"""Spatial resolution (height and width in pixels) to which images are resized."""

VAL_PER_CLASS: int = 500
"""Number of validation samples drawn per class (1,000 validation images total)."""

BATCH_SIZE: int = 96
"""Mini-batch size utilized across training, validation, and inference loaders."""

EPOCHS: int = 25
"""Maximum number of training epochs."""

EARLY_STOP_PATIENCE: int = 5
"""Number of consecutive epochs without validation accuracy improvement before stopping."""

LR: float = 3e-4
"""Initial learning rate for the AdamW optimizer."""

WEIGHT_DECAY: float = 1e-4
"""Weight decay regularization factor for the AdamW optimizer."""

NUM_WORKERS: int = 0
"""Number of subprocesses for PyTorch DataLoader workers (0 = main process)."""

RANDOM_SEED: int = 42
"""Deterministic random seed for reproducibility across random, numpy, and torch."""

CKPT: str = "ckpt_dual.pth"
"""Default filename/path for saving and loading the best model weights checkpoint."""

DEVICE: Any = DEFAULT_DEVICE
"""Hardware compute device selected dynamically (CUDA if available, else CPU)."""


@dataclass
class Config:
    """Production configuration dataclass for DualCNN pipeline.

    Encapsulates all hyperparameters, path specifications, hardware devices,
    and runtime settings required for model training and inference.

    Attributes:
        input_size: Target square dimensions (height/width) for image resizing.
        val_per_class: Number of samples partitioned per class for validation.
        batch_size: Mini-batch size for DataLoaders.
        epochs: Maximum training epochs.
        early_stop_patience: Epoch tolerance before early stopping triggers.
        learning_rate: Base learning rate for AdamW optimizer.
        weight_decay: L2 penalty factor for AdamW optimizer.
        num_workers: Data loading worker threads.
        random_seed: Seed for pseudo-random number generators.
        ckpt_path: Filesystem path to save or load PyTorch checkpoint weights.
        device: Hardware device target (torch.device or string identifier).
        allowed_extensions: Set of valid file extensions for image discovery.
        log_file: Path to destination file for rotating/appending execution logs.
        submission_filename: Output CSV filename for test predictions.
    """

    input_size: int = INPUT_SIZE
    val_per_class: int = VAL_PER_CLASS
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    early_stop_patience: int = EARLY_STOP_PATIENCE
    learning_rate: float = LR
    weight_decay: float = WEIGHT_DECAY
    num_workers: int = NUM_WORKERS
    random_seed: int = RANDOM_SEED
    ckpt_path: Union[Path, str] = CKPT
    device: Any = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch is not None
        else "cuda"
    )
    allowed_extensions: Set[str] = field(default_factory=lambda: set(ALLOWED_EXTENSIONS))
    log_file: str = "solution.log"
    submission_filename: str = "submission.csv"

    @property
    def ckpt(self) -> Path:
        """Returns the model checkpoint path normalized as a pathlib.Path object."""
        return Path(self.ckpt_path)
