"""DualCNN Backward-Compatible Facade.

This module preserves full backward-compatibility with existing scripts,
automated grading runners, and legacy evaluation workflows that invoke or
import directly from `DualCNN.py`.

Under the hood, all core convolutional architectures, CoordConv coordinate layers,
Sobel edge filtering, composite dual loss functions, trainer loops, and test-time
augmentation (TTA) inference routines are modularly implemented within the `src/`
package. This file re-exports the identical public interface, dataclasses,
functions, and global constants without introducing breaking changes.

Usage:
    # Standard evaluation invocation (reads directory path from standard input):
    $ python DualCNN.py
    /path/to/dataset

    # Programmatic invocation from Python:
    >>> from DualCNN import generate_predictions
    >>> generate_predictions("./data")
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

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
try:
    from src.data.dataset import ImageRecord, RGBDataset, TestPathDataset
    from src.data.split import build_split, data_split, list_images
    from src.data.transforms import _hsv_jitter, augment_train, load_image
    from src.engine.inference import generate_predictions, predict_tta
    from src.engine.trainer import _val, train, validate
    from src.losses.loss import DualLoss, dual_loss
    from src.models.dual_head_cnn import DualHeadCNN
    from src.models.layers import AddCoords, ConvBlock
except ImportError:
    generate_predictions = None  # type: ignore[assignment]

from src.utils.logger import get_logger
from src.utils.seed import seed_everything

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch is not None:
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

logger: logging.Logger = get_logger("solution", log_file="solution.log")
fmt: logging.Formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

ch: Optional[logging.Handler] = next(
    (h for h in logger.handlers if isinstance(h, logging.StreamHandler)), None
)
fh: Optional[logging.Handler] = next(
    (h for h in logger.handlers if isinstance(h, logging.FileHandler)), None
)

__all__: list[str] = [
    "ALLOWED_EXTENSIONS",
    "INPUT_SIZE",
    "VAL_PER_CLASS",
    "BATCH_SIZE",
    "EPOCHS",
    "EARLY_STOP_PATIENCE",
    "LR",
    "WEIGHT_DEVIATION" if "WEIGHT_DEVIATION" in globals() else "WEIGHT_DECAY",
    "WEIGHT_DECAY",
    "NUM_WORKERS",
    "RANDOM_SEED",
    "CKPT",
    "DEVICE",
    "logger",
    "_hsv_jitter",
    "augment_train",
    "ImageRecord",
    "list_images",
    "build_split",
    "load_image",
    "RGBDataset",
    "TestPathDataset",
    "AddCoords",
    "ConvBlock",
    "DualHeadCNN",
    "dual_loss",
    "_val",
    "train",
    "predict_tta",
    "data_split",
    "generate_predictions",
    "Config",
    "DualCNNException",
    "DataNotFoundError",
    "ImageProcessingError",
    "ModelCheckpointError",
]

if __name__ == "__main__":
    try:
        data_dir: str = input().strip()
    except (EOFError, KeyboardInterrupt):
        data_dir = "."

    if generate_predictions is None:
        logger.error(
            "Deep learning runtime dependencies (PyTorch, Torchvision, Scipy) are not installed. "
            "Please install dependencies using: pip install -r requirements.txt"
        )
        sys.exit(1)

    generate_predictions(data_dir)
