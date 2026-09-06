"""Deterministic random seed utility for reproducible experiment execution.

Sets pseudo-random seeds across standard Python, NumPy, and PyTorch (including CUDA),
enforcing deterministic algorithmic execution where supported.
"""

from __future__ import annotations

import os
import random
import numpy as np

try:
    import torch
except ImportError:
    torch = None


def seed_everything(seed: int = 42) -> None:
    """Enforces strict deterministic seeding across all random number generators.

    Configures:
    1. Python built-in `random` module.
    2. Python hash seed via `PYTHONHASHSEED` environment variable.
    3. `numpy.random` global state.
    4. `torch.manual_seed` and CUDA GPU seeds if PyTorch is present.
    5. cuDNN deterministic backend flags (disabling non-deterministic auto-tuner algorithms).

    Args:
        seed: Integer seed value to apply (default is 42).

    Example:
        >>> seed_everything(42)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic convolution algorithms in cuDNN
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
