"""Dataset splitting and directory resolution utilities for DualCNN.

Contains:
1. `list_images`: Discovers and sorts image paths matching allowed extensions.
2. `build_split`: Constructs stratified train and validation splits with deterministic seeding.
3. `data_split`: Dynamically resolves flat or nested Kaggle dataset split directories.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import numpy as np

from src.config.config import ALLOWED_EXTENSIONS, RANDOM_SEED, VAL_PER_CLASS
from src.data.dataset import ImageRecord
from src.exceptions.exceptions import DataNotFoundError


def list_images(
    root: Union[str, Path],
    allowed_extensions: Optional[Set[str]] = None,
) -> List[Path]:
    """Recursively traverses a directory and returns a sorted list of matching image paths.

    Args:
        root: Root directory to search recursively.
        allowed_extensions: Set of lowercased file extensions to permit (e.g. {'.png', '.jpg'}).
            If None, defaults to `ALLOWED_EXTENSIONS`.

    Returns:
        A sorted list of `pathlib.Path` objects pointing to discovered images.

    Raises:
        DataNotFoundError: If the provided `root` path does not exist on disk.
    """
    root_path: Path = Path(root)
    if not root_path.exists():
        raise DataNotFoundError(f"Image directory does not exist: {root_path}")

    exts: Set[str] = (
        allowed_extensions if allowed_extensions is not None else ALLOWED_EXTENSIONS
    )
    return sorted(
        p for p in root_path.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )


def build_split(
    train_dir: Union[str, Path],
    val_per_class: int = VAL_PER_CLASS,
    random_seed: int = RANDOM_SEED,
) -> Tuple[List[ImageRecord], List[ImageRecord]]:
    """Creates stratified, deterministic training and validation splits from class subdirectories.

    Iterates over expected binary classes (0 and 1), retrieves image paths from `train_dir/0`
    and `train_dir/1`, applies deterministic permutation, partitions the first `val_per_class`
    samples to validation and the remainder to training, and finally shuffles both collections.

    Args:
        train_dir: Base training directory containing '0' and '1' class subdirectories.
        val_per_class: Number of samples per class assigned to the validation set (default: 500).
        random_seed: Base seed for reproducible pseudo-random permutation (default: 42).

    Returns:
        A tuple `(train_records, val_records)` of ImageRecord lists.

    Raises:
        DataNotFoundError: If no images are found for either class directory.
    """
    train_dir_path: Path = Path(train_dir)
    train: List[ImageRecord] = []
    val: List[ImageRecord] = []

    for label in (0, 1):
        class_dir: Path = train_dir_path / str(label)
        files: List[Path] = list_images(class_dir)
        if len(files) == 0:
            raise DataNotFoundError(f"No image files found for class {label} in {class_dir}")

        # Deterministic stratified permutation
        rng: np.random.Generator = np.random.default_rng(random_seed + label)
        order: np.ndarray = rng.permutation(len(files))
        shuffled: List[Path] = [files[i] for i in order]

        # Partition into validation and training sets
        val.extend(ImageRecord(p, label) for p in shuffled[:val_per_class])
        train.extend(ImageRecord(p, label) for p in shuffled[val_per_class:])

    # Final shuffle across both classes
    global_rng: np.random.Generator = np.random.default_rng(random_seed)
    global_rng.shuffle(train)
    global_rng.shuffle(val)

    return train, val


def data_split(data_dir: Union[str, Path], split: str) -> Path:
    """Resolves flat or nested split directory paths.

    Handles differences in Kaggle zip extraction hierarchies, checking both:
    1. `data_dir / split / split` (nested archive structure)
    2. `data_dir / split` (flat standard structure)

    Args:
        data_dir: Root dataset folder path.
        split: Target split name (e.g. 'train' or 'test').

    Returns:
        Resolved `pathlib.Path` pointing to the directory.

    Raises:
        DataNotFoundError: If neither flat nor nested directory candidates exist.
    """
    base_dir: Path = Path(data_dir)
    for candidate in (base_dir / split / split, base_dir / split):
        if candidate.is_dir():
            return candidate
    raise DataNotFoundError(f"Could not find '{split}' directory under '{base_dir}'")
