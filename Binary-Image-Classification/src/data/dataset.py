"""PyTorch Dataset implementations and image sample abstractions for DualCNN.

Defines:
1. `ImageRecord`: Memory-efficient record containing sample path and integer target label.
2. `RGBDataset`: PyTorch Dataset yielding `(image_tensor, label_tensor)` pairs.
3. `TestPathDataset`: PyTorch Dataset yielding `(image_tensor, filename)` pairs for inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union

try:
    import torch
    from torch.utils.data import Dataset
    _BaseDataset = Dataset
except ImportError:
    torch = None
    _BaseDataset = object  # type: ignore

from src.data.transforms import augment_train, load_image


class ImageRecord:
    """Lightweight in-memory record structure representing a labeled image sample.

    Utilizes `__slots__` to minimize memory overhead when indexing large collections
    of dataset records across training and validation splits.

    Attributes:
        path: Path to the image file on disk.
        label: Binary ground truth classification label (0 or 1).
    """

    __slots__ = ("path", "label")

    def __init__(self, path: Union[str, Path], label: int) -> None:
        self.path: Path = Path(path)
        self.label: int = int(label)

    def __repr__(self) -> str:
        return f"ImageRecord(path={self.path.name}, label={self.label})"


class RGBDataset(_BaseDataset):
    """PyTorch Dataset for labeled training and validation images.

    Applies Sobel edge-filtering on the fly via `load_image` and conditionally
    applies stochastic photometric/geometric augmentations during training.

    Args:
        records: Sequence of `ImageRecord` objects containing file paths and labels.
        augment: Boolean flag enabling training data augmentation if True (default: False).
    """

    def __init__(self, records: Sequence[ImageRecord], augment: bool = False) -> None:
        self.records: Sequence[ImageRecord] = records
        self.augment: bool = augment

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.records)

    def __getitem__(self, i: int) -> Tuple[Any, Any]:
        """Retrieves and processes the i-th sample.

        Args:
            i: Integer index of the sample to fetch.

        Returns:
            A tuple `(image_tensor, label_tensor)`:
            - `image_tensor`: Float tensor of shape `(3, H, W)` containing edge gradients.
            - `label_tensor`: Float scalar tensor containing the target label.

        Raises:
            ImportError: If PyTorch is not installed.
        """
        if torch is None:
            raise ImportError("PyTorch is required to use RGBDataset.")
        rec: ImageRecord = self.records[i]
        raw_img = load_image(rec.path)
        img = augment_train(raw_img) if self.augment else raw_img

        # Convert HWC numpy array to CHW PyTorch float tensor
        tensor_img: torch.Tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
        label_tensor: torch.Tensor = torch.tensor(rec.label, dtype=torch.float32)
        return tensor_img, label_tensor


class TestPathDataset(_BaseDataset):
    """PyTorch Dataset for unlabelled test images yielding preprocessed tensors and filenames.

    Args:
        paths: Sequence of filesystem paths to the test images.
    """

    def __init__(self, paths: Sequence[Union[str, Path]]) -> None:
        self.paths: List[Path] = [Path(p) for p in paths]

    def __len__(self) -> int:
        """Returns the total number of test images."""
        return len(self.paths)

    def __getitem__(self, i: int) -> Tuple[Any, str]:
        """Retrieves and processes the i-th test sample.

        Args:
            i: Integer index of the test sample.

        Returns:
            A tuple `(image_tensor, filename)`:
            - `image_tensor`: Float tensor of shape `(3, H, W)` containing edge gradients.
            - `filename`: String basename of the image file (e.g. 'test_0001.png').

        Raises:
            ImportError: If PyTorch is not installed.
        """
        if torch is None:
            raise ImportError("PyTorch is required to use TestPathDataset.")
        path: Path = self.paths[i]
        raw_img = load_image(path)

        # Convert HWC numpy array to CHW PyTorch float tensor
        tensor_img: torch.Tensor = torch.from_numpy(raw_img.transpose(2, 0, 1)).float()
        return tensor_img, str(path.name)
