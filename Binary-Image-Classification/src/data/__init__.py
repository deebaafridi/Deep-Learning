"""Dataset loaders, image transformation pipelines, splitting routines, and validators."""

from __future__ import annotations

from src.data.download import COMPETITION_ID, COMPETITION_URL, download_kaggle_dataset
from src.data.validator import DatasetValidator, ValidationReport

try:
    from src.data.dataset import ImageRecord, RGBDataset, TestPathDataset
    from src.data.split import build_split, data_split, list_images
    from src.data.transforms import _hsv_jitter, augment_train, load_image
except ImportError:
    pass

__all__: list[str] = [
    "download_kaggle_dataset",
    "COMPETITION_ID",
    "COMPETITION_URL",
    "DatasetValidator",
    "ValidationReport",
    "list_images",
    "build_split",
    "data_split",
]

