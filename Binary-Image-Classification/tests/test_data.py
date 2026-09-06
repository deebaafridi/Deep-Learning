"""Unit tests for dataset loading, path resolution, and record structures.

Verifies memory-efficient `ImageRecord` container functionality, flat vs. nested
`train/` directory resolution, file extension filtering, and custom exception
handling when directories are missing.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import List

from src.data.dataset import ImageRecord
from src.data.split import data_split, list_images
from src.exceptions.exceptions import DataNotFoundError


class TestDataUtilities(unittest.TestCase):
    """Test suite validating dataset records, path resolvers, and file scanners."""

    def test_image_record_slots(self) -> None:
        """Verifies that ImageRecord properly initializes path, integer label, and string representation."""
        rec: ImageRecord = ImageRecord("sample.png", 1)
        self.assertEqual(rec.path, Path("sample.png"))
        self.assertEqual(rec.label, 1)
        self.assertIn("sample.png", repr(rec))

    def test_data_split_resolution_flat(self) -> None:
        """Verifies standard flat directory resolution (`base / train`)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            base: Path = Path(tmp_dir)
            train_dir: Path = base / "train"
            train_dir.mkdir()
            resolved: Path = data_split(base, "train")
            self.assertEqual(resolved, train_dir)

    def test_data_split_resolution_nested(self) -> None:
        """Verifies resolution when competition zip creates an extra nested directory (`base / train / train`)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            base: Path = Path(tmp_dir)
            nested: Path = base / "train" / "train"
            nested.mkdir(parents=True)
            resolved: Path = data_split(base, "train")
            self.assertEqual(resolved, nested)

    def test_data_split_missing_raises_error(self) -> None:
        """Verifies DataNotFoundError is raised when a requested split directory does not exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(DataNotFoundError):
                data_split(Path(tmp_dir), "nonexistent_split")

    def test_list_images_filters_extensions(self) -> None:
        """Verifies list_images scans only files matching specified image extensions."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            base: Path = Path(tmp_dir)
            (base / "img1.png").touch()
            (base / "img2.jpg").touch()
            (base / "readme.txt").touch()
            (base / "script.py").touch()

            found: List[Path] = list_images(base, allowed_extensions={".png", ".jpg"})
            found_names: List[str] = [f.name for f in found]
            self.assertEqual(sorted(found_names), ["img1.png", "img2.jpg"])

    def test_list_images_missing_dir_raises_error(self) -> None:
        """Verifies list_images raises DataNotFoundError when pointing to a nonexistent path."""
        with self.assertRaises(DataNotFoundError):
            list_images(Path("/path/that/does/not/exist/999"))


if __name__ == "__main__":
    unittest.main()

