"""Unit tests for YAML schema-driven dataset structure and submission CSV validation.

Verifies validation reporting (`ValidationReport`), missing or corrupted column
handling in submission CSV files, label value verification (must be 0 or 1),
duplicate identifier detection, and hierarchical filesystem structure checks.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from typing import Dict

from src.data.validator import DatasetValidator, ValidationReport
from src.exceptions.exceptions import DataValidationError


class TestDatasetValidator(unittest.TestCase):
    """Test suite validating schema-driven directory layout and submission CSV integrity."""

    def setUp(self) -> None:
        """Initializes a DatasetValidator instance loaded with the project schema configuration."""
        self.validator: DatasetValidator = DatasetValidator(schema_path="dataset_schema.yaml")

    def test_validation_report_error_handling(self) -> None:
        """Verifies ValidationReport records error messages and raises DataValidationError on raise_for_status()."""
        report: ValidationReport = ValidationReport()
        self.assertTrue(report.is_valid, "Fresh report must be valid initially")

        report.add_error("Test failure reason")
        self.assertFalse(report.is_valid, "Report must be invalid after adding an error")
        self.assertEqual(len(report.errors), 1)

        with self.assertRaises(DataValidationError):
            report.raise_for_status()

    def test_valid_submission_csv(self) -> None:
        """Verifies that a well-formed submission CSV passes validation and populates summary metrics."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Label"])
            writer.writerow(["image_001.png", 1])
            writer.writerow(["image_002.png", 0])
            writer.writerow(["image_003.png", 1])
            csv_path: Path = Path(f.name)

        try:
            report: ValidationReport = self.validator.validate_submission_csv(csv_path)
            self.assertTrue(report.is_valid, f"Validation failed unexpectedly: {report.errors}")
            self.assertEqual(report.metrics.get("total_rows"), 3)
            distribution: Dict[int, int] = report.metrics.get("label_distribution", {})
            self.assertEqual(distribution.get(1), 2)
            self.assertEqual(distribution.get(0), 1)
        finally:
            csv_path.unlink(missing_ok=True)

    def test_missing_column_csv(self) -> None:
        """Verifies that omission of the mandatory 'Label' column flags a validation error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Score"])  # Missing required "Label" column
            writer.writerow(["image_001.png", 0.9])
            csv_path: Path = Path(f.name)

        try:
            report: ValidationReport = self.validator.validate_submission_csv(csv_path)
            self.assertFalse(report.is_valid)
            self.assertTrue(any("Missing required CSV column: 'Label'" in e for e in report.errors))
        finally:
            csv_path.unlink(missing_ok=True)

    def test_invalid_label_values_csv(self) -> None:
        """Verifies that non-binary or unparseable labels (e.g. 2, 'invalid') trigger validation errors."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Label"])
            writer.writerow(["img1.png", 2])  # 2 is outside allowed binary set {0, 1}
            writer.writerow(["img2.png", "invalid_string"])
            csv_path: Path = Path(f.name)

        try:
            report: ValidationReport = self.validator.validate_submission_csv(csv_path)
            self.assertFalse(report.is_valid)
            self.assertTrue(any("not in allowed set" in e for e in report.errors))
            self.assertTrue(any("not a valid integer" in e for e in report.errors))
        finally:
            csv_path.unlink(missing_ok=True)

    def test_duplicate_ids_csv(self) -> None:
        """Verifies that duplicate image identifiers in the submission CSV trigger validation errors."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Label"])
            writer.writerow(["duplicate.png", 1])
            writer.writerow(["duplicate.png", 0])
            csv_path: Path = Path(f.name)

        try:
            report: ValidationReport = self.validator.validate_submission_csv(csv_path)
            self.assertFalse(report.is_valid)
            self.assertTrue(any("Duplicate identifier 'duplicate.png'" in e for e in report.errors))
        finally:
            csv_path.unlink(missing_ok=True)

    def test_directory_structure_validation(self) -> None:
        """Verifies dataset folder validation passes on compliant hierarchy and records sample counts."""
        with tempfile.TemporaryDirectory() as tmp:
            base: Path = Path(tmp)
            # Create valid training and test split structures
            (base / "train" / "0").mkdir(parents=True)
            (base / "train" / "1").mkdir(parents=True)
            (base / "test").mkdir(parents=True)

            (base / "train" / "0" / "0_001.png").touch()
            (base / "train" / "1" / "1_001.png").touch()
            (base / "test" / "test_001.png").touch()

            report: ValidationReport = self.validator.validate_directory_structure(base)
            self.assertTrue(report.is_valid, f"Directory validation failed unexpectedly: {report.errors}")
            self.assertEqual(report.metrics["test_count"], 1)
            self.assertEqual(report.metrics["train_counts"]["class_0"], 1)
            self.assertEqual(report.metrics["train_counts"]["class_1"], 1)


if __name__ == "__main__":
    unittest.main()

