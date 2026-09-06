"""YAML-driven dataset hierarchy and submission CSV column schema validator for DualCNN.

Validates that:
1. Expected train/test directory structures exist with allowed image formats and sample counts.
2. Submission CSV files conform to required column names, non-null rules, unique keys, and label ranges.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    yaml = None
    YAML_AVAILABLE = False

from src.data.split import data_split, list_images
from src.exceptions.exceptions import DataValidationError
from src.utils.logger import get_logger

logger: logging.Logger = get_logger("solution")

# Default embedded fallback schema if YAML file is missing or unreadable
DEFAULT_SCHEMA: Dict[str, Any] = {
    "dataset": {
        "name": "iith-deep-learning-2026-hackathon",
        "splits": {
            "train": {
                "required": True,
                "classes": [
                    {"label": 0, "min_samples": 1},
                    {"label": 1, "min_samples": 1},
                ],
                "allowed_extensions": [".png", ".jpg", ".jpeg", ".bmp"],
            },
            "test": {
                "required": True,
                "min_samples": 1,
                "allowed_extensions": [".png", ".jpg", ".jpeg", ".bmp"],
            },
        },
        "submission_csv": {
            "expected_columns": [
                {"name": "ID", "type": "string", "allow_null": False, "unique": True},
                {"name": "Label", "type": "integer", "allow_null": False, "allowed_values": [0, 1]},
            ]
        },
    }
}


@dataclass
class ValidationReport:
    """Encapsulates the outcome of dataset structure and CSV schema validations.

    Attributes:
        is_valid: Boolean indicating overall validation success.
        errors: List of critical validation error messages.
        warnings: List of non-fatal diagnostic warning messages.
        metrics: Dictionary capturing summary statistics (e.g., sample counts, label distribution).
    """

    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        """Appends an error message and marks the report as invalid.

        Args:
            message: Descriptive error message string.
        """
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Appends a diagnostic warning message without invalidating the report.

        Args:
            message: Informational warning message string.
        """
        self.warnings.append(message)

    def raise_for_status(self) -> None:
        """Raises DataValidationError if errors were encountered during validation.

        Raises:
            DataValidationError: If `is_valid` is False.
        """
        if not self.is_valid:
            err_summary: str = "\n".join(f"- {e}" for e in self.errors)
            raise DataValidationError(
                f"Dataset schema validation failed with {len(self.errors)} error(s):\n{err_summary}"
            )

    def __repr__(self) -> str:
        status: str = "PASSED" if self.is_valid else "FAILED"
        return (
            f"<ValidationReport: status={status}, errors={len(self.errors)}, "
            f"warnings={len(self.warnings)}, metrics={self.metrics}>"
        )


class DatasetValidator:
    """Validates dataset folder structures and submission CSV columns against a declarative YAML schema.

    Args:
        schema_path: Path to YAML configuration file or a pre-loaded dictionary (default: 'dataset_schema.yaml').
    """

    def __init__(self, schema_path: Union[str, Path, Dict[str, Any]] = "dataset_schema.yaml") -> None:
        if isinstance(schema_path, dict):
            self.schema: Dict[str, Any] = schema_path
        else:
            self.schema = self._load_schema(Path(schema_path))

    def _load_schema(self, path: Path) -> Dict[str, Any]:
        """Loads and parses the YAML schema file, falling back to embedded defaults if absent.

        Args:
            path: Path to target YAML schema file.

        Returns:
            Dictionary containing parsed schema specifications.
        """
        if not path.exists():
            logger.warning(f"Schema file '{path}' not found. Using default embedded schema.")
            return DEFAULT_SCHEMA

        if not YAML_AVAILABLE:
            logger.warning("PyYAML is not installed. Using default embedded schema.")
            return DEFAULT_SCHEMA

        try:
            with path.open("r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict) and "dataset" in loaded:
                    return loaded
                logger.warning(f"Invalid schema structure in '{path}'. Using default embedded schema.")
                return DEFAULT_SCHEMA
        except Exception as exc:
            logger.warning(f"Could not parse YAML schema '{path}': {exc}. Using default embedded schema.")
            return DEFAULT_SCHEMA

    def validate_directory_structure(self, data_dir: Union[str, Path]) -> ValidationReport:
        """Verifies that dataset split directories, class subfolders, and image files meet schema rules.

        Checks:
        1. Existence of base directory.
        2. Presence of required splits (e.g. `train`, `test`) handling nested/flat folders.
        3. Existence of required class subdirectories (`0/`, `1/`).
        4. Image file extensions match allowed set.
        5. Sample counts exceed minimum requirements.

        Args:
            data_dir: Path to root dataset directory.

        Returns:
            ValidationReport detailing detected errors, warnings, and sample counts.
        """
        report: ValidationReport = ValidationReport()
        base_path: Path = Path(data_dir)

        if not base_path.exists() or not base_path.is_dir():
            report.add_error(f"Base data directory does not exist: {base_path}")
            return report

        splits_cfg: Dict[str, Any] = self.schema.get("dataset", {}).get("splits", {})

        # 1. Validate Training Split
        train_cfg: Dict[str, Any] = splits_cfg.get("train", {})
        if train_cfg.get("required", True):
            try:
                train_dir: Path = data_split(base_path, "train")
                allowed_exts: Set[str] = set(
                    train_cfg.get("allowed_extensions", [".png", ".jpg", ".jpeg", ".bmp"])
                )

                classes_cfg: List[Dict[str, Any]] = train_cfg.get("classes", [])
                train_counts: Dict[str, int] = {}
                for cls in classes_cfg:
                    lbl: str = str(cls.get("label", 0))
                    cls_dir: Path = train_dir / lbl
                    if not cls_dir.exists() or not cls_dir.is_dir():
                        report.add_error(f"Missing train class directory: {cls_dir}")
                        continue

                    images: List[Path] = list_images(cls_dir, allowed_extensions=allowed_exts)
                    train_counts[f"class_{lbl}"] = len(images)
                    min_s: int = cls.get("min_samples", 1)
                    if len(images) < min_s:
                        report.add_error(
                            f"Class {lbl} in {cls_dir} has {len(images)} images, expected at least {min_s}."
                        )

                report.metrics["train_counts"] = train_counts
            except Exception as exc:
                report.add_error(f"Train split verification failed: {exc}")

        # 2. Validate Test Split
        test_cfg: Dict[str, Any] = splits_cfg.get("test", {})
        if test_cfg.get("required", True):
            try:
                test_dir: Path = data_split(base_path, "test")
                allowed_exts = set(
                    test_cfg.get("allowed_extensions", [".png", ".jpg", ".jpeg", ".bmp"])
                )
                test_images: List[Path] = list_images(test_dir, allowed_extensions=allowed_exts)
                min_test: int = test_cfg.get("min_samples", 1)
                report.metrics["test_count"] = len(test_images)
                if len(test_images) < min_test:
                    report.add_error(
                        f"Test directory has {len(test_images)} images, expected at least {min_test}."
                    )
            except Exception as exc:
                report.add_error(f"Test split verification failed: {exc}")

        return report

    def validate_submission_csv(self, csv_path: Union[str, Path]) -> ValidationReport:
        """Validates submission CSV format, required column headers, null constraints, and label values.

        Checks:
        1. File existence and CSV parseability.
        2. Expected column headers exist (e.g. `ID`, `Label`).
        3. Non-null constraints enforced across all rows.
        4. Primary key uniqueness for `ID` column.
        5. Valid integer casting and membership in allowed values for `Label` (`[0, 1]`).

        Args:
            csv_path: Path to target CSV file.

        Returns:
            ValidationReport detailing row counts, label distribution, and errors.
        """
        report: ValidationReport = ValidationReport()
        path: Path = Path(csv_path)

        if not path.exists() or not path.is_file():
            report.add_error(f"Submission CSV file not found: {path}")
            return report

        csv_cfg: Dict[str, Any] = self.schema.get("dataset", {}).get("submission_csv", {})
        expected_cols_cfg: List[Dict[str, Any]] = csv_cfg.get(
            "expected_columns",
            [
                {"name": "ID", "type": "string", "allow_null": False, "unique": True},
                {"name": "Label", "type": "integer", "allow_null": False, "allowed_values": [0, 1]},
            ],
        )
        expected_col_names: List[str] = [col["name"] for col in expected_cols_cfg]
        col_rules: Dict[str, Dict[str, Any]] = {col["name"]: col for col in expected_cols_cfg}

        try:
            with path.open("r", encoding="utf-8") as f:
                reader: csv.DictReader = csv.DictReader(f)
                headers: List[str] = reader.fieldnames or []

                # Ensure all required column headers are present
                for exp in expected_col_names:
                    if exp not in headers:
                        report.add_error(f"Missing required CSV column: '{exp}'. Found columns: {headers}")

                seen_ids: Set[str] = set()
                total_rows: int = 0
                label_counts: Dict[int, int] = {0: 0, 1: 0}

                for row_idx, row in enumerate(reader, start=1):
                    total_rows += 1
                    for col_name, rules in col_rules.items():
                        if col_name not in row:
                            continue

                        val: Optional[str] = row[col_name]

                        # 1. Null / empty string validation
                        if not rules.get("allow_null", True) and (val is None or val.strip() == ""):
                            report.add_error(f"Row {row_idx}: Column '{col_name}' contains null or empty value.")

                        # 2. Uniqueness validation
                        if rules.get("unique", False) and val is not None:
                            if val in seen_ids:
                                report.add_error(f"Row {row_idx}: Duplicate identifier '{val}' found in '{col_name}'.")
                            else:
                                seen_ids.add(val)

                        # 3. Integer type and allowed-values validation for Label
                        if col_name == "Label" and val is not None and val.strip() != "":
                            try:
                                int_val: int = int(val)
                                allowed_vals = rules.get("allowed_values")
                                if allowed_vals is not None and int_val not in allowed_vals:
                                    report.add_error(
                                        f"Row {row_idx}: Label value '{int_val}' not in allowed set {allowed_vals}."
                                    )
                                else:
                                    label_counts[int_val] = label_counts.get(int_val, 0) + 1
                            except ValueError:
                                report.add_error(
                                    f"Row {row_idx}: Column 'Label' value '{val}' is not a valid integer."
                                )

                report.metrics["total_rows"] = total_rows
                report.metrics["label_distribution"] = label_counts

                if total_rows == 0:
                    report.add_error("Submission CSV contains 0 data rows.")

        except Exception as exc:
            report.add_error(f"Failed to read or parse CSV file: {exc}")

        return report

    def validate_all(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        csv_path: Optional[Union[str, Path]] = None,
    ) -> ValidationReport:
        """Executes aggregate validation covering both dataset directories and submission CSV files.

        Args:
            data_dir: Optional path to dataset directory.
            csv_path: Optional path to submission CSV file.

        Returns:
            Combined ValidationReport reflecting all performed validations.
        """
        combined_report: ValidationReport = ValidationReport()

        if data_dir is not None:
            dir_report: ValidationReport = self.validate_directory_structure(data_dir)
            combined_report.errors.extend(dir_report.errors)
            combined_report.warnings.extend(dir_report.warnings)
            combined_report.metrics.update(dir_report.metrics)

        if csv_path is not None:
            csv_report: ValidationReport = self.validate_submission_csv(csv_path)
            combined_report.errors.extend(csv_report.errors)
            combined_report.warnings.extend(csv_report.warnings)
            combined_report.metrics.update(csv_report.metrics)

        combined_report.is_valid = len(combined_report.errors) == 0
        return combined_report
