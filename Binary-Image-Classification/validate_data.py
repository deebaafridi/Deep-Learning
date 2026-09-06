#!/usr/bin/env python3
"""Dataset and Submission CSV Schema Validation CLI for DualCNN.

Validates the dataset folder hierarchy and generated submission CSV columns
against the declarative rules specified in `dataset_schema.yaml`.

Validation Checks Conducted:
    1. Directory Structure:
       - Presence of `train/` split with class subdirectories `0/` and `1/`.
       - Presence of `test/` split with unlabelled query images.
       - Verification that image files use allowed extensions (.png, .jpg, .jpeg).
       - Detection of empty folders or unexpected non-image artifacts.
    2. Submission CSV Format:
       - Presence and ordering of required header columns ('ID', 'Label').
       - Verification that ID values are non-empty and unique.
       - Verification that Label values are valid binary integers (0 or 1).
       - Consistency check matching submission row count against total test images.

CLI Examples:
    # Validate dataset folder layout only:
    $ python validate_data.py --data-dir ./data

    # Validate generated submission CSV only:
    $ python validate_data.py --csv-path submission.csv

    # Validate both simultaneously in strict mode (fails on warnings):
    $ python validate_data.py --data-dir ./data --csv-path submission.csv --strict
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from src.data.validator import DatasetValidator, ValidationReport
from src.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for dataset and CSV schema validation.

    Returns:
        argparse.Namespace populated with parsed options:
            data_dir (Optional[str]): Path to dataset root directory.
            csv_path (Optional[str]): Path to submission CSV file.
            schema (str): Path to YAML schema definition file.
            strict (bool): Whether to treat warnings as fatal errors.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Validate dataset folder hierarchy and CSV columns against YAML specification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default=None,
        help="Path to dataset directory containing train/ and test/ splits",
    )
    parser.add_argument(
        "--csv-path",
        "-c",
        type=str,
        default=None,
        help="Path to generated submission.csv file to validate columns and values",
    )
    parser.add_argument(
        "--schema",
        "-s",
        type=str,
        default="dataset_schema.yaml",
        help="Path to YAML schema configuration file",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero code if any validation warnings are found",
    )
    return parser.parse_args()


def main() -> None:
    """Main CLI execution routine for data schema and submission validation."""
    args: argparse.Namespace = parse_args()
    logger: logging.Logger = get_logger("validator")

    # Guard: At least one validation target must be provided
    if not args.data_dir and not args.csv_path:
        print(
            "Error: Please provide either --data-dir, --csv-path, or both to validate.\n"
            "Example: python validate_data.py --data-dir ./data --csv-path submission.csv"
        )
        sys.exit(1)

    # Initialize validator using the YAML schema
    validator: DatasetValidator = DatasetValidator(schema_path=args.schema)
    logger.info(f"Loaded schema configuration from '{args.schema}'")

    data_path: Optional[Path] = Path(args.data_dir) if args.data_dir else None
    csv_file: Optional[Path] = Path(args.csv_path) if args.csv_path else None

    # Execute comprehensive validation
    report: ValidationReport = validator.validate_all(
        data_dir=data_path,
        csv_path=csv_file,
    )

    print("\nDATASET & COLUMN VALIDATION REPORT")
    print(f"Overall Status: {'PASSED' if report.is_valid else 'FAILED'}")
    print(f"Total Errors:   {len(report.errors)}")
    print(f"Total Warnings: {len(report.warnings)}")

    if report.metrics:
        print("\nMetrics & Distributions:")
        for metric_name, metric_val in report.metrics.items():
            print(f"  • {metric_name}: {metric_val}")

    if report.errors:
        print("\nErrors Encountered:")
        for err_msg in report.errors:
            print(f"  [ERROR] {err_msg}")

    if report.warnings:
        print("\nWarnings:")
        for warn_msg in report.warnings:
            print(f"  [WARNING] {warn_msg}")

    print()

    if not report.is_valid or (args.strict and report.warnings):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

