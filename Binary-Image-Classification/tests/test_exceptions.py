"""Unit tests for custom domain exceptions and dual-inheritance hierarchy.

Validates that every pipeline exception inherits from both the root domain class
`DualCNNException` and an appropriate standard library exception (`FileNotFoundError`,
`ValueError`, or `RuntimeError`). This design ensures that both domain-specific
and generic error handlers work properly across external libraries.
"""

from __future__ import annotations

import unittest

from src.exceptions.exceptions import (
    ConfigurationError,
    DataNotFoundError,
    DataValidationError,
    DualCNNException,
    ImageProcessingError,
    InvalidDataSplitError,
    ModelCheckpointError,
    TrainingError,
)


class TestExceptions(unittest.TestCase):
    """Test suite validating domain exception class hierarchies and catch semantics."""

    def test_data_not_found_inheritance(self) -> None:
        """Verifies DataNotFoundError derives from DualCNNException and FileNotFoundError."""
        err: DataNotFoundError = DataNotFoundError("Dataset missing")
        self.assertIsInstance(err, DualCNNException)
        self.assertIsInstance(err, FileNotFoundError)

    def test_invalid_data_split_inheritance(self) -> None:
        """Verifies InvalidDataSplitError derives from DualCNNException and ValueError."""
        err: InvalidDataSplitError = InvalidDataSplitError("Bad split ratio")
        self.assertIsInstance(err, DualCNNException)
        self.assertIsInstance(err, ValueError)

    def test_data_validation_error_inheritance(self) -> None:
        """Verifies DataValidationError derives from DualCNNException and ValueError."""
        err: DataValidationError = DataValidationError("Invalid columns detected")
        self.assertIsInstance(err, DualCNNException)
        self.assertIsInstance(err, ValueError)

    def test_image_processing_inheritance(self) -> None:
        """Verifies ImageProcessingError derives from DualCNNException and RuntimeError."""
        err: ImageProcessingError = ImageProcessingError("Corrupted JPEG bitstream")
        self.assertIsInstance(err, DualCNNException)
        self.assertIsInstance(err, RuntimeError)

    def test_model_checkpoint_inheritance(self) -> None:
        """Verifies ModelCheckpointError derives from DualCNNException and RuntimeError."""
        err: ModelCheckpointError = ModelCheckpointError("Checkpoint file corrupted")
        self.assertIsInstance(err, DualCNNException)
        self.assertIsInstance(err, RuntimeError)

    def test_configuration_error_inheritance(self) -> None:
        """Verifies ConfigurationError derives from DualCNNException and ValueError."""
        err: ConfigurationError = ConfigurationError("Invalid batch size specified")
        self.assertIsInstance(err, DualCNNException)
        self.assertIsInstance(err, ValueError)

    def test_training_error_inheritance(self) -> None:
        """Verifies TrainingError derives from DualCNNException and RuntimeError."""
        err: TrainingError = TrainingError("Gradient explosion encountered during epoch")
        self.assertIsInstance(err, DualCNNException)
        self.assertIsInstance(err, RuntimeError)

    def test_catch_by_base_class(self) -> None:
        """Verifies that catching `DualCNNException` catches all domain-derived errors."""
        with self.assertRaises(DualCNNException):
            raise DataNotFoundError("Test path not found")

    def test_catch_by_standard_exception(self) -> None:
        """Verifies that catching built-in `FileNotFoundError` catches `DataNotFoundError`."""
        with self.assertRaises(FileNotFoundError):
            raise DataNotFoundError("Standard handler catch")


if __name__ == "__main__":
    unittest.main()

