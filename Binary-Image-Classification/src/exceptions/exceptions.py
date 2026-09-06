"""Domain-specific exception hierarchy for the DualCNN pipeline.

Provides structured error classes subclassing both custom base exceptions and
appropriate standard library exceptions (e.g., FileNotFoundError, ValueError,
RuntimeError) to preserve full backward compatibility with generic exception handlers.
"""

from __future__ import annotations


class DualCNNException(Exception):
    """Root base exception for all DualCNN pipeline errors.

    Attributes:
        message: Human-readable diagnostic description of the failure.
    """

    def __init__(self, message: str = "") -> None:
        super().__init__(message)
        self.message: str = message

    def __str__(self) -> str:
        return self.message or self.__class__.__name__


class DataNotFoundError(DualCNNException, FileNotFoundError):
    """Raised when expected dataset directories, split directories, or image files cannot be located.

    Subclasses FileNotFoundError to ensure existing callers catching standard
    file-missing errors will handle this exception seamlessly.
    """


class InvalidDataSplitError(DualCNNException, ValueError):
    """Raised when an invalid dataset split name or malformed split structure is encountered.

    Subclasses ValueError as it indicates an invalid argument or partitioned configuration.
    """


class DataValidationError(DualCNNException, ValueError):
    """Raised when dataset directories, image formats, or CSV columns fail validation against the YAML schema.

    Subclasses ValueError to indicate structural or semantic data inconsistencies.
    """


class ImageProcessingError(DualCNNException, RuntimeError):
    """Raised when loading, decoding, resizing, transforming, or calculating edge filters on an image fails.

    Subclasses RuntimeError to signal execution-time failures in image decoding or filtering.
    """


class ModelCheckpointError(DualCNNException, RuntimeError):
    """Raised when saving, loading, or verifying model checkpoint weights fails.

    Subclasses RuntimeError to represent state serialization or deserialization failures.
    """


class ConfigurationError(DualCNNException, ValueError):
    """Raised when invalid configuration parameters, incompatible dimensions, or out-of-range hyperparameters are provided.

    Subclasses ValueError to represent configuration validation errors.
    """


class TrainingError(DualCNNException, RuntimeError):
    """Raised when an unrecoverable failure occurs during model optimization, backward passes, or validation loops.

    Subclasses RuntimeError to indicate training execution faults.
    """
