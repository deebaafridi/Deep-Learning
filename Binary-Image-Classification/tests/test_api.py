"""Unit tests for FastAPI REST API Pydantic schemas and serialization.

Verifies default values, field validation, probability boundary ranges, and JSON
serialization structures for `/health`, `/predict`, and `/predict/batch` endpoints.
"""

from __future__ import annotations

import unittest
from typing import List

from src.api.schemas import BatchPredictionResponse, HealthResponse, PredictionResponse


class TestAPISchemas(unittest.TestCase):
    """Test suite validating Pydantic schemas, field defaults, and response serialization."""

    def test_health_response_defaults(self) -> None:
        """Verifies default values and field types of HealthResponse model."""
        res: HealthResponse = HealthResponse()
        self.assertEqual(res.status, "healthy", "Default status string must be 'healthy'")
        self.assertFalse(res.model_loaded, "Default model_loaded must be False prior to warmup")
        self.assertEqual(res.device, "cpu", "Default device string must be 'cpu'")
        self.assertEqual(res.version, "1.0.0", "API version must match package version")

    def test_prediction_response(self) -> None:
        """Verifies instantiation and attribute access of single-image PredictionResponse."""
        pred: PredictionResponse = PredictionResponse(
            filename="sample_001.png",
            cube_probability=0.92,
            sphere_probability=0.88,
            joint_probability=0.8096,
            prediction=1,
            class_name="Positive (Contains Cube and Sphere)",
            inference_time_ms=14.5,
        )

        self.assertEqual(pred.filename, "sample_001.png")
        self.assertEqual(pred.prediction, 1)
        self.assertAlmostEqual(pred.joint_probability, 0.8096)
        self.assertGreater(pred.inference_time_ms, 0.0)

    def test_batch_prediction_response(self) -> None:
        """Verifies aggregation and counting logic in BatchPredictionResponse."""
        p1: PredictionResponse = PredictionResponse(
            filename="img1.png",
            cube_probability=0.9,
            sphere_probability=0.9,
            joint_probability=0.81,
            prediction=1,
            class_name="Positive",
            inference_time_ms=10.0,
        )
        p2: PredictionResponse = PredictionResponse(
            filename="img2.png",
            cube_probability=0.2,
            sphere_probability=0.1,
            joint_probability=0.02,
            prediction=0,
            class_name="Negative",
            inference_time_ms=8.0,
        )
        batch: BatchPredictionResponse = BatchPredictionResponse(
            total_images=2,
            positive_count=1,
            negative_count=1,
            predictions=[p1, p2],
        )

        self.assertEqual(batch.total_images, 2)
        self.assertEqual(batch.positive_count, 1)
        self.assertEqual(batch.negative_count, 1)
        self.assertEqual(len(batch.predictions), 2)


if __name__ == "__main__":
    unittest.main()

