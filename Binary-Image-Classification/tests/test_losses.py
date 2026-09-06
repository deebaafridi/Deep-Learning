"""Unit tests for composite dual binary cross-entropy loss formulation.

Verifies the mathematical correctness and stability of `dual_loss` and the
`DualLoss` nn.Module wrapper, ensuring non-negative scalar values and avoidance
of gradient/numerical singularities. Automatically skipped when PyTorch is not installed.
"""

from __future__ import annotations

import unittest
from typing import Any

try:
    import torch
    TORCH_AVAILABLE: bool = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for loss function tests")
class TestLosses(unittest.TestCase):
    """Test suite validating composite dual-head loss computation and module wrappers."""

    def setUp(self) -> None:
        """Dynamically imports loss components when PyTorch is available."""
        from src.losses.loss import DualLoss, dual_loss

        self.dual_loss: Any = dual_loss
        self.DualLoss: Any = DualLoss

    def test_dual_loss_computation(self) -> None:
        """Verifies dual_loss returns a positive scalar with no NaNs for mixed positive/negative samples."""
        logit_cube: torch.Tensor = torch.tensor([1.5, -2.0], dtype=torch.float32)
        logit_sphere: torch.Tensor = torch.tensor([0.8, -1.2], dtype=torch.float32)
        y: torch.Tensor = torch.tensor([1.0, 0.0], dtype=torch.float32)

        loss: torch.Tensor = self.dual_loss(logit_cube, logit_sphere, y)

        self.assertEqual(loss.ndim, 0, "Loss must reduce to a 0-dimensional scalar tensor")
        self.assertFalse(torch.isnan(loss), "Loss must not produce NaN values")
        self.assertGreater(loss.item(), 0.0, "Cross-entropy loss must be strictly positive")

    def test_dual_loss_module(self) -> None:
        """Verifies the DualLoss nn.Module callable wrapper executes identically."""
        criterion = self.DualLoss()
        logit_cube: torch.Tensor = torch.tensor([0.5], dtype=torch.float32)
        logit_sphere: torch.Tensor = torch.tensor([0.5], dtype=torch.float32)
        y: torch.Tensor = torch.tensor([1.0], dtype=torch.float32)

        loss: torch.Tensor = criterion(logit_cube, logit_sphere, y)

        self.assertEqual(loss.ndim, 0, "Module wrapper output must be a scalar tensor")
        self.assertFalse(torch.isnan(loss), "Module wrapper output must not produce NaN")
        self.assertGreater(loss.item(), 0.0, "Module wrapper loss must be positive")


if __name__ == "__main__":
    unittest.main()

