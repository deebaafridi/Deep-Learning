"""Unit tests for neural network architectures, layers, and dual classification heads.

Verifies coordinate channel concatenation (`AddCoords`), residual downsampling
convolutional blocks (`ConvBlock`), and the complete end-to-end forward pass
of `DualHeadCNN`. Skipped automatically on environments without PyTorch.
"""

from __future__ import annotations

import unittest
from typing import Any, Tuple

try:
    import torch
    TORCH_AVAILABLE: bool = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for model tests")
class TestModels(unittest.TestCase):
    """Test suite validating neural network layers, shape transforms, and dual prediction heads."""

    def setUp(self) -> None:
        """Dynamically imports model components when PyTorch is available."""
        from src.models.dual_head_cnn import DualHeadCNN
        from src.models.layers import AddCoords, ConvBlock

        self.AddCoords: Any = AddCoords
        self.ConvBlock: Any = ConvBlock
        self.DualHeadCNN: Any = DualHeadCNN

    def test_add_coords_output_shape(self) -> None:
        """Verifies that AddCoords appends normalized [-1, 1] X and Y channels to the input tensor."""
        add_coords = self.AddCoords()
        x: torch.Tensor = torch.randn(2, 3, 32, 32)
        out: torch.Tensor = add_coords(x)

        # 3 input image channels + 2 coordinate channels (x_coords, y_coords) = 5 total channels
        self.assertEqual(
            out.shape,
            (2, 5, 32, 32),
            "AddCoords must expand channel dimension from 3 to 5 preserving spatial height and width",
        )

    def test_conv_block_shapes(self) -> None:
        """Verifies that ConvBlock with stride=2 halves spatial dimensions and projects channels."""
        block = self.ConvBlock(c_in=32, c_out=64, stride=2)
        x: torch.Tensor = torch.randn(2, 32, 32, 32)
        out: torch.Tensor = block(x)

        # Stride 2 downsamples 32x32 -> 16x16 with 64 feature channels
        self.assertEqual(
            out.shape,
            (2, 64, 16, 16),
            "ConvBlock must produce expected output tensor dimensions with residual projection",
        )

    def test_dual_head_cnn_forward(self) -> None:
        """Verifies DualHeadCNN forward pass returns two scalar logits per batch item."""
        model = self.DualHeadCNN()
        model.eval()
        x: torch.Tensor = torch.randn(2, 3, 96, 96)

        with torch.no_grad():
            logit_cube: torch.Tensor
            logit_sphere: torch.Tensor
            logit_cube, logit_sphere = model(x)

        self.assertEqual(logit_cube.shape, (2,), "Cube logit shape must match batch size (B,)")
        self.assertEqual(logit_sphere.shape, (2,), "Sphere logit shape must match batch size (B,)")
        self.assertFalse(torch.isnan(logit_cube).any(), "Cube logits must not contain NaN")
        self.assertFalse(torch.isnan(logit_sphere).any(), "Sphere logits must not contain NaN")


if __name__ == "__main__":
    unittest.main()

