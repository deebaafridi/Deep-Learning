"""Custom neural network layers and residual building blocks for DualCNN.

Implements:
1. `AddCoords`: Coordinate Convolution (CoordConv) layer encoding explicit spatial positions.
2. `ConvBlock`: Residual convolutional block with Instance Normalization and identity/projection shortcuts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AddCoords(nn.Module):
    """Coordinate Convolution (CoordConv) layer.

    Appends two extra continuous spatial coordinate feature maps (X and Y)
    normalized to [-1, 1] along the channel dimension. This equips convolutional
    filters with explicit translational coordinate awareness, addressing standard
    convolutional translation invariance when detecting absolute spatial geometry.

    Input shape:  `(batch_size, channels, height, width)`
    Output shape: `(batch_size, channels + 2, height, width)`
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenates normalized coordinate grids to the input feature tensor.

        Args:
            x: Input tensor of shape `(B, C, H, W)`.

        Returns:
            Output tensor of shape `(B, C + 2, H, W)` containing coordinate maps.
        """
        b, c, h, w = x.size()

        # Generate linear coordinate ramps in range [-1.0, 1.0]
        y_coords: torch.Tensor = (
            torch.linspace(-1.0, 1.0, h, device=x.device)
            .view(1, 1, h, 1)
            .expand(b, 1, h, w)
        )
        x_coords: torch.Tensor = (
            torch.linspace(-1.0, 1.0, w, device=x.device)
            .view(1, 1, 1, w)
            .expand(b, 1, h, w)
        )

        # Concatenate along channel dimension (dim=1)
        return torch.cat([x, y_coords, x_coords], dim=1)


class ConvBlock(nn.Module):
    """Residual convolutional block with Instance Normalization and ReLU activation.

    Comprises two 3x3 convolutions with Instance Normalization (`InstanceNorm2d`).
    Instance Normalization is chosen over standard BatchNorm because it normalizes
    across spatial dimensions independently per channel and per sample, providing
    invariance to synthetic lighting, specular contrast, and style variations.

    Args:
        c_in: Number of input channels.
        c_out: Number of output channels.
        stride: Convolutional stride for spatial downsampling (default is 1).
    """

    def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
        super().__init__()
        # First convolutional stage with optional spatial stride
        self.conv1: nn.Conv2d = nn.Conv2d(
            c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1: nn.InstanceNorm2d = nn.InstanceNorm2d(c_out, affine=True)

        # Second convolutional stage (stride 1)
        self.conv2: nn.Conv2d = nn.Conv2d(
            c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2: nn.InstanceNorm2d = nn.InstanceNorm2d(c_out, affine=True)

        # Residual shortcut: 1x1 projection if shape or channels change, else identity
        if stride != 1 or c_in != c_out:
            self.short: nn.Module = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(c_out, affine=True),
            )
        else:
            self.short = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block with skip connection addition.

        Args:
            x: Input feature tensor `(B, c_in, H, W)`.

        Returns:
            Activated residual sum `(B, c_out, H // stride, W // stride)`.
        """
        shortcut: torch.Tensor = self.short(x)
        out: torch.Tensor = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)) + shortcut, inplace=True)
        return out
