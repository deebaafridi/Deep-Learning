"""Dual-Head Convolutional Neural Network architecture for DualCNN.

Decomposes the binary classification task into two concurrent object-presence
detection sub-tasks:
1. Cube presence detection (`head_cube`)
2. Sphere presence detection (`head_sphere`)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from src.models.layers import AddCoords, ConvBlock


class DualHeadCNN(nn.Module):
    """Deep Convolutional Neural Network with dual binary classification heads.

    Architecture Overview:
        1. **Stem (5 channels -> 32 channels)**:
           - Injects X/Y coordinate planes via `AddCoords` (3 RGB -> 5 channels).
           - 3x3 convolution, InstanceNorm2d, ReLU.
        2. **Stage 1 (32 -> 64 channels)**:
           - Stride-2 residual downsampling + identity residual block.
        3. **Stage 2 (64 -> 128 channels)**:
           - Stride-2 residual downsampling + identity residual block.
        4. **Stage 3 (128 -> 256 channels)**:
           - Stride-2 residual downsampling + identity residual block.
        5. **Stage 4 (256 -> 384 channels)**:
           - Stride-2 residual downsampling + identity residual block.
        6. **Feature Aggregator**:
           - Adaptive Max Pooling (`AdaptiveMaxPool2d(1)`): extracts strongest visual signals.
           - Dropout regularization (`p=0.3`).
        7. **Dual Output Heads**:
           - Linear head for cube presence logit (`head_cube`: 384 -> 1).
           - Linear head for sphere presence logit (`head_sphere`: 384 -> 1).

    Args:
        dropout_rate: Dropout probability applied to the pooled 384-d feature vector (default is 0.3).
    """

    def __init__(self, dropout_rate: float = 0.3) -> None:
        super().__init__()
        # Stem: CoordConv + 3x3 Conv + InstanceNorm + ReLU
        self.stem: nn.Sequential = nn.Sequential(
            AddCoords(),
            nn.Conv2d(5, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
        )

        # 4 Residual downsampling stages
        self.l1: nn.Sequential = nn.Sequential(
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 64, stride=1),
        )
        self.l2: nn.Sequential = nn.Sequential(
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 128, stride=1),
        )
        self.l3: nn.Sequential = nn.Sequential(
            ConvBlock(128, 256, stride=2),
            ConvBlock(256, 256, stride=1),
        )
        self.l4: nn.Sequential = nn.Sequential(
            ConvBlock(256, 384, stride=2),
            ConvBlock(384, 384, stride=1),
        )

        # Global pooling and regularization
        self.pool: nn.AdaptiveMaxPool2d = nn.AdaptiveMaxPool2d(1)
        self.drop: nn.Dropout = nn.Dropout(dropout_rate)

        # Independent classification heads
        self.head_cube: nn.Linear = nn.Linear(384, 1)
        self.head_sphere: nn.Linear = nn.Linear(384, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs forward feature extraction and produces independent head logits.

        Args:
            x: Input image tensor of shape `(batch_size, 3, height, width)`.

        Returns:
            A tuple `(logit_cube, logit_sphere)`:
            - `logit_cube`: Raw unnormalized logit tensor for cube presence `(batch_size,)`.
            - `logit_sphere`: Raw unnormalized logit tensor for sphere presence `(batch_size,)`.
        """
        # Feature hierarchy extraction
        feat: torch.Tensor = self.stem(x)
        feat = self.l1(feat)
        feat = self.l2(feat)
        feat = self.l3(feat)
        feat = self.l4(feat)

        # Global max pooling and dropout
        pooled: torch.Tensor = self.drop(self.pool(feat).flatten(1))

        # Head projections squeezed to 1D tensors
        logit_cube: torch.Tensor = self.head_cube(pooled).squeeze(1)
        logit_sphere: torch.Tensor = self.head_sphere(pooled).squeeze(1)

        return logit_cube, logit_sphere
