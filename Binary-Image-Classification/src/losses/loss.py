"""Composite loss formulation for joint multi-task binary classification in DualCNN.

Formulation Details:
1. Computes probabilities via sigmoid activation:
   p_cube = sigmoid(z_cube), p_sphere = sigmoid(z_sphere)
2. Joint probability:
   p_both = clamp(p_cube * p_sphere, eps, 1 - eps)
3. Logit inversion for numerical stability with BCEWithLogits:
   z_both = log(p_both / (1 - p_both))
4. Joint loss:
   L_joint = BCEWithLogits(z_both, y)
5. Auxiliary supervision on positive targets (y > 0.5):
   L_ind = BCEWithLogits(z_cube, y) + BCEWithLogits(z_sphere, y)
6. Total objective:
   L_total = L_joint + 0.5 * L_ind
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dual_loss(
    logit_cube: torch.Tensor,
    logit_sphere: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Computes composite dual-head loss combining joint probability BCE with auxiliary supervision.

    Args:
        logit_cube: Predicted unnormalized cube logits `(batch_size,)`.
        logit_sphere: Predicted unnormalized sphere logits `(batch_size,)`.
        y: Ground truth binary target tensor `(batch_size,)` in {0.0, 1.0}.
        eps: Small numerical stability constant to avoid log(0) errors (default: 1e-6).

    Returns:
        Scalar loss tensor combining joint loss and auxiliary individual head loss.

    Note:
        The auxiliary loss `0.5 * L_ind` is only computed for samples belonging
        to the positive class (y > 0.5) to explicitly incentivize both individual
        heads to fire when a composite scene contains both primitives.
    """
    # 1. Transform logits into individual object probabilities
    p_cube: torch.Tensor = torch.sigmoid(logit_cube)
    p_sphere: torch.Tensor = torch.sigmoid(logit_sphere)

    # 2. Joint probability representing logical AND condition: P(cube AND sphere)
    p_both: torch.Tensor = (p_cube * p_sphere).clamp(eps, 1.0 - eps)

    # 3. Invert sigmoid to retrieve joint logit: log(p / (1 - p))
    logit_both: torch.Tensor = torch.log(p_both) - torch.log(1.0 - p_both)

    # 4. Compute primary joint binary cross-entropy
    l_joint: torch.Tensor = F.binary_cross_entropy_with_logits(logit_both, y)

    # 5. Compute auxiliary individual supervision for positive instances (y > 0.5)
    mask_positive: torch.Tensor = y > 0.5
    l_ind: torch.Tensor = torch.zeros(1, device=y.device)

    if mask_positive.any():
        l_ind = (
            F.binary_cross_entropy_with_logits(logit_cube[mask_positive], y[mask_positive])
            + F.binary_cross_entropy_with_logits(logit_sphere[mask_positive], y[mask_positive])
        )

    # 6. Combined loss with 0.5 weighting factor on auxiliary supervision
    return l_joint + 0.5 * l_ind


class DualLoss(nn.Module):
    """PyTorch Module wrapper for the composite dual loss function.

    Args:
        eps: Small numerical clamping epsilon (default: 1e-6).
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps: float = eps

    def forward(
        self, logit_cube: torch.Tensor, logit_sphere: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Executes the dual_loss function.

        Args:
            logit_cube: Unnormalized cube logits `(B,)`.
            logit_sphere: Unnormalized sphere logits `(B,)`.
            y: Binary target labels `(B,)`.

        Returns:
            Scalar tensor representing total dual loss.
        """
        return dual_loss(logit_cube, logit_sphere, y, eps=self.eps)
