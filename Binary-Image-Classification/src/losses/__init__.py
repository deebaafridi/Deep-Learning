"""Composite loss functions and optimization criteria for dual-head learning."""

from __future__ import annotations

from src.losses.loss import DualLoss, dual_loss

__all__: list[str] = ["dual_loss", "DualLoss"]

