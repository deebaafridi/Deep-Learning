"""Training and validation engine routines for DualCNN.

Contains:
1. `validate` (`_val`): Evaluates validation accuracy over both heads simultaneously.
2. `train`: Orchestrates the training loop with AdamW optimizer, Cosine Annealing,
   gradient clipping, early stopping patience, and best state dict checkpointing.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Optional

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ImportError:
    torch = None
    nn = None
    DataLoader = None

from src.config.config import Config, DEVICE
from src.exceptions.exceptions import ModelCheckpointError, TrainingError
from src.losses.loss import dual_loss
from src.models.dual_head_cnn import DualHeadCNN
from src.utils.logger import get_logger


def validate(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[Any] = None,
) -> float:
    """Computes classification accuracy across the validation DataLoader.

    Evaluates the joint condition: both cube and sphere heads must independently
    predict probabilities above 0.5 to declare a positive class detection:
    `pred = (sigmoid(lc) > 0.5) & (sigmoid(ls) > 0.5)`

    Args:
        model: PyTorch neural network module (typically `DualHeadCNN`).
        loader: Validation PyTorch DataLoader yielding `(batch_images, batch_labels)`.
        device: Hardware device for tensor evaluation. Defaults to `config.DEVICE`.

    Returns:
        Validation classification accuracy as a float in range `[0.0, 1.0]`.

    Raises:
        ImportError: If PyTorch is unavailable.
    """
    if torch is None:
        raise ImportError("PyTorch is required for validation execution.")

    target_device: Any = device if device is not None else DEVICE
    model.eval()
    correct: int = 0
    total: int = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(target_device), y.to(target_device)
            lc, ls = model(x)

            # Combined positive decision: both heads must satisfy sigmoid > 0.5
            preds: torch.Tensor = (
                (torch.sigmoid(lc) > 0.5) & (torch.sigmoid(ls) > 0.5)
            ).long()

            correct += (preds == y.long()).sum().item()
            total += y.size(0)

    return correct / max(total, 1)


# Backward-compatible function alias matching original script
_val = validate


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Optional[Config] = None,
    logger: Optional[logging.Logger] = None,
) -> DualHeadCNN:
    """Trains the DualHeadCNN model according to competition hyperparameters.

    Optimization Setup:
        - **Optimizer**: `torch.optim.AdamW` with learning rate `cfg.learning_rate` and `cfg.weight_decay`.
        - **Scheduler**: `torch.optim.lr_scheduler.CosineAnnealingLR` decaying to `1e-6` over `cfg.epochs`.
        - **Loss**: `dual_loss` (joint BCE + positive auxiliary BCE).
        - **Regularization**: Gradient clipping (`clip_grad_norm_`) bounded at `1.0`.
        - **Checkpointing**: Tracks best validation accuracy and saves `model.state_dict()` to `cfg.ckpt_path`.
        - **Early Stopping**: Halts execution if validation accuracy does not improve for `cfg.early_stop_patience` epochs.

    Args:
        train_loader: Training DataLoader yielding `(images, labels)`.
        val_loader: Validation DataLoader yielding `(images, labels)`.
        config: Optional configuration instance overriding default hyperparameters.
        logger: Optional logger instance for diagnostic output.

    Returns:
        The trained `DualHeadCNN` model with state dict restored from the best epoch.

    Raises:
        ImportError: If PyTorch is unavailable.
        ModelCheckpointError: If saving the model weights fails.
        TrainingError: If unhandled exceptions disrupt the training loop.
    """
    if torch is None:
        raise ImportError("PyTorch is required for model training.")

    log: logging.Logger = logger if logger is not None else get_logger("solution")
    cfg: Config = config if config is not None else Config()

    device: Any = cfg.device
    model: DualHeadCNN = DualHeadCNN().to(device)

    # Configure AdamW optimizer and Cosine Annealing learning rate schedule
    opt: torch.optim.AdamW = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    sch: torch.optim.lr_scheduler.CosineAnnealingLR = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs, eta_min=1e-6
    )

    best_acc: float = -1.0
    best_state: Optional[dict] = None
    patience: int = 0

    try:
        for ep in range(cfg.epochs):
            model.train()
            total_loss: float = 0.0
            num_batches: int = 0

            # Mini-batch optimization loop
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()

                lc, ls = model(x)
                loss: torch.Tensor = dual_loss(lc, ls, y)
                loss.backward()

                # Gradient clipping to prevent gradient explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                total_loss += loss.item()
                num_batches += 1

            # Update learning rate schedule
            sch.step()

            # Evaluate on held-out validation set
            val_acc: float = validate(model, val_loader, device=device)
            avg_loss: float = total_loss / max(num_batches, 1)
            log.info(f"episode {ep+1}/{cfg.epochs}  loss={avg_loss:.4f}  val={val_acc:.4f}")

            # Checkpoint tracking & Early stopping
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                try:
                    torch.save(best_state, str(cfg.ckpt_path))
                except Exception as save_err:
                    raise ModelCheckpointError(
                        f"Failed to save checkpoint to '{cfg.ckpt_path}': {save_err}"
                    ) from save_err
                patience = 0
            else:
                patience += 1
                if patience >= cfg.early_stop_patience:
                    log.info(f"early stop @ ep {ep+1}")
                    break

        # Reload best model checkpoint
        if best_state is not None:
            model.load_state_dict(best_state)
        log.info(f"best val_acc={best_acc:.4f}")
        return model

    except (TrainingError, ModelCheckpointError):
        raise
    except Exception as exc:
        raise TrainingError(f"Training loop failed with error: {exc}") from exc
