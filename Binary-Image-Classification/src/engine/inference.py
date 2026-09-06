"""Inference engine and Test-Time Augmentation (TTA) pipeline for DualCNN.

Contains:
1. `predict_tta`: Multi-pass geometric Test-Time Augmentation (TTA) predicting
   independent cube and sphere probabilities and computing composite joint probabilities.
2. `generate_predictions`: End-to-end orchestration pipeline linking dataset discovery,
   checkpoint loading (or automated training if missing), TTA inference, and CSV output.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import numpy as np
try:
    from scipy import ndimage
except ImportError:
    ndimage = None  # type: ignore[assignment]

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]


from src.config.config import Config, DEVICE
from src.data.dataset import RGBDataset, TestPathDataset
from src.data.split import build_split, data_split, list_images
from src.engine.trainer import train
from src.exceptions.exceptions import ModelCheckpointError
from src.models.dual_head_cnn import DualHeadCNN
from src.utils.logger import get_logger


@torch.no_grad()
def predict_tta(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[Any] = None,
) -> np.ndarray:
    """Performs 4-pass Test-Time Augmentation (TTA) and outputs combined predicted probabilities.

    The 4 TTA Passes:
        1. Original image tensor (Identity).
        2. Horizontal flip (`im[:, ::-1, :]`).
        3. Counter-clockwise rotation of +8 degrees.
        4. Clockwise rotation of -8 degrees.

    For each pass, predicted cube and sphere probabilities are extracted:
        `p_cube = sigmoid(logit_cube)`
        `p_sphere = sigmoid(logit_sphere)`
    The probabilities are averaged across the 4 passes, and the final joint probability
    is computed as:
        `p_final = p_cube_avg * p_sphere_avg`

    Args:
        model: Trained `DualHeadCNN` module in evaluation mode.
        loader: DataLoader for the test dataset yielding `(image_tensor, image_name)`.
        device: Hardware device target (default: `config.DEVICE`).

    Returns:
        1D NumPy float32 array of shape `(num_samples,)` containing combined probabilities in [0, 1].

    Raises:
        ImportError: If PyTorch is unavailable.
    """
    if torch is None:
        raise ImportError("PyTorch is required to run predict_tta.")

    target_device: Any = device if device is not None else DEVICE
    model.eval()

    num_samples: int = len(loader.dataset)
    p_cube_acc: np.ndarray = np.zeros(num_samples, dtype=np.float32)
    p_sphere_acc: np.ndarray = np.zeros(num_samples, dtype=np.float32)
    n_passes: int = 0

    def _execute_pass(transform_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> None:
        """Executes an inference pass over the loader with optional spatial transformation."""
        nonlocal n_passes
        idx: int = 0
        for batch_images, _ in loader:
            # Permute CHW tensor back to HWC numpy format for scipy transformations
            x_np: np.ndarray = batch_images.permute(0, 2, 3, 1).numpy()
            if transform_fn is not None:
                x_np = np.stack([transform_fn(img) for img in x_np])

            # Convert back to CHW float tensor on target device
            xa: torch.Tensor = (
                torch.from_numpy(x_np.transpose(0, 3, 1, 2)).float().to(target_device)
            )
            logit_cube, logit_sphere = model(xa)

            # Accumulate sigmoid probabilities
            batch_len: int = len(logit_cube)
            p_cube_acc[idx : idx + batch_len] += torch.sigmoid(logit_cube).cpu().numpy()
            p_sphere_acc[idx : idx + batch_len] += torch.sigmoid(logit_sphere).cpu().numpy()
            idx += batch_len

        n_passes += 1

    # Pass 1: Standard unaltered pass
    _execute_pass()

    # Pass 2: Horizontal mirror flip
    _execute_pass(lambda im: im[:, ::-1, :].copy())

    # Pass 3: Counter-clockwise rotation (+8 degrees)
    _execute_pass(
        lambda im: ndimage.rotate(
            im, 8, reshape=False, order=1, mode="constant", cval=0.5, axes=(0, 1)
        ).astype(np.float32)
    )

    # Pass 4: Clockwise rotation (-8 degrees)
    _execute_pass(
        lambda im: ndimage.rotate(
            im, -8, reshape=False, order=1, mode="constant", cval=0.5, axes=(0, 1)
        ).astype(np.float32)
    )

    # Compute arithmetic mean across the 4 passes
    p_cube_mean: np.ndarray = p_cube_acc / n_passes
    p_sphere_mean: np.ndarray = p_sphere_acc / n_passes

    # Combined composite probability: P(cube AND sphere)
    return p_cube_mean * p_sphere_mean


def generate_predictions(
    data_dir: Union[str, Path],
    config: Optional[Config] = None,
) -> Path:
    """Executes the complete training, inference, and submission generation pipeline.

    Workflow Steps:
        1. Resolves `test` directory path and indexes all test images.
        2. Checks if model checkpoint weights exist at `config.ckpt_path`:
           - If missing, builds stratified train/validation splits and trains model.
        3. Loads model weights onto the target compute device.
        4. Evaluates test set using 4-pass Test-Time Augmentation (`predict_tta`).
        5. Discretizes probabilities using 0.5 threshold (`y = 1 if p > 0.5 else 0`).
        6. Writes `submission.csv` to `data_dir` with headers `ID,Label`.

    Args:
        data_dir: Filesystem path to root data directory containing `train/` and `test/`.
        config: Optional configuration instance overriding defaults.

    Returns:
        A `pathlib.Path` pointing to the generated `submission.csv`.

    Raises:
        ImportError: If PyTorch is unavailable.
        DataNotFoundError: If test directory or images cannot be found.
        ModelCheckpointError: If checkpoint file loading fails.
    """
    if torch is None:
        raise ImportError("PyTorch is required to generate predictions.")

    cfg: Config = config if config is not None else Config()
    logger: logging.Logger = get_logger("solution", log_file=cfg.log_file)

    base_dir: Path = Path(data_dir)
    test_dir: Path = data_split(base_dir, "test")
    test_paths: List[Path] = list_images(test_dir, cfg.allowed_extensions)
    logger.info(f"Found {len(test_paths)} test images in {test_dir}")

    ckpt_path: Path = Path(cfg.ckpt_path)

    # Train model if checkpoint does not exist
    if not ckpt_path.exists():
        train_dir: Path = data_split(base_dir, "train")
        train_recs, val_recs = build_split(
            train_dir, val_per_class=cfg.val_per_class, random_seed=cfg.random_seed
        )
        logger.info(f"train={len(train_recs)}  val={len(val_recs)}  device={cfg.device}")

        train_loader: DataLoader = DataLoader(
            RGBDataset(train_recs, augment=True),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
        val_loader: DataLoader = DataLoader(
            RGBDataset(val_recs, augment=False),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        logger.info("=== Training Dual-Head CNN ===")
        train(train_loader, val_loader, config=cfg, logger=logger)

    # Instantiate model and load checkpoint weights
    model: DualHeadCNN = DualHeadCNN().to(cfg.device)
    try:
        state_dict: dict = torch.load(str(ckpt_path), map_location=cfg.device)
        model.load_state_dict(state_dict)
    except Exception as exc:
        raise ModelCheckpointError(f"Could not load checkpoint from '{ckpt_path}': {exc}") from exc

    # Prepare test DataLoader
    test_loader: DataLoader = DataLoader(
        TestPathDataset(test_paths),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda batch: (torch.stack([item[0] for item in batch]), [item[1] for item in batch]),
    )

    # Perform 4-pass TTA inference
    p_final: np.ndarray = predict_tta(model, test_loader, device=cfg.device)
    preds: np.ndarray = (p_final > 0.5).astype(np.int64)

    # Export to submission.csv
    out_path: Path = base_dir / cfg.submission_filename
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer: csv.writer = csv.writer(f)
        writer.writerow(["ID", "Label"])
        for path_item, pred_val in zip(test_paths, preds):
            writer.writerow([path_item.name, int(pred_val)])

    num_pos: int = int(preds.sum())
    num_neg: int = len(preds) - num_pos
    print(
        f"{len(preds)} predictions  # +ve predictions 1={num_pos}  # -ve predicitons 0={num_neg}"
    )
    logger.info(f"Predictions successfully written to {out_path}")
    return out_path
