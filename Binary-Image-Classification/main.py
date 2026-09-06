#!/usr/bin/env python3
"""CLI entrypoint for DualCNN model training and inference.

Supports both interactive prompt (stdin) and modern command-line argument flags.

Examples:
    # CLI execution with custom hyperparameters
    $ python main.py --data-dir ./data --epochs 25 --batch-size 96 --lr 3e-4

    # Interactive execution (prompts for dataset path)
    $ python main.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    import torch
except ImportError:
    torch = None

try:
    from src.engine.inference import generate_predictions
except ImportError:
    generate_predictions = None  # type: ignore[assignment]

from src.config.config import Config
from src.utils.logger import get_logger
from src.utils.seed import seed_everything



def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for training, validation, and inference.

    Returns:
        argparse.Namespace populated with parsed CLI flags.
    """
    default_device: str = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Dual-Head CNN for Kaggle 3D Multi-Object Binary Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default=None,
        help="Path to root data directory containing 'train/' and 'test/' subdirectories",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=96,
        help="Batch size for training and inference DataLoaders",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Initial learning rate for AdamW optimizer",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay L2 regularization factor",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="ckpt_dual.pth",
        help="Path to save or load PyTorch model checkpoint weights",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        help="Compute device target ('cuda' or 'cpu')",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution function resolving data paths, configuration, and pipeline invocation."""
    args: argparse.Namespace = parse_args()
    logger = get_logger("solution")

    # Resolve data directory: CLI flag takes precedence, then stdin, fallback to current dir
    data_dir_str: Optional[str] = args.data_dir
    if not data_dir_str:
        if not sys.stdin.isatty():
            try:
                data_dir_str = sys.stdin.readline().strip()
            except Exception:
                data_dir_str = "."
        else:
            try:
                data_dir_str = input("Enter dataset directory path: ").strip()
            except (EOFError, KeyboardInterrupt):
                data_dir_str = "."

    if not data_dir_str:
        data_dir_str = "."

    target_device = (
        torch.device(args.device)
        if torch is not None
        else args.device
    )

    # Initialize configuration dataclass
    config: Config = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        ckpt_path=args.ckpt,
        random_seed=args.seed,
        device=target_device,
    )

    # Set deterministic random seeds
    seed_everything(config.random_seed)
    if generate_predictions is None:
        logger.error(
            "Deep learning runtime dependencies (PyTorch, Torchvision, Scipy) are not installed. "
            "Please install dependencies using: pip install -r requirements.txt"
        )
        sys.exit(1)

    # Launch prediction pipeline
    generate_predictions(data_dir_str, config=config)



if __name__ == "__main__":
    main()
