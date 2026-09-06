"""Unit tests for pipeline configuration and hyperparameter management.

Verifies default hyperparameter values, custom keyword argument overrides,
type conversions, and filesystem path resolution for model checkpoints.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.config.config import (
    ALLOWED_EXTENSIONS,
    BATCH_SIZE,
    CKPT,
    DEVICE,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    INPUT_SIZE,
    LR,
    NUM_WORKERS,
    RANDOM_SEED,
    VAL_PER_CLASS,
    WEIGHT_DECAY,
    Config,
)


class TestConfig(unittest.TestCase):
    """Test suite for validating the `Config` dataclass and global constants."""

    def test_default_config_values(self) -> None:
        """Verifies that default configuration values align exactly with benchmark specifications."""
        cfg: Config = Config()

        # Image spatial dimensions and batching
        self.assertEqual(cfg.input_size, 96, "Default input image spatial size must be 96x96")
        self.assertEqual(cfg.batch_size, 96, "Default mini-batch size must be 96")
        self.assertEqual(cfg.val_per_class, 500, "Validation set per class must be 500 samples")

        # Optimization and training horizon
        self.assertEqual(cfg.epochs, 25, "Default training epochs must be 25")
        self.assertEqual(cfg.early_stop_patience, 5, "Early stopping patience must default to 5")
        self.assertAlmostEqual(cfg.learning_rate, 3e-4, places=6, msg="Learning rate must be 3e-4")
        self.assertAlmostEqual(cfg.weight_decay, 1e-4, places=6, msg="Weight decay must be 1e-4")

        # Determinism and checkpointing
        self.assertEqual(cfg.random_seed, 42, "Seed must default to 42 for exact determinism")
        self.assertEqual(cfg.ckpt_path, "ckpt_dual.pth", "Default checkpoint filename must match")

    def test_custom_config_override(self) -> None:
        """Verifies that constructor parameters correctly override default hyperparameter fields."""
        cfg: Config = Config(
            epochs=10,
            batch_size=32,
            learning_rate=1e-3,
            ckpt_path="custom_weights.pth",
        )

        self.assertEqual(cfg.epochs, 10)
        self.assertEqual(cfg.batch_size, 32)
        self.assertAlmostEqual(cfg.learning_rate, 1e-3)
        self.assertEqual(cfg.ckpt, Path("custom_weights.pth"))

    def test_ckpt_path_property(self) -> None:
        """Verifies that the `ckpt` property returns a Path object resolved from `ckpt_path`."""
        cfg: Config = Config(ckpt_path="weights/model.pt")
        self.assertIsInstance(cfg.ckpt, Path)
        self.assertEqual(str(cfg.ckpt), "weights/model.pt")


if __name__ == "__main__":
    unittest.main()

