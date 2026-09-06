"""Unit tests for Kaggle competition dataset download client.

Verifies competition URL constants, slug identifiers, credential validation
mechanisms, and error handling when Kaggle credentials are missing.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.data.download import COMPETITION_ID, COMPETITION_URL, download_kaggle_dataset
from src.exceptions.exceptions import DataNotFoundError, DualCNNException


class TestDownloader(unittest.TestCase):
    """Test suite validating Kaggle competition downloader configurations and authentication logic."""

    def test_competition_constants(self) -> None:
        """Verifies that the competition slug and Kaggle URL match official competition targets."""
        self.assertEqual(COMPETITION_ID, "iith-deep-learning-2026-hackathon")
        self.assertIn("iith-deep-learning-2026-hackathon", COMPETITION_URL)
        self.assertTrue(COMPETITION_URL.startswith("https://www.kaggle.com/competitions/"))

    @patch("pathlib.Path.home")
    def test_missing_credentials_raises_error(self, mock_home: MagicMock) -> None:
        """Verifies DataNotFoundError is raised when neither ~/.kaggle/kaggle.json nor env vars exist."""
        mock_home.return_value = Path("/nonexistent/home")

        # Clear environment variables for Kaggle auth
        with patch.dict(os.environ, {}, clear=True):
            # Mock kaggle python module so the test passes regardless of kaggle package installation
            with patch.dict("sys.modules", {"kaggle": MagicMock(), "kaggle.api.kaggle_api_extended": MagicMock()}):
                with self.assertRaises(DataNotFoundError) as ctx:
                    download_kaggle_dataset(output_dir="/tmp/test_download")

                self.assertIn("Kaggle credentials not detected", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

