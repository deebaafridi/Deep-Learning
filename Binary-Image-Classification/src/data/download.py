"""Kaggle dataset download and extraction utilities for DualCNN.

Facilitates automated acquisition of the IITH Deep Learning 2026 Hackathon competition
dataset via the Kaggle Python API.
"""

from __future__ import annotations

import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional, Union

from src.exceptions.exceptions import DataNotFoundError, DualCNNException
from src.utils.logger import get_logger

COMPETITION_ID: str = "iith-deep-learning-2026-hackathon"
"""Official Kaggle competition slug identifier."""

COMPETITION_URL: str = "https://www.kaggle.com/competitions/iith-deep-learning-2026-hackathon"
"""Direct hyperlink to competition website and leaderboard."""


def download_kaggle_dataset(
    competition: str = COMPETITION_ID,
    output_dir: Union[str, Path] = "./data",
    unzip: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Authenticates with the Kaggle API and downloads competition dataset files.

    Authentication Requirements:
        Must provide credentials via either:
        1. Local token file located at `~/.kaggle/kaggle.json`.
        2. Environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`.

    Args:
        competition: Kaggle competition slug identifier (default: 'iith-deep-learning-2026-hackathon').
        output_dir: Destination folder path for downloaded assets (default: './data').
        unzip: Boolean flag indicating whether to extract downloaded zip archives (default: True).
        logger: Optional logger instance for diagnostic output.

    Returns:
        A resolved `pathlib.Path` pointing to the directory containing dataset files.

    Raises:
        DualCNNException: If `kaggle` package is missing or competition download fails.
        DataNotFoundError: If Kaggle credentials cannot be located on the system.

    Example:
        >>> download_kaggle_dataset(output_dir="./data")
    """
    log: logging.Logger = logger if logger is not None else get_logger("solution")
    out_path: Path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Verify Kaggle API library is installed
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        msg: str = (
            f"The 'kaggle' library is required to download data automatically.\n"
            f"Please install it with: pip install kaggle\n"
            f"Alternatively, manually download the dataset from:\n"
            f"{COMPETITION_URL}"
        )
        log.error(msg)
        raise DualCNNException(msg) from exc

    # 2. Check for existence of authentication credentials
    kaggle_json: Path = Path.home() / ".kaggle" / "kaggle.json"
    has_env: bool = "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ

    if not kaggle_json.exists() and not has_env:
        msg = (
            f"Kaggle credentials not detected!\n"
            f"Please place your 'kaggle.json' token into {kaggle_json} or set "
            f"KAGGLE_USERNAME and KAGGLE_KEY environment variables.\n"
            f"You can generate a token at: https://www.kaggle.com/settings -> 'Create New Token'.\n"
            f"Or download directly from: {COMPETITION_URL}"
        )
        log.error(msg)
        raise DataNotFoundError(msg)

    # 3. Authenticate and download archive
    log.info(f"Authenticating with Kaggle API...")
    try:
        api = KaggleApi()
        api.authenticate()
        log.info(f"Downloading dataset for competition '{competition}' to {out_path}...")
        api.competition_download_files(competition, path=str(out_path), quiet=False)
    except Exception as exc:
        msg = (
            f"Failed to download competition files for '{competition}': {exc}\n"
            f"Please verify you have accepted the competition rules at:\n"
            f"{COMPETITION_URL}"
        )
        log.error(msg)
        raise DualCNNException(msg) from exc

    # 4. Decompress downloaded zip archives
    zip_files: List[Path] = list(out_path.glob("*.zip"))
    if unzip and zip_files:
        for zf in zip_files:
            log.info(f"Extracting {zf.name} into {out_path}...")
            with zipfile.ZipFile(zf, "r") as zip_ref:
                zip_ref.extractall(out_path)
            log.info(f"Extraction complete for {zf.name}.")

    log.info(f"Dataset successfully prepared in {out_path}")
    return out_path
