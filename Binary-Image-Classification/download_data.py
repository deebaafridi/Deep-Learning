#!/usr/bin/env python3
"""Dataset Download CLI for IITH Deep Learning 2026 Hackathon Kaggle Competition.

Automates authenticating and downloading competition assets directly from Kaggle.
Extracts the raw dataset into the target directory, preserving the required
`train/` and `test/` folder hierarchy.

Competition Details:
    - Title: IITH Deep Learning 2026 Hackathon
    - URL: https://www.kaggle.com/competitions/iith-deep-learning-2026-hackathon
    - Identifier: iith-deep-learning-2026-hackathon

Authentication Setup:
    Option 1: API Token JSON file
        Place your Kaggle API key at `~/.kaggle/kaggle.json`:
        $ chmod 600 ~/.kaggle/kaggle.json

    Option 2: Environment Variables
        $ export KAGGLE_USERNAME="your-username"
        $ export KAGGLE_KEY="your-api-key"

CLI Examples:
    # Standard download into default ./data folder:
    $ python download_data.py

    # Download into a custom directory without auto-unzipping:
    $ python download_data.py --output-dir ./raw_dataset --no-unzip
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.data.download import COMPETITION_ID, COMPETITION_URL, download_kaggle_dataset
from src.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for Kaggle dataset download.

    Returns:
        argparse.Namespace populated with parsed options:
            competition (str): Kaggle competition slug identifier.
            output_dir (str): Target directory for downloaded files.
            no_unzip (bool): If True, suppresses automatic unzipping of archive files.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=f"Download and unpack dataset from Kaggle ({COMPETITION_URL})",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--competition",
        "-c",
        type=str,
        default=COMPETITION_ID,
        help="Kaggle competition slug identifier",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./data",
        help="Destination directory for downloaded and extracted files",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Skip automatic unzipping of downloaded archives",
    )
    return parser.parse_args()


def main() -> None:
    """Main CLI execution routine for Kaggle dataset downloader."""
    args: argparse.Namespace = parse_args()
    logger: logging.Logger = get_logger("downloader")

    logger.info(f"Target Kaggle Competition: {args.competition}")
    logger.info(f"Official Competition Link:  {COMPETITION_URL}")
    logger.info(f"Destination Directory:      {args.output_dir}")

    # Invoke download utility
    download_kaggle_dataset(
        competition=args.competition,
        output_dir=Path(args.output_dir),
        unzip=not args.no_unzip,
        logger=logger,
    )


if __name__ == "__main__":
    main()

