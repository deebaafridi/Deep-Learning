"""DualCNN package: A production-ready dual-head convolutional network pipeline."""

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
from src.data.download import COMPETITION_ID, COMPETITION_URL, download_kaggle_dataset
from src.data.validator import DatasetValidator, ValidationReport
from src.exceptions.exceptions import (
    ConfigurationError,
    DataNotFoundError,
    DataValidationError,
    DualCNNException,
    ImageProcessingError,
    InvalidDataSplitError,
    ModelCheckpointError,
    TrainingError,
)
from src.utils.logger import get_logger

# Import data and deep learning modules (guarded for environments where heavy dependencies are not yet installed)
try:
    from src.data.split import build_split, data_split, list_images
    from src.data.dataset import ImageRecord, RGBDataset, TestPathDataset
    from src.data.transforms import _hsv_jitter, augment_train, load_image
    from src.engine.inference import generate_predictions, predict_tta
    from src.engine.trainer import _val, train, validate
    from src.losses.loss import DualLoss, dual_loss
    from src.models.dual_head_cnn import DualHeadCNN
    from src.models.layers import AddCoords, ConvBlock
    from src.utils.seed import seed_everything
except ImportError:
    pass

__version__ = "1.0.0"

__all__ = [
    "Config",
    "ALLOWED_EXTENSIONS",
    "INPUT_SIZE",
    "VAL_PER_CLASS",
    "BATCH_SIZE",
    "EPOCHS",
    "EARLY_STOP_PATIENCE",
    "LR",
    "WEIGHT_DECAY",
    "NUM_WORKERS",
    "RANDOM_SEED",
    "CKPT",
    "DEVICE",
    "download_kaggle_dataset",
    "COMPETITION_ID",
    "COMPETITION_URL",
    "DatasetValidator",
    "ValidationReport",
    "list_images",
    "build_split",
    "data_split",
    "get_logger",
    "DualCNNException",
    "DataNotFoundError",
    "DataValidationError",
    "InvalidDataSplitError",
    "ImageProcessingError",
    "ModelCheckpointError",
    "ConfigurationError",
    "TrainingError",
]
