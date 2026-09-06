"""Image loading, Sobel edge filtering, and data augmentation routines for DualCNN.

Contains:
1. `_hsv_jitter`: Modulates hue, saturation, and value channels.
2. `augment_train`: Applies photometric jitter, horizontal flips, and rotations.
3. `load_image`: Resizes, performs ITU-R 601-2 grayscale conversion, computes Sobel
   horizontal/vertical gradient magnitudes, max-normalizes, and stacks into 3 channels.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Union

import numpy as np

try:
    from PIL import Image
    from scipy import ndimage
except ImportError:
    Image = None
    ndimage = None

try:
    import torch
    import torchvision.transforms.functional as TF
except ImportError:
    torch = None
    TF = None

from src.config.config import INPUT_SIZE
from src.exceptions.exceptions import ImageProcessingError


def _hsv_jitter(img: np.ndarray, hue: float, sat: float, val: float) -> np.ndarray:
    """Applies HSV colour jitter augmentation to an RGB image normalized in [0, 1].

    Converts the normalized RGB float array to uint8, transforms colour space to
    HSV using PIL, adds hue shifts with modulo wrapping, scales saturation and value,
    and converts back to normalized RGB float32.

    Args:
        img: Input image as float32 NumPy array of shape `(H, W, 3)` with values in [0.0, 1.0].
        hue: Fractional hue offset to add in range [-0.5, 0.5] (wrapped modulo 1.0).
        sat: Saturation scaling factor (clamped to [0.0, 1.0]).
        val: Value / brightness scaling factor (clamped to [0.0, 1.0]).

    Returns:
        Augmented RGB float32 image array of shape `(H, W, 3)` with values in [0.0, 1.0].

    Raises:
        ImageProcessingError: If PIL is missing or colour conversion fails.
    """
    if Image is None:
        raise ImageProcessingError("Pillow (PIL) is required for HSV jitter augmentation.")
    try:
        pil: Image.Image = Image.fromarray((img * 255.0).astype(np.uint8))
        hsv: np.ndarray = np.array(pil.convert("HSV"), dtype=np.float32) / 255.0

        # Apply jitter with boundary constraints
        hsv[..., 0] = (hsv[..., 0] + hue) % 1.0
        hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0.0, 1.0)
        hsv[..., 2] = np.clip(hsv[..., 2] * val, 0.0, 1.0)

        pil2: Image.Image = Image.fromarray((hsv * 255.0).astype(np.uint8), mode="HSV")
        return np.array(pil2.convert("RGB"), dtype=np.float32) / 255.0
    except Exception as exc:
        raise ImageProcessingError(f"Error during HSV jitter augmentation: {exc}") from exc


def augment_train(img: np.ndarray) -> np.ndarray:
    """Applies stochastic photometric and geometric augmentations to a training image.

    Transformations:
        1. Photometric jitter (applied with 70% probability):
           - Brightness: random factor in [0.4, 1.8]
           - Contrast: random factor in [0.4, 2.0]
           - Saturation: random factor in [0.0, 2.0]
           - Hue: random factor in [-0.5, 0.5]
        2. Horizontal flip (applied with 50% probability).
        3. Random rotation: angle sampled uniformly from [-15.0, +15.0] degrees.

    Args:
        img: Input image as float32 NumPy array of shape `(H, W, 3)`.

    Returns:
        Transformed image as float32 NumPy array of shape `(H, W, 3)`.

    Raises:
        ImportError: If PyTorch or torchvision are unavailable.
        ImageProcessingError: If transformation execution fails.
    """
    if torch is None or TF is None:
        raise ImportError("PyTorch and torchvision are required to run augment_train.")
    try:
        # Convert HWC numpy array to CHW PyTorch tensor
        img_t: torch.Tensor = torch.from_numpy(img).permute(2, 0, 1)

        # 1. Photometric transformations (70% probability)
        if random.random() > 0.3:
            img_t = TF.adjust_brightness(img_t, brightness_factor=random.uniform(0.4, 1.8))
            img_t = TF.adjust_contrast(img_t, contrast_factor=random.uniform(0.4, 2.0))
            img_t = TF.adjust_saturation(img_t, saturation_factor=random.uniform(0.0, 2.0))
            img_t = TF.adjust_hue(img_t, hue_factor=random.uniform(-0.5, 0.5))

        # 2. Geometric horizontal flip (50% probability)
        if random.random() < 0.5:
            img_t = TF.hflip(img_t)

        # 3. Geometric rotation (-15 to +15 degrees)
        angle: float = random.uniform(-15.0, 15.0)
        img_t = TF.rotate(img_t, angle=angle)

        # Permute back to HWC numpy array
        return img_t.permute(1, 2, 0).numpy()
    except Exception as exc:
        raise ImageProcessingError(f"Error during image augmentation: {exc}") from exc


def load_image(path: Union[str, Path], input_size: int = INPUT_SIZE) -> np.ndarray:
    """Loads an image, applies Sobel edge filtering, and returns a normalized 3-channel array.

    Processing Pipeline:
        1. Open image from disk via PIL and convert to RGB.
        2. Resize to square spatial resolution `(input_size, input_size)`.
        3. Normalize pixel intensities into `[0.0, 1.0]` as float32.
        4. Convert RGB to grayscale using standard ITU-R 601-2 luma weighting:
           `Y = 0.2989*R + 0.5870*G + 0.1140*B`
        5. Apply horizontal ($S_x$) and vertical ($S_y$) Sobel differential operators.
        6. Compute Euclidean gradient magnitude: `mag = sqrt(S_x^2 + S_y^2)`.
        7. Max-normalize magnitude: `mag = mag / (max(mag) + 1e-6)`.
        8. Stack the 2D edge map into 3 identical channels to match standard CNN input shapes.

    Args:
        path: Filesystem path to the target image file.
        input_size: Height and width in pixels to resize the image (default: 96).

    Returns:
        A float32 NumPy array of shape `(input_size, input_size, 3)` with values in [0.0, 1.0].

    Raises:
        ImageProcessingError: If image loading, decoding, resizing, or Sobel filtering fails.
    """
    if Image is None or ndimage is None:
        raise ImageProcessingError("Pillow (PIL) and SciPy are required to load and process images.")
    try:
        path_obj: Path = Path(path)
        with Image.open(path_obj) as im:
            # Resize and normalize RGB to [0.0, 1.0]
            img: np.ndarray = np.array(
                im.convert("RGB").resize((input_size, input_size)), dtype=np.float32
            ) / 255.0

            # Convert to Grayscale using standard ITU-R 601-2 luma weights
            gray: np.ndarray = np.dot(img, [0.2989, 0.5870, 0.1140])

            # Sobel gradient computation in X and Y spatial directions
            sx: np.ndarray = ndimage.sobel(gray, axis=0)
            sy: np.ndarray = ndimage.sobel(gray, axis=1)
            mag: np.ndarray = np.hypot(sx, sy)

            # Max-normalize gradient magnitude
            mag = mag / (mag.max() + 1e-6)

            # Replicate magnitude across 3 channels to preserve network input shape
            stacked: np.ndarray = np.stack([mag, mag, mag], axis=-1)
            return stacked.astype(np.float32)
    except ImageProcessingError:
        raise
    except Exception as exc:
        raise ImageProcessingError(f"Failed to load or preprocess image at '{path}': {exc}") from exc
