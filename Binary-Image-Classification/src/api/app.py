"""FastAPI web application and REST endpoints for serving DualCNN predictions.

Features:
- Lifespan model loader restoring `ckpt_dual.pth` on application startup.
- In-memory Sobel gradient image preprocessing from uploaded byte streams.
- Interactive OpenAPI / Swagger UI documentation at `/docs`.
- Endpoints:
  - `GET /health`: Model status, active hardware device, version.
  - `POST /predict`: Real-time single image upload classification.
  - `POST /predict/batch`: Multi-image batch classification.
"""

from __future__ import annotations

import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import numpy as np

try:
    from PIL import Image
    from scipy import ndimage
except ImportError:
    Image = None
    ndimage = None

try:
    import torch
except ImportError:
    torch = None

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, RedirectResponse
    FASTAPI_AVAILABLE: bool = True
except ImportError:
    FastAPI = None
    FASTAPI_AVAILABLE = False

from src.api.schemas import BatchPredictionResponse, HealthResponse, PredictionResponse
from src.config.config import Config, INPUT_SIZE
from src.exceptions.exceptions import ImageProcessingError
from src.utils.logger import get_logger

logger: logging.Logger = get_logger("solution")

# Global runtime state dictionary storing model instance, device, and configuration
app_state: Dict[str, Any] = {
    "model": None,
    "device": "cpu",
    "model_loaded": False,
    "config": Config(),
}


def preprocess_image_bytes(image_bytes: bytes, input_size: int = INPUT_SIZE) -> np.ndarray:
    """Decodes image bytes in-memory, resizes, and applies Sobel gradient edge filtering.

    Pipeline:
        1. Parse bytes using PIL into an in-memory RGB image.
        2. Resize to square spatial dimension `(input_size, input_size)`.
        3. Convert RGB to grayscale using standard ITU-R 601-2 luma weights.
        4. Compute horizontal and vertical Sobel gradient magnitudes via SciPy.
        5. Max-normalize magnitude to [0.0, 1.0] and stack into 3 identical channels.

    Args:
        image_bytes: Raw binary image payload (PNG, JPEG, etc.).
        input_size: Height and width in pixels to resize (default: 96).

    Returns:
        A float32 NumPy array of shape `(input_size, input_size, 3)` with values in [0.0, 1.0].

    Raises:
        ImageProcessingError: If decoding or Sobel filtering fails.
    """
    if Image is None or ndimage is None:
        raise ImageProcessingError("Pillow and SciPy are required to preprocess image bytes.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            # Resize and normalize RGB to [0.0, 1.0]
            img: np.ndarray = np.array(
                im.convert("RGB").resize((input_size, input_size)), dtype=np.float32
            ) / 255.0

            # Grayscale conversion via ITU-R 601-2 weights: Y = 0.2989*R + 0.5870*G + 0.1140*B
            gray: np.ndarray = np.dot(img, [0.2989, 0.5870, 0.1140])

            # Spatial horizontal and vertical Sobel gradient computation
            sx: np.ndarray = ndimage.sobel(gray, axis=0)
            sy: np.ndarray = ndimage.sobel(gray, axis=1)
            mag: np.ndarray = np.hypot(sx, sy)

            # Max normalization
            mag = mag / (mag.max() + 1e-6)

            # Stack into 3-channel input tensor format
            return np.stack([mag, mag, mag], axis=-1).astype(np.float32)
    except Exception as exc:
        raise ImageProcessingError(f"Could not decode or filter image bytes: {exc}") from exc


@asynccontextmanager
async def lifespan(app: Any) -> AsyncIterator[None]:
    """Lifespan context manager: initializes DualHeadCNN model and loads weights on startup.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control to the application lifecycle until shutdown.
    """
    cfg: Config = app_state["config"]
    if torch is not None:
        try:
            from src.models.dual_head_cnn import DualHeadCNN

            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            app_state["device"] = str(device)
            model: DualHeadCNN = DualHeadCNN().to(device)

            ckpt_file: Path = Path(cfg.ckpt_path)
            if ckpt_file.exists():
                logger.info(f"Loading checkpoint weights from '{ckpt_file}'...")
                state_dict: dict = torch.load(str(ckpt_file), map_location=device)
                model.load_state_dict(state_dict)
                app_state["model_loaded"] = True
                logger.info("Model weights loaded successfully into memory.")
            else:
                logger.warning(
                    f"Checkpoint '{ckpt_file}' not found. Model will serve with uninitialized weights."
                )
                app_state["model_loaded"] = False

            model.eval()
            app_state["model"] = model
        except Exception as exc:
            logger.error(f"Failed to initialize model during startup: {exc}")
    else:
        logger.warning("PyTorch not installed. Serving API in standby mode.")

    yield
    logger.info("Shutting down DualCNN inference API.")


def create_app() -> Any:
    """Application factory creating and configuring the FastAPI instance.

    Returns:
        Configured `fastapi.FastAPI` application instance.

    Raises:
        ImportError: If FastAPI is not installed in the environment.
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is required to run the web service. Install with: pip install fastapi uvicorn"
        )

    api = FastAPI(
        title="Dual-Head CNN Inference API",
        description=(
            "Production REST API serving the Dual-Head CNN model for 3D multi-object binary "
            "classification from the IITH Deep Learning 2026 Hackathon."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # Enable Cross-Origin Resource Sharing (CORS)
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        """Redirects root path to interactive Swagger documentation."""
        return RedirectResponse(url="/docs")

    @api.get(
        "/health",
        response_model=HealthResponse,
        summary="Service Health Check",
        tags=["Monitoring"],
    )
    def health() -> HealthResponse:
        """Returns the operational status, hardware compute device, and model initialization state."""
        return HealthResponse(
            status="healthy",
            model_loaded=app_state["model_loaded"],
            device=app_state["device"],
            version="1.0.0",
        )

    @api.post(
        "/predict",
        response_model=PredictionResponse,
        summary="Predict Single Image",
        tags=["Inference"],
    )
    async def predict_single(file: UploadFile = File(...)) -> PredictionResponse:
        """Classifies an uploaded image file into binary Class 0 or Class 1.

        Args:
            file: Multi-part uploaded image file.

        Returns:
            PredictionResponse containing cube probability, sphere probability,
            combined joint probability, binary label, and latency in milliseconds.
        """
        model = app_state["model"]
        if model is None or torch is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not initialized or PyTorch is unavailable in this environment.",
            )

        start_time: float = time.perf_counter()
        try:
            content: bytes = await file.read()
            processed_np: np.ndarray = preprocess_image_bytes(content, input_size=INPUT_SIZE)
        except ImageProcessingError as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image file: {err}",
            )

        # Convert HWC numpy array to BCHW PyTorch tensor on target device
        device: torch.device = torch.device(app_state["device"])
        tensor_input: torch.Tensor = (
            torch.from_numpy(processed_np.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        )

        with torch.no_grad():
            logit_cube, logit_sphere = model(tensor_input)
            p_cube: float = float(torch.sigmoid(logit_cube).cpu().item())
            p_sphere: float = float(torch.sigmoid(logit_sphere).cpu().item())

        p_both: float = p_cube * p_sphere
        pred_label: int = 1 if p_both > 0.5 else 0
        elapsed_ms: float = (time.perf_counter() - start_time) * 1000.0

        return PredictionResponse(
            filename=file.filename or "unknown.jpg",
            cube_probability=round(p_cube, 4),
            sphere_probability=round(p_sphere, 4),
            joint_probability=round(p_both, 4),
            prediction=pred_label,
            class_name="Positive (Contains Cube and Sphere)" if pred_label == 1 else "Negative",
            inference_time_ms=round(elapsed_ms, 2),
        )

    @api.post(
        "/predict/batch",
        response_model=BatchPredictionResponse,
        summary="Predict Batch of Images",
        tags=["Inference"],
    )
    async def predict_batch(files: List[UploadFile] = File(...)) -> BatchPredictionResponse:
        """Classifies a list of uploaded images concurrently and aggregates results.

        Args:
            files: List of uploaded image files.

        Returns:
            BatchPredictionResponse with list of individual predictions and counts.
        """
        results: List[PredictionResponse] = []
        for file in files:
            res: PredictionResponse = await predict_single(file)
            results.append(res)

        positives: int = sum(1 for r in results if r.prediction == 1)
        negatives: int = len(results) - positives

        return BatchPredictionResponse(
            total_images=len(results),
            positive_count=positives,
            negative_count=negatives,
            predictions=results,
        )

    return api


# Expose default app instance if FastAPI is installed
if FASTAPI_AVAILABLE:
    app = create_app()
else:
    app = None
