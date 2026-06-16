"""
FastAPI deepfake detection endpoint.

Run:
    uvicorn src.inference.api:app --reload

Endpoints:
    GET  /health       — liveness probe
    GET  /methods      — list all available method keys
    POST /predict      — classify an image (multipart form: file + method)
"""

import os
import shutil
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.inference.predict import (
    VALID_METHODS,
    DeepfakePredictor,
    ModelNotTrainedError,
)

# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

_predictor: DeepfakePredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    _predictor = DeepfakePredictor()
    yield
    _predictor = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Deepfake Detection API",
    description=(
        "Classify images as real or fake using one of 12 detection methods "
        "spanning classical ML (LBP/GLCM/FFT + SVM/MLP), fine-tuned CNNs "
        "(ResNet50, InceptionV3, EfficientNetB0), and a dual-channel "
        "spatial+frequency detector."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

METHOD_ACCURACY = {
    "lbp_svm": 65.11,
    "fft_svm": 68.34,
    "combined_svm": 74.87,
    "glcm_mlp": 66.89,
    "lbp_mlp": 79.93,
    "fft_mlp": 62.32,
    "combined_mlp": 82.33,
    "resnet50": 71.80,
    "inceptionv3": 87.63,
    "efficientnet": 94.41,
    "dual_channel_inception": 98.92,
    "dual_channel_resnet": 98.92,
}


class PredictionResponse(BaseModel):
    method: str = Field(..., description="Method key used for classification")
    label: str = Field(..., description='"real" or "fake"')
    prediction: int = Field(..., description="0 = real, 1 = fake")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model confidence in predicted label (0–1)"
    )
    reported_accuracy: float = Field(
        ..., description="Test-set accuracy reported in the experiment log (%)"
    )


class MethodsResponse(BaseModel):
    methods: list[str]


class HealthResponse(BaseModel):
    status: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Utility"])
def health():
    return {"status": "ok"}


@app.get("/methods", response_model=MethodsResponse, tags=["Utility"])
def list_methods():
    """Return all supported method keys."""
    return {"methods": VALID_METHODS}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, etc.)"),
    method: str = Form(
        ...,
        description=(
            "Detection method. One of: "
            + ", ".join(VALID_METHODS)
        ),
    ),
):
    """
    Classify an uploaded image as real or fake.

    - **file**: image to classify (JPEG / PNG recommended)
    - **method**: which detection pipeline to use
    """
    if method not in VALID_METHODS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown method '{method}'. Valid methods: {VALID_METHODS}",
        )

    # Save upload to a temp file so cv2 / PIL can read it by path
    suffix = os.path.splitext(file.filename or ".jpg")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = _predictor.predict(tmp_path, method)
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
    finally:
        os.unlink(tmp_path)

    return PredictionResponse(
        method=method,
        reported_accuracy=METHOD_ACCURACY[method],
        **result,
    )
