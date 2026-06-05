# =============================================================================
# backend/main.py — FastAPI Backend
#
# Endpoints:
#   POST /predict        — single image upload → class + confidence
#   POST /batch_predict  — folder path → predictions + accuracy + loss
#   GET  /metrics        — return logs/metrics.json
#   GET  /comparison     — return logs/comparison.json
# =============================================================================

import os
import sys
import json
import shutil
import tempfile
import subprocess
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add parent directory to path so inference.py is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import inference  # noqa: E402

# ======================
# PATHS
# ======================
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR         = os.path.join(BASE_DIR, "logs")
METRICS_PATH    = os.path.join(LOG_DIR, "metrics.json")
COMPARISON_PATH = os.path.join(LOG_DIR, "comparison.json")
BASELINE_SCRIPT = os.path.join(BASE_DIR, "train_supervised_baseline.py")

# ======================
# BASELINE SUBPROCESS STATE
# ======================
_baseline_proc: Optional[subprocess.Popen] = None
_baseline_done: bool = False

# ======================
# APP SETUP
# ======================
app = FastAPI(
    title="Stain-Invariant SimCLR API",
    description="Histopathology (PCam) tumor/normal classification API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Update to specific frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve graphs as static files so index.html can load them
# NOTE: Must be mounted AFTER all API routes to avoid shadowing them
GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")
os.makedirs(GRAPHS_DIR, exist_ok=True)

# ======================
# PYDANTIC MODELS
# ======================
class BatchPredictRequest(BaseModel):
    folder: str


class PredictionResult(BaseModel):
    predict_class: str
    confidence: float


class BatchResult(BaseModel):
    predictions: list
    accuracy: float | None
    loss: float | None

# ======================
# ROUTES
# ======================

@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "message": "Stain-Invariant SimCLR API is running."}


@app.get("/health", summary="Health check")
def health():
    metrics_ok     = os.path.exists(METRICS_PATH)
    comparison_ok  = os.path.exists(COMPARISON_PATH)
    return {
        "status":        "healthy",
        "metrics_ready": metrics_ok,
        "comparison_ready": comparison_ok,
    }


@app.post("/predict", summary="Predict a single uploaded image")
async def predict(file: UploadFile = File(...)):
    """
    Upload an image file and get tumor/normal prediction.

    - **file**: Image file (JPEG, PNG, BMP, TIFF)

    Returns:
    - **class**: "Tumor" or "Normal"
    - **confidence**: float 0–1
    """
    # Validate extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Use JPEG, PNG, BMP, or TIFF."
        )

    # Save to temp file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        result = inference.predict_single(tmp_path)
        return JSONResponse({
            "class":      result["class"],
            "confidence": result["confidence"],
        })

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/batch_predict", summary="Predict all images in a server-side folder")
def batch_predict(request: BatchPredictRequest):
    """
    Predict all images in a local folder path.

    Body:
    - **folder**: Absolute path to image folder on server

    Optionally place a `labels.json` file in the folder for accuracy/loss calculation.

    Returns:
    - **predictions**: List of {file, class, confidence}
    - **accuracy**: If labels.json provided
    - **loss**: If labels.json provided
    """
    try:
        result = inference.predict_batch(request.folder)
        return JSONResponse(result)
    except NotADirectoryError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/gradcam", summary="Generate Grad-CAM saliency map for an uploaded image")
async def gradcam(file: UploadFile = File(...)):
    """
    Upload an image and receive a Grad-CAM saliency heatmap alongside the prediction.

    Returns:
    - **class**: "Tumor" or "Normal"
    - **confidence**: float 0–1
    - **saliency_map**: 96×96 nested list of floats (0=cold, 1=hot)
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        result = inference.generate_gradcam(tmp_path)
        return JSONResponse(result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/metrics", summary="Retrieve training metrics (SSL pre-training)")
def get_metrics():
    """
    Return the contents of `logs/metrics.json`.

    Schema:
    ```json
    {
      "epoch":          [1, 2, ...],
      "train_loss":     [...],
      "val_loss":       [...],
      "train_accuracy": [...],
      "val_accuracy":   [...],
      "epoch_time":     [...]
    }
    ```
    """
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(
            status_code=404,
            detail="metrics.json not found. Run train.py to generate it."
        )
    with open(METRICS_PATH, 'r') as f:
        data = json.load(f)
    return JSONResponse(data)


@app.get("/comparison", summary="Retrieve model comparison data")
def get_comparison():
    """
    Return the contents of `logs/comparison.json`.

    Schema:
    ```json
    {
      "resnet_supervised": {"accuracy": X, "training_time": X, "epochs": X},
      "simclr_no_macenko": {"accuracy": X, "training_time": X, "epochs": X},
      "final_model":       {"accuracy": X, "training_time": X, "epochs": X}
    }
    ```
    """
    if not os.path.exists(COMPARISON_PATH):
        raise HTTPException(
            status_code=404,
            detail="comparison.json not found. Run train.py and fine_tune.py first."
        )
    with open(COMPARISON_PATH, 'r') as f:
        data = json.load(f)
    return JSONResponse(data)


# ======================
# BASELINE TRAINING ENDPOINTS
# ======================

def _read_resnet_supervised() -> dict:
    """Read resnet_supervised entry from comparison.json, return zeros if missing."""
    try:
        if os.path.exists(COMPARISON_PATH):
            with open(COMPARISON_PATH, 'r') as f:
                data = json.load(f)
            return data.get("resnet_supervised", {"accuracy": 0.0, "training_time": 0, "epochs": 0})
    except Exception:
        pass
    return {"accuracy": 0.0, "training_time": 0, "epochs": 0}


@app.post("/train_baseline", summary="Spawn supervised baseline training subprocess")
def train_baseline():
    """
    Start `train_supervised_baseline.py` as a non-blocking background subprocess.

    Returns:
    - **status**: "started" (202), "already_running" (200), or "already_done" (200)
    """
    global _baseline_proc, _baseline_done

    # 1. Already done — comparison.json has non-zero resnet accuracy
    rs = _read_resnet_supervised()
    if rs.get("accuracy", 0.0) > 0.0:
        _baseline_done = True
        return JSONResponse({"status": "already_done"}, status_code=200)

    # 2. Already running — process exists and hasn't exited
    if _baseline_proc is not None and _baseline_proc.poll() is None:
        return JSONResponse({"status": "already_running"}, status_code=200)

    # 3. Start fresh subprocess
    try:
        _baseline_proc = subprocess.Popen(
            [sys.executable, BASELINE_SCRIPT],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return JSONResponse({"status": "started"}, status_code=202)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start baseline training: {e}")


@app.get("/train_baseline/status", summary="Get supervised baseline training status")
def train_baseline_status():
    """
    Return current training status and resnet_supervised metrics.

    Returns:
    - **status**: "idle" | "running" | "done"
    - **resnet_supervised**: accuracy, training_time, epochs
    """
    global _baseline_proc, _baseline_done

    rs = _read_resnet_supervised()

    # Done if comparison.json has non-zero accuracy
    if rs.get("accuracy", 0.0) > 0.0:
        _baseline_done = True
        return JSONResponse({"status": "done", "resnet_supervised": rs})

    # Running if process is alive
    if _baseline_proc is not None and _baseline_proc.poll() is None:
        return JSONResponse({"status": "running", "resnet_supervised": rs})

    # Idle
    return JSONResponse({"status": "idle", "resnet_supervised": rs})


# Serve individual graph images via explicit API routes (avoids StaticFiles route conflicts)
from fastapi.responses import FileResponse

@app.get("/graphs/{filename}", summary="Serve a graph image")
def serve_graph(filename: str):
    path = os.path.join(GRAPHS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Graph not found: {filename}")
    return FileResponse(path)
