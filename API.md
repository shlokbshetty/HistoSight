# HistoSight — REST & Python API Reference (`API.md`)

HistoSight provides a production-ready **FastAPI** diagnostic microservice (`backend/main.py`) and a modular **Python API** (`inference.py`, `explainability.py`, `stain_norm.py`) for histopathology patch classification, Macenko stain normalization, and multi-modal Explainable AI (XAI) saliency generation.

---

## 1. Base URLs & Authentication

| Property | Value |
| :--- | :--- |
| **Base URL (Default)** | `http://127.0.0.1:8000` |
| **Swagger UI (`/docs`)** | `http://127.0.0.1:8000/docs` |
| **ReDoc (`/redoc`)** | `http://127.0.0.1:8000/redoc` |
| **Authentication** | None (Local diagnostic microservice with `CORSMiddleware` enabled) |
| **Supported Image Formats** | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif` |

---

## 2. REST API Endpoints (`backend/main.py`)

### 2.1 `GET /` — Root Service Status
Verifies that the FastAPI ASGI service is online.

- **Request Parameters**: None
- **Response (`200 OK`)**:
```json
{
  "status": "ok",
  "message": "Stain-Invariant SimCLR API is running."
}
```

---

### 2.2 `GET /health` — Readiness & Telemetry Health Check
Checks service health and whether `logs/metrics.json` and `logs/comparison.json` are available on disk.

- **Request Parameters**: None
- **Response (`200 OK`)**:
```json
{
  "status": "healthy",
  "metrics_ready": true,
  "comparison_ready": true
}
```

---

### 2.3 `POST /predict` — Single Histopathology Patch Classification
Uploads a single histopathology image patch, applies Macenko stain normalization (`stain_norm.normalize_image`), runs a forward pass through the cached `ResNet-18` encoder and `nn.Linear(512, 2)` classifier, and returns the predicted class (`"Tumor"` or `"Normal"`) and softmax confidence.

- **Content-Type**: `multipart/form-data`
- **Form Parameters**:
  - `file` (`UploadFile`, required): Image file (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`).
- **Example `curl` Request**:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@pcam_test_100_per_class/pcam_test_100_per_class/1_tumor/test_patch_2.png"
```
- **Response (`200 OK`)**:
```json
{
  "class": "Tumor",
  "confidence": 0.9142
}
```
- **Error Responses**:
  - `400 Bad Request`: Unsupported file extension.
    ```json
    {
      "detail": "Unsupported file type: .gif. Use JPEG, PNG, BMP, or TIFF."
    }
    ```
  - `503 Service Unavailable`: Model checkpoints (`models/best_classifier.pth` or `checkpoints/best_simclr.pth`) are missing.
  - `500 Internal Server Error`: Runtime inference error.

---

### 2.4 `POST /batch_predict` — Server-Side Directory Batch Inference
Runs Macenko stain normalization and classification across all supported images in a server-side folder path. If a `labels.json` file (`{"filename.png": 0_or_1, ...}`) is present in the folder, also computes aggregate accuracy and cross-entropy loss.

- **Content-Type**: `application/json`
- **Request Schema (`BatchPredictRequest`)**:
  - `folder` (`str`, required): Path to directory containing histopathology patches.
- **Example Request Body**:
```json
{
  "folder": "c:/GitHub/HistoSight/pcam_test_100_per_class/pcam_test_100_per_class/0_normal"
}
```
- **Response (`200 OK`)**:
```json
{
  "predictions": [
    {
      "file": "test_patch_0.png",
      "class": "Normal",
      "confidence": 0.8831
    },
    {
      "file": "test_patch_101.png",
      "class": "Normal",
      "confidence": 0.9415
    }
  ],
  "accuracy": 0.91,
  "loss": 0.2418
}
```
*(Note: `accuracy` and `loss` are `null` if `labels.json` is not present in the target folder.)*
- **Error Responses**:
  - `400 Bad Request`: `{"detail": "Not a valid directory: <path>"}`
  - `404 Not Found`: `{"detail": "No supported images found in: <path>"}`
  - `503 Service Unavailable`: Model checkpoints missing.

---

### 2.5 `POST /gradcam` — Grad-CAM Saliency Map Generation
Uploads a single image patch and computes a `96x96` Grad-CAM activation heatmap from `encoder.layer4` alongside the diagnostic prediction.

- **Content-Type**: `multipart/form-data`
- **Form Parameters**:
  - `file` (`UploadFile`, required): Image file (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`).
- **Response (`200 OK`)**:
```json
{
  "class": "Tumor",
  "confidence": 0.9142,
  "saliency_map": [
    [0.012, 0.018, 0.025, "... 96 floats per row ..."],
    ["... 96 rows total (values in [0.0, 1.0]) ..."]
  ]
}
```
- **Error Responses**:
  - `400 Bad Request`: `{"detail": "Unsupported file type: <ext>"}`
  - `503 Service Unavailable`: Model checkpoints missing.

---

### 2.6 `POST /explain` — Multi-Modal XAI Saliency Generation
Uploads an image patch and computes a `96x96` normalized (`[0.0, 1.0]`) interpretability heatmap using any of the 6 supported XAI methods in `explainability.py`.

- **Content-Type**: `multipart/form-data`
- **Query Parameters**:
  - `method` (`str`, optional, default `"gradcam"`): One of:
    - `"gradcam"`: Gradient-weighted Class Activation Mapping on `layer4`
    - `"saliency"`: Vanilla input-space gradient magnitude ($\max_c |\nabla_x y|$)
    - `"smoothgrad"`: Noise-averaged gradients ($N=50, \sigma=0.15$)
    - `"occlusion"`: Sliding-window `16x16` patch occlusion sensitivity ($\text{stride}=8$)
    - `"lime"`: Local Interpretable Model-agnostic Explanations (`num_samples=1000`)
    - `"shap"`: Deep learning `shap.GradientExplainer` Shapley attributions
- **Form Parameters**:
  - `file` (`UploadFile`, required): Image file.
- **Example `curl` Request**:
```bash
curl -X POST "http://127.0.0.1:8000/explain?method=smoothgrad" \
  -F "file=@pcam_test_100_per_class/pcam_test_100_per_class/1_tumor/test_patch_2.png"
```
- **Response (`200 OK`)**:
```json
{
  "class": "Tumor",
  "confidence": 0.9142,
  "saliency_map": [
    [0.104, 0.219, 0.341, "... 96 floats ..."],
    ["... 96 rows ..."]
  ]
}
```
- **Error Responses**:
  - `400 Bad Request`: `{"detail": "Unsupported explainability method: <method>"}` or `{"detail": "Unsupported file type: <ext>"}`
  - `503 Service Unavailable`: Model checkpoints missing.
  - `500 Internal Server Error`: If optional package (`lime` or `shap`) is not installed or heatmap generation fails.

---

### 2.7 `GET /metrics` — Retrieve SimCLR Pre-Training Telemetry
Returns the contents of `logs/metrics.json` recorded by `train.py`.

- **Request Parameters**: None
- **Response (`200 OK`)**:
```json
{
  "epoch": [1, 2, 3, "... 30"],
  "train_loss": [4.990238, 4.764574, 4.715504, "..."],
  "val_loss": [5.013978, 5.169463, 4.785013, "..."],
  "train_accuracy": [0.096029, 0.162159, 0.208784, "..."],
  "val_accuracy": [0.131426, 0.122606, 0.227067, "..."],
  "epoch_time": [62.24, 50.02, 63.69, "..."]
}
```
- **Error Responses**:
  - `404 Not Found`: `{"detail": "metrics.json not found. Run train.py to generate it."}`

---

### 2.8 `GET /comparison` — Retrieve Model Benchmark Comparison
Returns the contents of `logs/comparison.json` generated by `train.py`, `fine_tune.py`, and `compare_baselines.py`.

- **Request Parameters**: None
- **Response (`200 OK`)**:
```json
{
  "resnet18_pretrained": {
    "accuracy": 0.8495,
    "training_time": 0,
    "epochs": 0,
    "technique": "Residual connections",
    "display_name": "ResNet18 (ImageNet)"
  },
  "resnet50_pretrained": {
    "accuracy": 0.9317,
    "training_time": 4.7,
    "epochs": 0,
    "technique": "Deep residual + bottleneck",
    "display_name": "ResNet50 (ImageNet)"
  },
  "efficientnet_b0": {
    "accuracy": 0.9345,
    "training_time": 3.4,
    "epochs": 0,
    "technique": "Compound scaling",
    "display_name": "EfficientNet-B0 (ImageNet)"
  },
  "mobilenet_v3_small": {
    "accuracy": 0.8655,
    "training_time": 2.9,
    "epochs": 0,
    "technique": "Depthwise separable + SE",
    "display_name": "MobileNetV3-Small (ImageNet)"
  },
  "simclr_ssl": {
    "accuracy": 0.5989,
    "training_time": 2171.8,
    "epochs": 30,
    "macenko": true
  },
  "final_model": {
    "accuracy": 0.8083,
    "training_time": 442.9,
    "epochs": 20
  }
}
```
- **Error Responses**:
  - `404 Not Found`: `{"detail": "comparison.json not found. Run train.py and fine_tune.py first."}`

---

### 2.9 `POST /train_baseline` — Spawn Background Baseline Benchmark
Spawns `compare_baselines.py` as a non-blocking background subprocess if not already completed or running.

- **Request Parameters**: None
- **Responses**:
  - `202 Accepted`: `{"status": "started"}` (Subprocess launched)
  - `200 OK`: `{"status": "already_running"}` or `{"status": "already_done"}`
  - `500 Internal Server Error`: `{"detail": "Failed to start baseline training: <error>"}`

---

### 2.10 `GET /train_baseline/status` — Poll Baseline Benchmark Status
Returns the current execution state of the background `compare_baselines.py` worker and the `resnet18_pretrained` metrics entry.

- **Request Parameters**: None
- **Response (`200 OK`)**:
```json
{
  "status": "done",
  "resnet_supervised": {
    "accuracy": 0.8495,
    "training_time": 0,
    "epochs": 0,
    "technique": "Residual connections",
    "display_name": "ResNet18 (ImageNet)"
  }
}
```
*(Possible `status` values: `"idle"`, `"running"`, `"done"`.)*

---

### 2.11 `GET /graphs/{filename}` — Serve Generated Chart PNGs
Streams static PNG chart files from the `graphs/` directory (`loss_curve.png`, `accuracy_curve.png`, `epoch_time.png`, `model_comparison.png`, `system_architecture.png`).

- **Path Parameter**: `filename` (`str`, required)
- **Response (`200 OK`)**: `image/png` binary stream (`FileResponse`)
- **Error Response (`404 Not Found`)**: `{"detail": "Graph not found: <filename>"}`

---

## 3. Python Module API Reference

### 3.1 `stain_norm.py`
```python
from PIL import Image
import numpy as np
import stain_norm

# Initialize / retrieve cached MacenkoNormalizer
normalizer = stain_norm.get_normalizer()

# Normalize a PIL Image (returns 96x96 RGB PIL.Image)
raw_img = Image.open("sample.png").convert("RGB")
norm_img = stain_norm.normalize_image(raw_img)

# Normalize a raw uint8 numpy array of shape (H, W, 3)
raw_arr = np.array(raw_img)
norm_arr = stain_norm.normalize_numpy(raw_arr)
```

### 3.2 `inference.py`
```python
import inference

# Single image prediction -> {"class": "Tumor" | "Normal", "confidence": float}
single_res = inference.predict_single("pcam_test_100_per_class/pcam_test_100_per_class/1_tumor/test_patch_2.png")

# Batch directory prediction -> {"predictions": [...], "accuracy": float | None, "loss": float | None}
batch_res = inference.predict_batch("pcam_test_100_per_class/pcam_test_100_per_class/0_normal")

# Grad-CAM dictionary -> {"class": str, "confidence": float, "saliency_map": list[list[float]]}
gradcam_res = inference.generate_gradcam("pcam_test_100_per_class/pcam_test_100_per_class/1_tumor/test_patch_2.png")
```

### 3.3 `explainability.py`
```python
import explainability

img_path = "pcam_test_100_per_class/pcam_test_100_per_class/1_tumor/test_patch_2.png"

heatmap, pred_class = explainability.gradcam(img_path, target_layer_name="layer4")
heatmap, pred_class = explainability.smoothgrad(img_path, n_samples=50, noise_level=0.15)
heatmap, pred_class = explainability.occlusion(img_path, patch_size=16, stride=8)
heatmap, pred_class = explainability.saliency(img_path)
heatmap, pred_class = explainability.lime_explain(img_path, num_samples=1000)
heatmap, pred_class = explainability.shap_explain(img_path, n_samples=100)

# Save 3-panel matplotlib overlay figure
explainability.visualize(img_path, heatmap, pred_class, method_name="GRADCAM", save_path="out.png")
```
