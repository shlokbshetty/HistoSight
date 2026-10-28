# HistoSight — Developer Guide (`DEVELOPMENT.md`)

This document provides a technical deep-dive into setting up your local development environment, managing PatchCamelyon (PCam) HDF5 datasets and test fixtures, running unit tests and XAI verification suites, and maintaining code quality.

---

## 1. Local Developer Environment Setup

### 1.1 Python & Virtual Environment
HistoSight targets Python `3.10`–`3.12` (with `venv312` pre-configured for local development).

```powershell
# Create a fresh virtual environment (or use existing venv312)
python -m venv venv312

# Activate on Windows PowerShell
.\venv312\Scripts\Activate.ps1

# Install core dependencies from requirements.txt
pip install --upgrade pip
pip install -r requirements.txt

# Install development, XAI, and plotting packages
pip install pytest pytest-cov ruff flake8 black matplotlib lime scikit-image shap
```

### 1.2 Verifying GPU / CPU Detection
All HistoSight training and inference modules automatically detect CUDA availability:
```powershell
python -c "import torch, inference; print('PyTorch:', torch.__version__, '| Active Device:', inference.DEVICE)"
```

---

## 2. Dataset Preparation, Splits & Fixtures

### 2.1 HDF5 Self-Supervised Splits (`simclr/ssl_splits/`)
The training and evaluation pipelines expect HDF5 files under `simclr/ssl_splits/` containing a dataset key `'x'` of shape `(N, 96, 96, 3)` (`uint8`) for images and `'y'` of shape `(N, 1, 1, 1)` for binary labels (`0 = Normal`, `1 = Tumor`):

| File Path | Role | Automatic Fallback Behavior |
| :--- | :--- | :--- |
| `pcam_ssl_train_unlabeled_x_normalized.h5` | Primary SimCLR pre-training input | Falls back to `pcam_ssl_train_unlabeled_x.h5`, or `pcam_ssl_val_labeled_x_normalized.h5` if unreadable (`train.py` lines 169–185) |
| `pcam_ssl_val_labeled_x_normalized.h5` | SimCLR validation & `fine_tune.py` training | Falls back to `pcam_ssl_val_labeled_x.h5` with on-the-fly Macenko normalization in RAM (`fine_tune.py` lines 72–78) |
| `pcam_ssl_val_labeled_y.h5` | Ground-truth labels for validation/fine-tuning | Required by `fine_tune.py` and `compare_baselines.py` |
| `pcam_ssl_test_labeled_x_normalized.h5` | Held-out test images | Falls back to `pcam_ssl_test_labeled_x.h5` (`evaluate.py` lines 33–36) |
| `pcam_ssl_test_labeled_y.h5` | Held-out test labels | Required by `evaluate.py` |

To generate the `*_normalized.h5` files from raw `.h5` splits:
```powershell
python normalize_dataset.py
```

### 2.2 Curated PNG Test Fixture (`pcam_test_100_per_class/`)
For rapid CLI and REST API testing without loading multi-gigabyte HDF5 archives, HistoSight includes 200 pre-extracted PNG patches in:
- `pcam_test_100_per_class/pcam_test_100_per_class/0_normal/` (100 Normal lymph node patches)
- `pcam_test_100_per_class/pcam_test_100_per_class/1_tumor/` (100 Metastatic tumor patches)

You can also place an optional `labels.json` (`{"test_patch_0.png": 0, ...}`) inside any image folder passed to `POST /batch_predict` or `inference.predict_batch()` to compute batch accuracy and cross-entropy loss.

### 2.3 Model Checkpoints & Reference Template
- `checkpoints/template.png`: Reference `96x96` H&E patch automatically extracted by `stain_norm.get_normalizer()` on first run.
- `checkpoints/best_simclr.pth`: Saved by `train.py` whenever validation `NT-Xent` loss improves.
- `models/best_classifier.pth` & `models/best_encoder.pth`: Saved by `fine_tune.py` whenever validation accuracy improves.

---

## 3. Testing & Verification Commands

### 3.1 Unit Tests (`pytest`)
Run the stain normalization unit test suite (`tests/test_stain_norm.py`), which verifies:
1. `test_get_normalizer`: `MacenkoNormalizer` initialization and `checkpoints/template.png` fitting.
2. `test_normalize_image_shape`: Preservation of `(96, 96, 3)` spatial dimensions and `uint8` dtype on synthetic H&E tissue arrays.
3. `test_normalize_blank_fallback`: Graceful SVD fallback with zero `NaN` values on pure white (`255`) background images.

```powershell
# Run pytest suite
pytest -v

# Run pytest with coverage report
pytest --cov=stain_norm --cov=inference --cov-report=term-missing
```

### 3.2 End-to-End Explainability Verification (`generate_test_explanations.py`)
To verify that all 6 XAI methods (`gradcam`, `saliency`, `smoothgrad`, `occlusion`, `lime`, `shap`) execute cleanly and render 3-panel overlay images (`Original` | `Heatmap` | `Overlay`):
```powershell
python generate_test_explanations.py
```
Generated PNGs are written to `test/normal_sample_<method>.png` and `test/tumor_sample_<method>.png`.

### 3.3 Held-Out Test Split Evaluation (`evaluate.py`)
```powershell
python evaluate.py
```
Outputs overall Test Accuracy (`%`) and Cross-Entropy Loss on `pcam_ssl_test_labeled_x/y.h5`.

---

## 4. Linting & Formatting Scripts

To keep the codebase clean and PEP 8 compliant:

```powershell
# Run Ruff linter across root modules and backend
ruff check stain_norm.py normalize_dataset.py train.py fine_tune.py evaluate.py compare_baselines.py inference.py explainability.py generate_graphs.py generate_test_explanations.py backend/ tests/

# Run Flake8 syntax and undefined-name check
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=venv312,.git,__pycache__

# Format code with Black
black stain_norm.py normalize_dataset.py inference.py explainability.py backend/ tests/
```
