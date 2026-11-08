# Changelog

All notable changes to **HistoSight** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v1.1.0] - 2026-11-08

### Added

- **8-Fold Dihedral Test-Time Augmentation (TTA, $D_4$ Symmetry Group)**:
  - Added dihedral group averaging across all 4 planar rotations ($0^\circ, 90^\circ, 180^\circ, 270^\circ$) and horizontal reflections in `inference.py` (`predict_single`, `predict_batch`, `--tta`), `evaluate.py` (`--tta`), `backend/main.py` (`POST /predict?tta=true`, `BatchPredictRequest(tta=True)`), and `index.html` (interactive UI checkbox).
  - Demonstrates measurable accuracy improvements (e.g. $+1.00\%$ accuracy, lower cross-entropy loss, and higher ROC-AUC on held-out test splits).
- **High-Performance Clinical Evaluation Report (`evaluate.py`)**:
  - Implemented comprehensive clinical diagnostic metrics: Test Accuracy, Balanced Accuracy, Sensitivity (Recall), Specificity, Precision (PPV), F1 Score, ROC-AUC, PR-AUC, and Confusion Matrix ($TP, FP, TN, FN$).
  - Added `--max-samples`, `--batch-size`, `--in-memory`, and `--save-json` CLI arguments with high-throughput sequential slice loading.
- **Comprehensive Unit Test Suite (`tests/test_inference.py`, `tests/test_evaluate.py`)**:
  - Added unit tests for dihedral TTA batch generation, prediction structure, probability mass conservation ($\sum p_c = 1.0$), Grad-CAM shape/bounds, and clinical confusion matrix calculations.
  - All 9 pytest unit tests pass with zero warnings in under 12 seconds.

### Changed & Improved

- **Modern PyTorch AMP & Safe Checkpoint Loading**:
  - Migrated from deprecated `torch.cuda.amp.autocast()` to device-agnostic `torch.amp.autocast(device_type=...)` across `inference.py` and `evaluate.py`.
  - Set `weights_only=True` in `torch.load` calls for secure deserialization, eliminating PyTorch future deprecation warnings.
- **Macenko Stain Normalizer Numerical Stability (`stain_norm.py`)**:
  - Added proactive check for pure white or low-variance glass background patches (`mean > 245` or $\sigma < 3.0$) and wrapped SVD fitting in warning catchers to eliminate NumPy `Mean of empty slice` and `divide by zero` runtime warnings.
- **Documentation Typography & Mathematical Formula Spacing**:
  - Overhauled mathematical notation across `README.md`, `ARCHITECTURE.md`, `API.md`, `DEVELOPMENT.md`, and `SUPPORT.md` using spacious KaTeX formatting, proper fractions, and explicit spacing.
  - Fixed query string formatting (replaced errant `—` em-dashes in URLs and badges with `?`), fixed broken test patch file paths (`test_patch_2.png` $\to$ `test_patch_1.png`), and polished word spacing.

---

## [v1.0.0] - 2026-10-28

### Added

- **Macenko Stain Normalization Pipeline (`stain_norm.py`, `normalize_dataset.py`, `tests/test_stain_norm.py`)**:
  - Implemented singleton `MacenkoNormalizer` (`torchstain` NumPy backend) with automatic reference template extraction (`checkpoints/template.png`) from `pcam_ssl_val_labeled_x.h5`.
  - Added chunked (`1,000` patches/chunk) offline HDF5 dataset normalizer (`normalize_dataset.py`) producing `*_normalized.h5` splits.
  - Added `pytest` unit test suite (`tests/test_stain_norm.py`) verifying shape/dtype preservation and blank-patch SVD fallback.
- **SimCLR Self-Supervised Pre-Training (`train.py`)**:
  - Implemented `SimCLRModel` (`ResNet-18` backbone + `512->512->128` MLP projection head) with `NT-Xent` contrastive loss ($\tau = 0.5$), `AdamW`, `CosineAnnealingLR`, and CUDA mixed precision (`GradScaler` / `autocast`).
  - Added automatic checkpoint resumption (`checkpoints/best_simclr.pth`), `--scratch` flag, and atomic telemetry logging to `logs/metrics.json`.
- **Downstream Linear Evaluation & Layer-4 Fine-Tuning (`fine_tune.py`, `evaluate.py`)**:
  - Added fast RAM-cached `512-d` feature extraction (`extract_features`) for linear probing and `--finetune-encoder` mode for unfreezing `encoder.layer4` alongside `nn.Linear(512, 2)`.
  - Added held-out test split evaluator (`evaluate.py`) on `pcam_ssl_test_labeled_x/y.h5`.
- **Multi-Model Supervised Baseline Suite (`compare_baselines.py`)**:
  - Added automated LBFGS (`strong_wolfe`) linear probe benchmarking across ImageNet-pretrained `ResNet-18`, `ResNet-50`, `EfficientNet-B0`, and `MobileNetV3-Small`, saving results to `logs/comparison.json`.
- **6-Method Explainable AI (XAI) Suite (`explainability.py`, `generate_test_explanations.py`)**:
  - Implemented `gradcam`, `smoothgrad`, `saliency` (Vanilla Gradient), `occlusion`, `lime_explain` (LIME), and `shap_explain` (SHAP `GradientExplainer`), plus 3-panel matplotlib overlay visualization (`visualize`) and automated batch verification (`generate_test_explanations.py`).
- **FastAPI Diagnostic Microservice & Web Console (`backend/main.py`, `inference.py`, `index.html`, `generate_graphs.py`)**:
  - Added REST endpoints: `GET /`, `GET /health`, `POST /predict`, `POST /batch_predict`, `POST /gradcam`, `POST /explain`, `GET /metrics`, `GET /comparison`, `POST /train_baseline`, `GET /train_baseline/status`, and `GET /graphs/{filename}`.
  - Added interactive browser console (`index.html`) and dark-themed publication chart exporter (`generate_graphs.py`).

### Fixed

- Resolved PyTorch autograd hook conflict in `explainability.shap_explain()` by disabling in-place `nn.ReLU` operations (`module.inplace = False`) before invoking `shap.GradientExplainer`.
- Added automatic fallback in `train.py` from `pcam_ssl_train_unlabeled_x.h5` to `pcam_ssl_val_labeled_x_normalized.h5` if the unlabeled training HDF5 split is absent or unreadable.
