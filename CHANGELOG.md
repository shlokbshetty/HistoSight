# Changelog

All notable changes to **HistoSight** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
