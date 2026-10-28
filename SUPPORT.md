# HistoSight — Support & Troubleshooting (`SUPPORT.md`)

Need assistance running SimCLR pre-training, troubleshooting Macenko stain normalization, or configuring the FastAPI diagnostic backend? Consult the resources and FAQs below.

---

## 1. Channels for Help

| Resource | Purpose | Location |
| :--- | :--- | :--- |
| **Architecture & API Docs** | Mathematical formulations, system diagrams, and endpoint specs | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/API.md](docs/API.md), [DEVELOPMENT.md](DEVELOPMENT.md) |
| **Bug Tracker** | Reporting training crashes, CUDA OOM errors, or API bugs | [`.github/ISSUE_TEMPLATE/bug_report.md`](.github/ISSUE_TEMPLATE/bug_report.md) |
| **Feature Requests** | Suggesting new SSL objectives, backbones, or XAI methods | [`.github/ISSUE_TEMPLATE/feature_request.md`](.github/ISSUE_TEMPLATE/feature_request.md) |
| **Security Reports** | Private vulnerability disclosure | [SECURITY.md](SECURITY.md) |

---

## 2. Issue Tracker Labels

- `bug`: Reproducible crash, tensor shape mismatch, or API error
- `enhancement`: New feature or architecture improvement
- `stain-normalization`: `stain_norm.py`, `torchstain`, or `normalize_dataset.py` issues
- `simclr-ssl`: Contrastive pre-training (`train.py`) or linear probing (`fine_tune.py`)
- `explainability-xai`: `Grad-CAM`, `SmoothGrad`, `Occlusion`, `LIME`, `SHAP`, or `Saliency` (`explainability.py`)
- `fastapi-backend`: `backend/main.py` or `index.html` console issues

---

## 3. Project-Specific FAQs

### Q1: `POST /predict` or `POST /gradcam` returns `HTTP 503 Service Unavailable` with `"Classifier checkpoint not found"`.
`inference._load_models()` requires `models/best_classifier.pth` and either `models/best_encoder.pth` or `checkpoints/best_simclr.pth`. If you have not run `fine_tune.py` yet, run:
```powershell
python train.py --epochs 30
python fine_tune.py --epochs 20
```

### Q2: `POST /explain?method=lime` or `POST /explain?method=shap` returns `HTTP 500 Internal Server Error`.
`LIME` and `SHAP` rely on optional packages (`lime`, `scikit-image`, `shap`) that are imported lazily inside `explainability.lime_explain()` and `explainability.shap_explain()`. Install them in your active virtual environment:
```powershell
pip install lime scikit-image shap
```

### Q3: Why did `shap_explain` previously fail with a PyTorch `RuntimeError: Output 0 of BackwardHookFunctionBackward is a view and is being modified inplace`?
Standard `torchvision.models.resnet18` layers use `nn.ReLU(inplace=True)`, which conflicts with `shap.GradientExplainer` backward hooks. HistoSight automatically resolves this in `explainability.py` (lines 215–217) by iterating through `model.modules()` and setting `module.inplace = False` on all `nn.ReLU` instances prior to running `GradientExplainer`.

### Q4: What happens if a blank white background slide patch is passed to `stain_norm.normalize_numpy()`?
Pure white or near-empty glass regions lack sufficient optical density variance for Singular Value Decomposition (SVD) in Macenko normalization. `stain_norm.normalize_numpy()` wraps `normalizer.normalize()` in a `try...except` block and safely returns the original unmodified array when SVD fails (verified by `tests/test_stain_norm.py::test_normalize_blank_fallback`).

### Q5: How do I regenerate the training and comparison PNG charts in `graphs/`?
After running `train.py`, `fine_tune.py`, and `compare_baselines.py` (which populate `logs/metrics.json` and `logs/comparison.json`), run:
```powershell
python generate_graphs.py
```
This outputs `loss_curve.png`, `accuracy_curve.png`, `epoch_time.png`, `model_comparison.png`, and `system_architecture.png` in `graphs/`.
