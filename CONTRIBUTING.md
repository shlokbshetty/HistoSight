# Contributing to HistoSight

Thank you for contributing to **HistoSight**! Whether you are refining Macenko stain normalization, experimenting with self-supervised contrastive augmentations, adding new Explainable AI (XAI) attribution algorithms, or improving the FastAPI service and web console, please follow the guidelines below.

---

## 1. Branching Conventions

Create topic branches from `main` using the following naming pattern:

- `feat/<short-slug>`: New capabilities (e.g., `feat/vahadane-stain-normalizer`, `feat/integrated-gradients-xai`)
- `fix/<short-slug>`: Bug fixes (e.g., `fix/macenko-svd-blank-patch-fallback`, `fix/shap-inplace-relu`)
- `perf/<short-slug>`: Training speed, RAM caching, or mixed-precision optimizations
- `docs/<short-slug>`: Documentation, research paper (`paper.tex`), or SVG diagram updates
- `test/<short-slug>`: New `pytest` tests in `tests/` or explainability verification scripts
- `chore/<short-slug>`: Dependency updates or workspace maintenance

---

## 2. Conventional Commits Guidelines

All commit messages must follow [Conventional Commits v1.0.0](https://www.conventionalcommits.org/):

```text
<type>(<scope>): <concise summary in imperative mood>

[optional body explaining mathematical or architectural reasoning]

[optional footer(s): Closes #42]
```

### Supported Scopes
- `stain`: `stain_norm.py` and `normalize_dataset.py`
- `simclr`: `train.py` and `simclr/pre-train.py`
- `finetune`: `fine_tune.py`, `evaluate.py`, and `compare_baselines.py`
- `inference`: `inference.py`
- `xai`: `explainability.py` and `generate_test_explanations.py`
- `api`: `backend/main.py`
- `ui`: `index.html` and `generate_graphs.py`
- `docs`: Documentation and `paper.tex`

### Examples
- `feat(xai): add SmoothGrad and SHAP GradientExplainer to explainability.py`
- `fix(stain): catch SVD convergence errors on blank white patches in normalize_numpy`
- `perf(finetune): pre-extract 512-d ResNet18 features to RAM for linear probing`

---

## 3. Pull Request Lifecycle

1. **Fork & Branch**: Branch off `main` using the naming conventions above.
2. **Develop & Verify**:
   - Preserve numerical stability (guard against zero-division with `+ 1e-8` in normalization and XAI heatmaps).
   - Ensure temporary uploaded files in `backend/main.py` are always unlinked in `finally` blocks.
   - Add unit tests under `tests/` for any new preprocessing or inference logic.
3. **Run Pre-Commit Checks** (Section 4).
4. **Submit Pull Request**: Fill out `.github/PULL_REQUEST_TEMPLATE.md` completely.

---

## 4. Pre-Commit Verification Checks

Run the following checks before opening a PR:

```powershell
# 1. Run unit tests
pytest -v

# 2. Verify single-patch inference and Grad-CAM execution
python inference.py --image pcam_test_100_per_class/pcam_test_100_per_class/1_tumor/test_patch_2.png

# 3. Verify syntax and lint rules
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=venv312,.git,__pycache__
```

---

## 5. Code of Conduct

All contributors are expected to uphold our [Contributor Covenant Code of Conduct](.github/CODE_OF_CONDUCT.md).
