## Summary of Changes

<!-- Provide a concise summary of what this Pull Request accomplishes and its impact on model accuracy, stain invariance, XAI, or API behavior. -->

## Related Issues

<!-- Link related issues using keywords, e.g., Closes #15, Fixes #8 -->
- Closes #

## Type of Change

- [ ] `feat`: New feature (stain normalization, SSL training, XAI method, FastAPI endpoint, or web UI)
- [ ] `fix`: Bug fix (resolves numerical instability, autograd hook error, or endpoint bug)
- [ ] `perf`: Performance optimization (mixed precision, HDF5 chunking, RAM feature caching)
- [ ] `refactor`: Code restructuring with no behavioral change
- [ ] `docs`: Documentation, SVG architecture diagram, or `paper.tex` update
- [ ] `test`: Addition or update of `pytest` unit tests or XAI verification scripts
- [ ] `chore`: Dependencies or workspace configuration update

## Affected Modules

- [ ] Macenko Stain Normalization (`stain_norm.py`, `normalize_dataset.py`)
- [ ] SimCLR Pre-Training (`train.py`)
- [ ] Linear Probe & Fine-Tuning (`fine_tune.py`, `evaluate.py`, `compare_baselines.py`)
- [ ] Diagnostic Inference (`inference.py`)
- [ ] Explainable AI (XAI) Suite (`explainability.py`, `generate_test_explanations.py`)
- [ ] FastAPI Backend (`backend/main.py`)
- [ ] Web Console & Graphs (`index.html`, `generate_graphs.py`)

## Verification & Test Coverage

- [ ] `pytest -v` passes (`tests/test_stain_norm.py`)
- [ ] Tested single-patch inference (`python inference.py --image ...`)
- [ ] Tested XAI heatmap generation (`python explainability.py --image ... --method ...`)
- [ ] Verified FastAPI routes (`uvicorn backend.main:app`)

## Self-Review Checklist

- [ ] Temporary files created during multipart uploads are cleaned up in `finally` blocks
- [ ] JSON metric updates use atomic `.tmp` write + `os.replace()`
- [ ] Documentation (`README.md`, `docs/API.md`, `docs/ARCHITECTURE.md`, `CHANGELOG.md`) is updated
- [ ] No large binary `.h5` datasets or virtual environment directories (`venv312/`) are staged
