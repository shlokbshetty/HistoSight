---
name: Bug Report
about: Report a bug in HistoSight stain normalization, SimCLR training, XAI attribution, or FastAPI endpoints
title: "[BUG] <short descriptive summary>"
labels: ["bug"]
assignees: ""
---

## Bug Description

<!-- Provide a clear and concise description of the bug. -->

## Affected Component

- [ ] Macenko Stain Normalizer (`stain_norm.py`, `normalize_dataset.py`)
- [ ] SimCLR SSL Pre-Training (`train.py`)
- [ ] Linear Probe / Fine-Tuning / Evaluation (`fine_tune.py`, `evaluate.py`, `compare_baselines.py`)
- [ ] Inference Core (`inference.py`)
- [ ] Explainable AI (XAI) Suite (`explainability.py`)
- [ ] FastAPI Diagnostic Service (`backend/main.py`)
- [ ] Interactive Web Console (`index.html`)

## Steps to Reproduce

1. Run script or call endpoint: `...`
2. Input image / dataset split used: `...`
3. Observe exception or unexpected output: `...`

## Expected Behavior

<!-- Describe what should have occurred. -->

## Actual Behavior & Stack Trace

<!-- Paste full traceback, HTTP error payload, or console log below: -->
```text
Paste traceback or log output here
```

## Environment Information

- **OS & Version**: (e.g., Windows 11 / Ubuntu 22.04)
- **Python Version**: (e.g., Python 3.12)
- **PyTorch & Torchvision Version**: (e.g., `torch 2.3.0+cu121`)
- **Compute Device (`cuda` / `cpu`) & GPU Model**:
- **Torchstain Version**:
