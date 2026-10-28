# HistoSight — Product & Research Roadmap (`ROADMAP.md`)

This document outlines the short-term engineering milestones, medium-term clinical validation goals, and long-term research trajectory for **HistoSight**.

---

## Phase 1: Stain-Invariant SimCLR Core & 6-Method XAI Suite (Completed — `v1.0.0`)
- [x] Macenko optical density stain normalization (`stain_norm.py`) with automatic reference patch extraction (`checkpoints/template.png`) and chunked HDF5 preprocessing (`normalize_dataset.py`).
- [x] Self-supervised `SimCLRModel` (`ResNet-18` + `128-d` projection MLP) pre-trained with `NT-Xent` InfoNCE loss ($\tau = 0.5$) and mixed precision (`train.py`).
- [x] Downstream RAM-cached linear probing and `layer4` fine-tuning (`fine_tune.py`) with early stopping (`PATIENCE = 8`).
- [x] Multi-model LBFGS baseline comparison across ImageNet `ResNet-18`, `ResNet-50`, `EfficientNet-B0`, and `MobileNetV3-Small` (`compare_baselines.py`).
- [x] Six-method XAI attribution suite (`Grad-CAM`, `SmoothGrad`, `Vanilla Saliency`, `Occlusion`, `LIME`, `SHAP`) in `explainability.py`.
- [x] FastAPI diagnostic microservice (`backend/main.py`) and interactive clinical browser console (`index.html`).

---

## Phase 2: Short-Term Milestones (`v1.1.0` — Q1)
- [ ] **Vahadane & Reinhard Stain Normalization Backends**:
  - Extend `stain_norm.py` with a configurable `--method [macenko|vahadane|reinhard]` parameter to benchmark structure-preserving sparse non-negative matrix factorization (SNMF) against Macenko SVD.
- [ ] **Quantitative XAI Faithfulness Metrics**:
  - Add Insertion/Deletion AUC curves and Pointing Game localization scores to `explainability.py` using ground-truth tumor mask annotations (`pcamv1/camelyonpatch_level_2_split_train_mask.h5.gz`).
- [ ] **ONNX / TorchScript Export for Diagnostic Deployment**:
  - Export the fine-tuned `ResNet-18` encoder and `nn.Linear(512, 2)` classifier to TorchScript and ONNX for sub-millisecond CPU inference in `inference.py`.

---

## Phase 3: Medium-Term Milestones (`v1.2.0` — Q2–Q3)
- [ ] **Gigapixel Whole-Slide Image (WSI) Tiling & Pyramid Stitcher**:
  - Integrate `OpenSlide` / `tifffile` into `inference.py` to stream `.svs` and `.ndpi` whole-slide pyramids, classify non-background tissue tiles in batches, and assemble slide-level metastatic probability heatmaps.
- [ ] **Vision Transformer (ViT / DINOv2) Self-Supervised Backbones**:
  - Add `vit_small` and `convnext_tiny` encoder options to `train.py` and `compare_baselines.py` alongside `ResNet-18`.
- [ ] **Uncertainty Quantification (Temperature Scaling & Monte Carlo Dropout)**:
  - Calibrate output softmax probabilities in `inference.py` using Expected Calibration Error (ECE) and flag high-entropy borderline patches for manual pathologist review.

---

## Phase 4: Long-Term Milestones (`v2.0.0` — Q4+)
- [ ] **Multi-Institutional External Validation (CAMELYON16 / CAMELYON17 / TCGA)**:
  - Evaluate zero-shot domain generalization across 5+ independent clinical centers without retraining the linear classifier.
- [ ] **DICOM-WSI & HL7 FHIR Diagnostic Report Integration**:
  - Export structured diagnostic reports and Grad-CAM overlays compatible with hospital Picture Archiving and Communication Systems (PACS).
