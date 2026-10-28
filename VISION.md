# HistoSight — Architectural & Research Vision (`VISION.md`)

## 1. Mission Statement

**HistoSight** is built on the conviction that clinical computational pathology must be **stain-invariant**, **label-efficient**, and **intrinsically interpretable**. Our mission is to bridge the gap between self-supervised representation learning research and actionable bedside diagnostics—enabling pathologists to screen lymph node biopsies for micrometastases with high sensitivity, regardless of laboratory staining protocols, while inspecting transparent pixel-level attribution maps for every prediction.

---

## 2. Core Scientific & Architectural Pillars

### Pillar 1: Physics-Grounded Chromatic Standardization
Standard data augmentation (`ColorJitter`) treats H&E color variation as arbitrary RGB noise rather than a physical optical phenomenon governed by Beer-Lambert absorption of Hematoxylin (basophilic nuclei) and Eosin (eosinophilic cytoplasm/stroma). By placing **Macenko Optical Density Decomposition** (`stain_norm.py`) ahead of **SimCLR Contrastive Augmentation** (`train.py`), HistoSight projects every input tile onto a canonical stain basis (`checkpoints/template.png`). Consequently, the `ResNet-18` encoder devotes its full `512-dimensional` capacity to learning nuclear pleomorphism, mitotic figures, and glandular architecture rather than stain intensity artifacts.

### Pillar 2: Label-Efficient Self-Supervised Representation Learning
Acquiring pixel-level or patch-level annotations from board-certified pathologists across hundreds of thousands of histology tiles is a major bottleneck. Using `NT-Xent` contrastive learning (`train.py`), HistoSight learns rich morphologic representations from **100% unlabeled histopathology patches** (`pcam_ssl_train_unlabeled_x.h5`). When paired with a lightweight `nn.Linear(512, 2)` probe and selective `layer4` fine-tuning (`fine_tune.py`), the model achieves strong downstream diagnostic accuracy (`>80.8%` on PCam validation splits trained purely from scratch without ImageNet pre-training).

### Pillar 3: Multi-Perspective Explainable AI (XAI) Consensus
No single interpretability method is sufficient for high-stakes medical imaging:
- **Layer-Wise Activation Mapping (`Grad-CAM`)** highlights coarse semantic regions (`6x6` feature grid upsampled to `96x96`) where `layer4` detects tumor nuclei clusters.
- **Fine-Grained Gradient Attribution (`Vanilla Saliency` & `SmoothGrad`)** pinpoints individual hyperchromatic cell boundaries at native `96x96` resolution.
- **Perturbation & Game-Theoretic Attribution (`Occlusion`, `LIME`, & `SHAP`)** quantifies the counterfactual impact of masking specific tissue subregions or superpixels on the final tumor probability.

By unifying all six algorithms behind a single API (`POST /explain?method=...` in `backend/main.py`) and interactive console (`index.html`), HistoSight empowers clinicians and ML researchers to cross-verify model attention across multiple mathematical paradigms.

---

## 3. Long-Term Clinical Vision

HistoSight's modular architecture is designed to scale from `96x96` PatchCamelyon tiles to **Whole-Slide Image (WSI) Multiple Instance Learning (MIL)**:
1. **Slide-Level Aggregation**: Using the frozen stain-invariant SimCLR encoder to embed thousands of patches per gigapixel biopsy slide into `512-d` feature bags for Attention-based MIL (`ABMIL` / `TransMIL`).
2. **Human-in-the-Loop Triage**: Automatically surfacing the top-K highest-confidence tumor patches alongside their Grad-CAM and SmoothGrad heatmaps in the web console to accelerate pathologist sign-out workflows.
