# =============================================================================
# compare_baselines.py — Multi-Model Pretrained Baseline Comparison
#
# Evaluates 4 pretrained models using different techniques:
#   1. ResNet18  — ImageNet pretrained, linear probe (frozen features)
#   2. ResNet50  — Deeper ResNet, linear probe (frozen features)
#   3. EfficientNet-B0 — Compound scaling, linear probe
#   4. MobileNetV3-Small — Lightweight depthwise separable convolutions
#
# Each model: extract features → LBFGS logistic regression → val accuracy
# Updates comparison.json with all results.
# =============================================================================

import os, sys, json, time
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from torch.cuda.amp import autocast
from PIL import Image

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "simclr", "ssl_splits")
LOG_DIR         = os.path.join(BASE_DIR, "logs")
COMPARISON_PATH = os.path.join(LOG_DIR, "comparison.json")

VAL_X = os.path.join(DATA_DIR, "pcam_ssl_val_labeled_x.h5")
VAL_Y = os.path.join(DATA_DIR, "pcam_ssl_val_labeled_y.h5")
MAX_SAMPLES = 6000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {DEVICE}")

os.makedirs(LOG_DIR, exist_ok=True)

# ======================
# TRANSFORM
# ======================
TRANSFORM = T.Compose([
    T.Resize((96, 96)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                std=[0.229, 0.224, 0.225]),
])

# ======================
# DATA LOADING
# ======================
def load_val_data():
    print("[DATA] Loading val data into RAM…", end=" ", flush=True)
    with h5py.File(VAL_X, 'r') as f:
        n = min(len(f['x']), MAX_SAMPLES)
        x = f['x'][:n]
    with h5py.File(VAL_Y, 'r') as f:
        y = f['y'][:n].squeeze().astype(np.int64)
    print(f"done. {n} samples")
    return x, y

# ======================
# FEATURE EXTRACTION
# ======================
def extract_features(backbone, x_np, batch_size=256):
    backbone.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(x_np), batch_size):
            batch = x_np[i:i+batch_size]
            imgs  = torch.stack([TRANSFORM(Image.fromarray(b)) for b in batch]).to(DEVICE)
            with autocast(enabled=(DEVICE == "cuda")):
                f = backbone(imgs)
            feats.append(f.float().cpu())
    return torch.cat(feats)

# ======================
# LINEAR PROBE (LBFGS)
# ======================
def linear_probe(feats, labels_np, n_classes=2):
    """Fast logistic regression via LBFGS on extracted features."""
    labels = torch.tensor(labels_np, dtype=torch.long).to(DEVICE)
    feats  = feats.to(DEVICE)

    clf = nn.Linear(feats.shape[1], n_classes).to(DEVICE)
    opt = torch.optim.LBFGS(clf.parameters(), lr=0.1, max_iter=100,
                             line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(clf(feats), labels)
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        preds = clf(feats).argmax(dim=1)
        acc   = (preds == labels).float().mean().item()
    return round(acc, 4)

# ======================
# MODEL DEFINITIONS
# ======================
def get_models():
    """
    Returns list of (key, display_name, technique, backbone).
    Each backbone has .fc / .classifier replaced with Identity.
    """
    entries = []

    # 1. ResNet18 — classic residual connections, ImageNet pretrained
    r18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    feat_dim_r18 = r18.fc.in_features
    r18.fc = nn.Identity()
    entries.append(("resnet18_pretrained", "ResNet18 (ImageNet)", "Residual connections", r18, feat_dim_r18))

    # 2. ResNet50 — deeper residual network, bottleneck blocks
    r50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    feat_dim_r50 = r50.fc.in_features
    r50.fc = nn.Identity()
    entries.append(("resnet50_pretrained", "ResNet50 (ImageNet)", "Deep residual + bottleneck", r50, feat_dim_r50))

    # 3. EfficientNet-B0 — compound scaling (width/depth/resolution)
    eff = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    feat_dim_eff = eff.classifier[1].in_features
    eff.classifier = nn.Identity()
    entries.append(("efficientnet_b0", "EfficientNet-B0 (ImageNet)", "Compound scaling", eff, feat_dim_eff))

    # 4. MobileNetV3-Small — depthwise separable convolutions, SE blocks
    mob = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    feat_dim_mob = mob.classifier[0].in_features
    mob.classifier = nn.Identity()
    entries.append(("mobilenet_v3_small", "MobileNetV3-Small (ImageNet)", "Depthwise separable + SE", mob, feat_dim_mob))

    return entries

# ======================
# LOAD / SAVE COMPARISON
# ======================
def load_comparison():
    if os.path.exists(COMPARISON_PATH):
        with open(COMPARISON_PATH) as f:
            return json.load(f)
    return {}

def save_comparison(data):
    tmp = COMPARISON_PATH + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, COMPARISON_PATH)

# ======================
# MAIN
# ======================
def main():
    val_x, val_y = load_val_data()
    comp = load_comparison()

    model_entries = get_models()

    for key, name, technique, backbone, feat_dim in model_entries:
        # Skip if already evaluated
        if key in comp and comp[key].get("accuracy", 0.0) > 0.0:
            print(f"[SKIP] {name} already in comparison.json (acc={comp[key]['accuracy']})")
            continue

        print(f"\n[EVAL] {name}  ({technique})")
        backbone = backbone.to(DEVICE)
        for p in backbone.parameters():
            p.requires_grad = False

        t0 = time.time()
        feats = extract_features(backbone, val_x)
        print(f"       Features: {feats.shape}  ({time.time()-t0:.1f}s)")

        t1 = time.time()
        acc = linear_probe(feats, val_y)
        elapsed = time.time() - t0

        print(f"       Accuracy: {acc*100:.2f}%  (probe: {time.time()-t1:.1f}s)")

        comp[key] = {
            "accuracy":      acc,
            "training_time": round(elapsed, 1),
            "epochs":        0,
            "technique":     technique,
            "display_name":  name,
        }
        save_comparison(comp)
        print(f"       Saved → comparison.json")

        # Free GPU memory
        backbone.cpu()
        del backbone, feats
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "="*55)
    print(" COMPARISON RESULTS")
    print("="*55)
    comp = load_comparison()
    rows = [(k, v) for k, v in comp.items() if isinstance(v, dict) and "accuracy" in v]
    rows.sort(key=lambda x: x[1]["accuracy"], reverse=True)
    for k, v in rows:
        name = v.get("display_name", k)
        tech = v.get("technique", "")
        print(f"  {v['accuracy']*100:5.2f}%  {name:<35}  {tech}")
    print("="*55)


if __name__ == "__main__":
    main()
