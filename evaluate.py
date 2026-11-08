# =============================================================================
# evaluate.py — Test Phase Evaluation & Clinical Metrics
#
# Computes: Accuracy, Balanced Accuracy, Sensitivity, Specificity, F1, ROC-AUC,
# PR-AUC, Confusion Matrix, and Cross-Entropy Loss on PCam test splits.
# Supports 8-fold dihedral Test-Time Augmentation (D4 group).
# =============================================================================

import os
import sys
import json
import argparse
from typing import Optional, Dict, Any, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

try:
    from sklearn import metrics as sk_metrics
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ======================
# PATHS
# ======================
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "simclr", "ssl_splits")
CKPT_DIR        = os.path.join(BASE_DIR, "checkpoints")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
LOG_DIR         = os.path.join(BASE_DIR, "logs")
BEST_SIMCLR     = os.path.join(CKPT_DIR, "best_simclr.pth")
BEST_CLASSIFIER = os.path.join(MODELS_DIR, "best_classifier.pth")
BEST_ENCODER    = os.path.join(MODELS_DIR, "best_encoder.pth")
TEST_X_NORM     = os.path.join(DATA_DIR, "pcam_ssl_test_labeled_x_normalized.h5")
TEST_X_RAW      = os.path.join(DATA_DIR, "pcam_ssl_test_labeled_x.h5")
TEST_X          = TEST_X_NORM if os.path.exists(TEST_X_NORM) else TEST_X_RAW
TEST_Y          = os.path.join(DATA_DIR, "pcam_ssl_test_labeled_y.h5")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# DATASET
# ======================
class PCamTestDataset(Dataset):
    def __init__(self, x_path: str, y_path: str, transform=None, max_samples: Optional[int] = None, in_memory: bool = False):
        self.x_path    = x_path
        self.y_path    = y_path
        self.transform = transform
        self.in_memory = in_memory

        with h5py.File(x_path, 'r') as fx, h5py.File(y_path, 'r') as fy:
            total_available = len(fx['x'])
            self.length = min(total_available, max_samples) if max_samples else total_available
            if self.in_memory:
                print(f"[INFO] Pre-loading {self.length} test patches into RAM from {os.path.basename(x_path)}...")
                self.x_data = fx['x'][:self.length]
                self.y_data = np.array(fy['y'][:self.length]).squeeze()
            else:
                self.x_data = None
                self.y_data = None

        self.fx = None
        self.fy = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.in_memory:
            img_arr = self.x_data[idx]
            label   = int(self.y_data[idx])
        else:
            if self.fx is None:
                self.fx = h5py.File(self.x_path, 'r')
                self.fy = h5py.File(self.y_path, 'r')
            img_arr = self.fx['x'][idx]
            label   = int(np.array(self.fy['y'][idx]).squeeze())

        img = Image.fromarray(img_arr.astype(np.uint8))

        if "normalized" not in self.x_path:
            import stain_norm
            img = stain_norm.normalize_image(img)

        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

    def close(self):
        if self.fx is not None:
            self.fx.close()
            self.fx = None
        if self.fy is not None:
            self.fy.close()
            self.fy = None


transform = T.Compose([
    T.Resize((96, 96)),
    T.ToTensor(),
    T.Normalize(mean=[0.7011, 0.5859, 0.7054],
                std=[0.2128, 0.2438, 0.1955]),
])

# ======================
# MODEL LOADING
# ======================
def load_models():
    if not os.path.exists(BEST_CLASSIFIER):
        print(f"[ERROR] Classifier checkpoint not found at {BEST_CLASSIFIER}")
        sys.exit(1)

    # Encoder
    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Identity()
    encoder = backbone.to(DEVICE)

    # Load either fine-tuned encoder or pre-trained SimCLR encoder
    if os.path.exists(BEST_ENCODER):
        print(f"[INFO] Loading fine-tuned encoder checkpoint from {BEST_ENCODER}")
        encoder_ckpt = torch.load(BEST_ENCODER, map_location=DEVICE, weights_only=True)
        model_state = encoder_ckpt.get('model_state', encoder_ckpt.get('model', encoder_ckpt))
        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in model_state.items()
        }
        encoder.load_state_dict(encoder_state, strict=True)
    elif os.path.exists(BEST_SIMCLR):
        print(f"[INFO] Loading pre-trained SimCLR encoder from {BEST_SIMCLR}")
        simclr_ckpt = torch.load(BEST_SIMCLR, map_location=DEVICE, weights_only=True)
        model_state = simclr_ckpt.get('model_state', simclr_ckpt.get('model', None))
        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in model_state.items()
            if k.startswith("encoder.")
        }
        encoder.load_state_dict(encoder_state, strict=True)
    else:
        print(f"[ERROR] Neither fine-tuned encoder ({BEST_ENCODER}) nor pre-trained SimCLR checkpoint ({BEST_SIMCLR}) was found.")
        sys.exit(1)

    encoder.eval()

    # Classifier
    classifier = nn.Linear(512, 2).to(DEVICE)
    clf_ckpt   = torch.load(BEST_CLASSIFIER, map_location=DEVICE, weights_only=True)
    clf_state  = clf_ckpt.get('classifier_state', clf_ckpt)
    classifier.load_state_dict(clf_state)
    classifier.eval()

    print("[INFO] Models loaded successfully.")
    return encoder, classifier


def apply_dihedral_tta(batch_tensor: torch.Tensor) -> torch.Tensor:
    """
    Generate 8 dihedral views (D4 group) for every item in batch_tensor (B, C, H, W).
    Returns (8*B, C, H, W).
    """
    views = []
    for flip in (False, True):
        x = torch.flip(batch_tensor, dims=[-1]) if flip else batch_tensor
        for k in range(4):
            views.append(torch.rot90(x, k, dims=(-2, -1)))
    return torch.cat(views, dim=0)


# ======================
# EVALUATE
# ======================
def evaluate(
    max_samples: Optional[int] = None,
    batch_size: int = 64,
    tta: bool = False,
    in_memory: bool = False,
    save_json: Optional[str] = None
) -> Dict[str, Any]:
    for p in [TEST_X, TEST_Y]:
        if not os.path.exists(p):
            print(f"[ERROR] Test data not found: {p}")
            sys.exit(1)

    dataset = PCamTestDataset(
        TEST_X, TEST_Y,
        transform=transform,
        max_samples=max_samples,
        in_memory=in_memory
    )
    num_workers = 0 if sys.platform.startswith("win") or in_memory else 2
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(DEVICE == "cuda")
    )

    encoder, classifier = load_models()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    device_type = "cuda" if DEVICE == "cuda" else "cpu"

    total_loss = 0.0
    all_targets = []
    all_preds   = []
    all_probs   = []

    print(f"\n[INFO] Evaluating on {len(dataset)} test samples (Device: {DEVICE}, Batch: {batch_size}, TTA: {tta})...")

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Testing" + (" (TTA 8x)" if tta else "")):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            b_size = x.size(0)

            with torch.amp.autocast(device_type=device_type, enabled=(DEVICE == "cuda")):
                if tta:
                    tta_x = apply_dihedral_tta(x)
                    tta_feats = encoder(tta_x)
                    tta_logits = classifier(tta_feats)
                    tta_probs = torch.softmax(tta_logits, dim=1)  # (8*B, 2)
                    tta_probs = tta_probs.view(8, b_size, 2).mean(dim=0)  # (B, 2)
                    probs = tta_probs
                    log_probs = torch.log(probs.clamp(min=1e-7))
                    loss = criterion(log_probs, y)
                else:
                    features = encoder(x)
                    logits   = classifier(features)
                    probs    = torch.softmax(logits, dim=1)
                    loss     = criterion(logits, y)

            total_loss += loss.item()
            preds = probs.argmax(dim=1)

            all_targets.extend(y.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    dataset.close()

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    total  = len(y_true)

    # Basic metrics
    correct  = int((y_pred == y_true).sum())
    accuracy = correct / total
    avg_loss = total_loss / total

    # Confusion matrix elements: [0: Normal, 1: Tumor]
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1          = (2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0
    bal_acc     = (sensitivity + specificity) / 2.0

    # Scikit-learn advanced metrics
    roc_auc = None
    pr_auc  = None
    if SKLEARN_AVAILABLE and len(np.unique(y_true)) == 2:
        try:
            roc_auc = float(sk_metrics.roc_auc_score(y_true, y_prob))
            pr_auc  = float(sk_metrics.average_precision_score(y_true, y_prob))
        except Exception:
            pass

    results = {
        "samples":            total,
        "tta_enabled":        tta,
        "accuracy":           round(accuracy, 4),
        "balanced_accuracy":  round(bal_acc, 4),
        "sensitivity_recall": round(sensitivity, 4),
        "specificity":        round(specificity, 4),
        "precision":          round(precision, 4),
        "f1_score":           round(f1, 4),
        "roc_auc":            round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc":             round(pr_auc, 4) if pr_auc is not None else None,
        "loss":               round(avg_loss, 4),
        "confusion_matrix": {
            "true_positive":  tp,
            "true_negative":  tn,
            "false_positive": fp,
            "false_negative": fn,
        }
    }

    print("\n" + "=" * 50)
    print("      HistoSight Clinical Evaluation Report")
    print("=" * 50)
    print(f"  Test Samples        : {total}")
    print(f"  Test-Time Aug (TTA) : {'Enabled (8x D4 Dihedral)' if tta else 'Disabled'}")
    print(f"  Accuracy            : {accuracy * 100:.2f}%")
    print(f"  Balanced Accuracy   : {bal_acc * 100:.2f}%")
    print(f"  Sensitivity (Recall): {sensitivity * 100:.2f}%")
    print(f"  Specificity         : {specificity * 100:.2f}%")
    print(f"  Precision (PPV)     : {precision * 100:.2f}%")
    print(f"  F1 Score            : {f1:.4f}")
    if roc_auc is not None:
        print(f"  ROC-AUC             : {roc_auc:.4f}")
    if pr_auc is not None:
        print(f"  PR-AUC              : {pr_auc:.4f}")
    print(f"  Test Loss           : {avg_loss:.4f}")
    print("-" * 50)
    print("  Confusion Matrix:")
    print(f"    [TN={tn:5d}  FP={fp:5d}]")
    print(f"    [FN={fn:5d}  TP={tp:5d}]")
    print("=" * 50)

    if save_json:
        os.makedirs(os.path.dirname(os.path.abspath(save_json)), exist_ok=True)
        with open(save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Saved test evaluation report to: {save_json}")

    return results


def main():
    parser = argparse.ArgumentParser(description="HistoSight Test Set Evaluation & Benchmarking")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of test patches to evaluate")
    parser.add_argument("--batch-size",  type=int, default=64, help="Batch size for DataLoader")
    parser.add_argument("--tta",         action="store_true", help="Enable 8-fold dihedral Test-Time Augmentation")
    parser.add_argument("--in-memory",   action="store_true", help="Pre-load test samples into RAM for maximum throughput")
    parser.add_argument("--save-json",   type=str, default=None, help="Path to write evaluation results JSON")
    args = parser.parse_args()

    evaluate(
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        tta=args.tta,
        in_memory=args.in_memory,
        save_json=args.save_json
    )


if __name__ == "__main__":
    main()

