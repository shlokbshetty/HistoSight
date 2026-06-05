# =============================================================================
# fine_tune.py — Linear Evaluation on Labeled PCam Data
#
# Loads data into RAM (sequential h5 slice) for fast iteration.
# Trains nn.Linear(512, 2) on top of frozen SimCLR encoder.
# =============================================================================

import os, sys, json, time, argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

# ======================
# PATHS & CONFIG
# ======================
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_DIR         = os.path.join(BASE_DIR, "simclr", "ssl_splits")
CKPT_DIR         = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR          = os.path.join(BASE_DIR, "logs")
MODELS_DIR       = os.path.join(BASE_DIR, "models")
BEST_SIMCLR      = os.path.join(CKPT_DIR, "best_simclr.pth")
BEST_CLASSIFIER  = os.path.join(MODELS_DIR, "best_classifier.pth")
COMPARISON_PATH  = os.path.join(LOG_DIR, "comparison.json")
FINETUNE_METRICS = os.path.join(LOG_DIR, "finetune_metrics.json")

TRAIN_X = os.path.join(DATA_DIR, "pcam_ssl_val_labeled_x.h5")
TRAIN_Y = os.path.join(DATA_DIR, "pcam_ssl_val_labeled_y.h5")
TEST_X  = os.path.join(DATA_DIR, "pcam_ssl_test_labeled_x.h5")
TEST_Y  = os.path.join(DATA_DIR, "pcam_ssl_test_labeled_y.h5")

EPOCHS       = 30
BATCH_SIZE   = 256   # large batch, data in RAM
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 8
MAX_TRAIN    = 20000
MAX_TEST     = 8000

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {DEVICE}")

# ======================
# DATA LOADING — into RAM (sequential slice = fast)
# ======================
def load_h5_to_ram(x_path, y_path, max_samples=None):
    print(f"[DATA] Loading {os.path.basename(x_path)} + labels into RAM…", end=" ", flush=True)
    with h5py.File(x_path, 'r') as f:
        n = min(len(f['x']), max_samples) if max_samples else len(f['x'])
        x = f['x'][:n]
    with h5py.File(y_path, 'r') as f:
        y = f['y'][:n].squeeze().astype(np.int64)
    print(f"done. {n} samples")
    return x, y

# ======================
# TRANSFORMS
# ======================
TRAIN_TRANSFORM = T.Compose([
    T.Resize((96, 96)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.6),
    T.ToTensor(),
    T.Normalize(mean=[0.7011, 0.5859, 0.7054],
                std=[0.2128, 0.2438, 0.1955]),
])

VAL_TRANSFORM = T.Compose([
    T.Resize((96, 96)),
    T.ToTensor(),
    T.Normalize(mean=[0.7011, 0.5859, 0.7054],
                std=[0.2128, 0.2438, 0.1955]),
])

# ======================
# DATASET — RAM-backed
# ======================
class RamLabelDataset(Dataset):
    def __init__(self, x_np, y_np, transform):
        self.x = x_np
        self.y = y_np
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = Image.fromarray(self.x[idx])
        return self.transform(img), torch.tensor(int(self.y[idx]), dtype=torch.long)

# ======================
# ENCODER (FROZEN)
# ======================
def load_encoder():
    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Identity()
    encoder = backbone.to(DEVICE)

    if not os.path.exists(BEST_SIMCLR):
        print(f"[ERROR] Checkpoint not found: {BEST_SIMCLR}")
        sys.exit(1)

    ckpt = torch.load(BEST_SIMCLR, map_location=DEVICE)
    model_state = ckpt.get('model_state', ckpt.get('model'))
    encoder_state = {
        k.replace("encoder.", ""): v
        for k, v in model_state.items()
        if k.startswith("encoder.")
    }
    encoder.load_state_dict(encoder_state, strict=True)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    print("[INFO] Frozen encoder loaded.")
    return encoder

# ======================
# FEATURE EXTRACTION — pre-compute all features once
# ======================
def extract_features(encoder, x_np, transform, batch_size=512):
    """
    Pre-compute encoder features for the entire dataset once.
    This is much faster than running the encoder every batch during training.
    """
    print("[INFO] Extracting features…", end=" ", flush=True)
    all_feats = []
    encoder.eval()
    with torch.no_grad():
        for i in range(0, len(x_np), batch_size):
            batch_np = x_np[i:i+batch_size]
            imgs = torch.stack([transform(Image.fromarray(b)) for b in batch_np]).to(DEVICE)
            with autocast(enabled=(DEVICE == "cuda")):
                feats = encoder(imgs)
            all_feats.append(feats.float().cpu())
    feats = torch.cat(all_feats)
    print(f"done. shape={feats.shape}")
    return feats

# ======================
# FEATURE DATASET
# ======================
class FeatureDataset(Dataset):
    """Dataset of pre-extracted features + labels. No GPU needed per batch."""
    def __init__(self, features, labels):
        self.features = features   # (N, 512) CPU tensor
        self.labels   = labels     # (N,) int64 numpy

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], torch.tensor(int(self.labels[idx]), dtype=torch.long)

# ======================
# JSON HELPERS
# ======================
def load_comparison():
    if os.path.exists(COMPARISON_PATH):
        with open(COMPARISON_PATH) as f:
            return json.load(f)
    return {
        "resnet_pretrained": {"accuracy": 0.0, "training_time": 0, "epochs": 0},
        "simclr_ssl":        {"accuracy": 0.0, "training_time": 0, "epochs": 0},
        "final_model":       {"accuracy": 0.0, "training_time": 0, "epochs": 0},
    }

def save_json(data, path):
    tmp = path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

# ======================
# MAIN
# ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int,   default=EPOCHS)
    parser.add_argument("--lr",     type=float, default=LR)
    args = parser.parse_args()

    # ── Load data into RAM ──────────────────────────────────────────────────
    train_x, train_y = load_h5_to_ram(TRAIN_X, TRAIN_Y, MAX_TRAIN)
    test_x,  test_y  = load_h5_to_ram(TEST_X,  TEST_Y,  MAX_TEST)

    # ── Load encoder ────────────────────────────────────────────────────────
    encoder = load_encoder()

    # ── Pre-extract features (run encoder once, not every batch) ────────────
    train_feats = extract_features(encoder, train_x, VAL_TRANSFORM)
    test_feats  = extract_features(encoder, test_x,  VAL_TRANSFORM)

    train_set = FeatureDataset(train_feats, train_y)
    test_set  = FeatureDataset(test_feats,  test_y)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=False)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    print(f"[INFO] Train batches/epoch: {len(train_loader)}")

    # ── Classifier ──────────────────────────────────────────────────────────
    classifier = nn.Linear(512, 2).to(DEVICE)
    optimizer  = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion  = nn.CrossEntropyLoss()

    ft_metrics   = {"epoch": [], "train_loss": [], "val_loss": [],
                    "train_accuracy": [], "val_accuracy": [], "epoch_time": []}
    best_val_acc = 0.0
    no_improve   = 0
    total_time   = 0.0

    print(f"\n{'='*55}")
    print(f" Linear Evaluation  |  Epochs: {args.epochs}  |  LR: {args.lr}")
    print(f"{'='*55}\n")

    for epoch in range(args.epochs):
        t0 = time.time()
        classifier.train()
        total_loss = correct = total = 0

        for feats, labels in tqdm(train_loader, desc=f"Epoch {epoch+1:02d}", leave=False, ncols=80):
            feats  = feats.to(DEVICE,  non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            logits = classifier(feats)
            loss   = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        scheduler.step()
        epoch_time  = time.time() - t0
        total_time += epoch_time

        train_loss = total_loss / total
        train_acc  = correct / total

        # Eval
        classifier.eval()
        val_loss_sum = val_correct = val_total = 0
        with torch.no_grad():
            for feats, labels in test_loader:
                feats  = feats.to(DEVICE)
                labels = labels.to(DEVICE)
                logits = classifier(feats)
                loss   = criterion(logits, labels)
                val_loss_sum += loss.item() * labels.size(0)
                val_correct  += (logits.argmax(1) == labels).sum().item()
                val_total    += labels.size(0)
        val_loss = val_loss_sum / val_total
        val_acc  = val_correct  / val_total

        print(f"[{epoch+1:02d}/{args.epochs}] "
              f"train {train_loss:.4f}/{train_acc:.4f}  "
              f"val {val_loss:.4f}/{val_acc:.4f}  "
              f"t {epoch_time:.1f}s")

        ft_metrics["epoch"].append(epoch + 1)
        ft_metrics["train_loss"].append(round(train_loss, 6))
        ft_metrics["val_loss"].append(round(val_loss,   6))
        ft_metrics["train_accuracy"].append(round(train_acc, 6))
        ft_metrics["val_accuracy"].append(round(val_acc,   6))
        ft_metrics["epoch_time"].append(round(epoch_time, 2))
        save_json(ft_metrics, FINETUNE_METRICS)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
            torch.save({"classifier_state": classifier.state_dict(),
                        "epoch": epoch, "val_acc": val_acc}, BEST_CLASSIFIER)
            print(f"  ↳ [BEST] val_acc={best_val_acc:.4f}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"[EarlyStopping] epoch {epoch+1}")
                break

    # ── Update comparison.json ───────────────────────────────────────────────
    comp = load_comparison()
    comp["final_model"] = {
        "accuracy":      round(best_val_acc, 4),
        "training_time": round(total_time, 1),
        "epochs":        epoch + 1,
    }
    save_json(comp, COMPARISON_PATH)

    print(f"\n[DONE] Best val accuracy : {best_val_acc*100:.2f}%")
    print(f"[DONE] Classifier saved  → {BEST_CLASSIFIER}")
    print(f"[DONE] comparison.json   → {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
