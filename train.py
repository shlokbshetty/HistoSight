# =============================================================================
# train.py — SimCLR SSL Pre-Training with Stain Normalization
#
# Pre-trains a ResNet18 encoder using contrastive learning.
# Supports GPU acceleration, mixed precision, and metrics logging.
# =============================================================================

import os
import sys
import json
import time
import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from PIL import Image

# ======================
# CONFIG & PATHS
# ======================
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "simclr", "ssl_splits")
CKPT_DIR        = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR         = os.path.join(BASE_DIR, "logs")
BEST_SIMCLR     = os.path.join(CKPT_DIR, "best_simclr.pth")
METRICS_PATH    = os.path.join(LOG_DIR, "metrics.json")
COMPARISON_PATH = os.path.join(LOG_DIR, "comparison.json")

# Default Hyperparameters
EPOCHS       = 30
BATCH_SIZE   = 256
LR           = 1e-3
TEMP         = 0.5
WEIGHT_DECAY = 1e-4

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {DEVICE}")

# ======================
# DATASET
# ======================
class PCamDataset(Dataset):
    def __init__(self, x_path, transform=None):
        self.x_path = x_path
        self.transform = transform
        with h5py.File(self.x_path, 'r') as f:
            self.length = len(f['x'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.x_path, 'r') as f:
            img = f['x'][idx]
        
        img = Image.fromarray(img.astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        return img

# SimCLR Augmentations (Contrastive Learning)
simclr_transform = T.Compose([
    T.RandomResizedCrop(96, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    # Less color jitter is needed when we normalize stains, but keep a subtle amount for generalization
    T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.4),
    T.ToTensor()
])

class SimCLRTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        # Return two augmented views of the same image
        return self.base_transform(x), self.base_transform(x)

# ======================
# MODEL
# ======================
class SimCLRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet18(weights=None)
        self.encoder.fc = nn.Identity()

        # Projection head: maps representations to 128-dim space where contrastive loss is applied
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return z

# NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss
def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # 2N x D

    sim = torch.matmul(z, z.T) / temperature
    # Remove self-similarity by filling diagonal with negative infinity
    mask = torch.eye(2 * N, dtype=torch.bool).to(z.device)
    sim = sim.masked_fill(mask, -1e4)

    # positive pairs are at index (i + N) or (i - N)
    targets = torch.arange(2 * N).to(z.device)
    targets[:N] += N
    targets[N:] -= N

    loss = F.cross_entropy(sim, targets)

    # Compute top-1 contrastive accuracy
    preds = sim.argmax(dim=1)
    correct = (preds == targets).sum().item()
    acc = correct / (2 * N)

    return loss, acc

# ======================
# JSON HELPERS
# ======================
def load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default

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
    parser.add_argument("--scratch", action="store_true",
                        help="Start training from scratch, ignoring existing checkpoints")
    args = parser.parse_args()

    # ── Determine Paths (Prefer pre-normalized datasets) ──────────────────
    train_x_normalized = os.path.join(DATA_DIR, "pcam_ssl_train_unlabeled_x_normalized.h5")
    train_x_raw        = os.path.join(DATA_DIR, "pcam_ssl_train_unlabeled_x.h5")
    val_x_normalized   = os.path.join(DATA_DIR, "pcam_ssl_val_labeled_x_normalized.h5")
    val_x_raw          = os.path.join(DATA_DIR, "pcam_ssl_val_labeled_x.h5")

    train_path = train_x_normalized if os.path.exists(train_x_normalized) else train_x_raw
    val_path   = val_x_normalized if os.path.exists(val_x_normalized) else val_x_raw

    # Check if train_path is readable, fallback to val_path if not
    train_path_ok = False
    if os.path.exists(train_path):
        try:
            with h5py.File(train_path, 'r') as f:
                _ = len(f['x'])
            train_path_ok = True
        except Exception as e:
            print(f"[WARNING] Train path {train_path} is corrupted or unreadable: {e}")

    if not train_path_ok:
        print(f"[FALLBACK] Falling back to val dataset {val_path} for pre-training.")
        train_path = val_path

    is_macenko = "normalized" in train_path
    print(f"[DATA] Train path: {train_path}  (Macenko: {is_macenko})")
    print(f"[DATA] Val path:   {val_path}  (Macenko: {is_macenko})")

    # ── Load Dataloaders ────────────────────────────────────────────────────
    train_set = PCamDataset(train_path, transform=SimCLRTransform(simclr_transform))
    val_set   = PCamDataset(val_path,   transform=SimCLRTransform(simclr_transform))

    # Subset datasets for consistent execution speed
    train_indices = torch.randperm(len(train_set))[:20000]
    val_indices   = torch.randperm(len(val_set))[:8000]

    train_subset = Subset(train_set, train_indices)
    val_subset   = Subset(val_set,   val_indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=True)
    val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False, drop_last=True)

    print(f"[INFO] Train batches/epoch: {len(train_loader)}")

    # ── Model & Optimization ────────────────────────────────────────────────
    model     = SimCLRModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    # For newer PyTorch versions, use the standard torch.amp.GradScaler
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == "cuda"))
    except (AttributeError, TypeError):
        scaler = GradScaler(enabled=(DEVICE == "cuda"))

    # Resume Training if best checkpoint exists (and not training from scratch)
    start_epoch = 0
    total_time  = 0.0
    metrics = {"epoch": [], "train_loss": [], "val_loss": [],
               "train_accuracy": [], "val_accuracy": [], "epoch_time": []}

    if os.path.exists(BEST_SIMCLR) and not args.scratch:
        print("[INFO] Resuming pre-training from best checkpoint...")
        ckpt = torch.load(BEST_SIMCLR, map_location=DEVICE)
        model.load_state_dict(ckpt.get('model_state', ckpt.get('model', {})))
        optimizer.load_state_dict(ckpt.get('optimizer_state', ckpt.get('optimizer', {})))
        start_epoch = ckpt.get('epoch', 0) + 1
        metrics = load_json(METRICS_PATH, metrics)
        total_time = ckpt.get('training_time', 0.0)
        print(f"[INFO] Resuming at epoch {start_epoch + 1}")
    elif args.scratch:
        print("[INFO] --scratch specified. Starting training from epoch 1.")

    avg_val_acc = metrics["val_accuracy"][-1] if len(metrics["val_accuracy"]) > 0 else 0.0

    print(f"\n{'='*55}")
    print(f" SimCLR SSL Training  |  Epochs: {args.epochs}  |  LR: {args.lr}")
    print(f"{'='*55}\n")

    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        
        train_loss_sum = train_acc_sum = 0.0
        train_batches  = 0

        for x1, x2 in tqdm(train_loader, desc=f"Epoch {epoch+1:02d}", leave=False, ncols=80):
            x1 = x1.to(DEVICE, non_blocking=True)
            x2 = x2.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(DEVICE == "cuda")):
                z1 = model(x1)
                z2 = model(x2)
                loss, acc = nt_xent_loss(z1, z2, temperature=TEMP)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            train_acc_sum  += acc
            train_batches  += 1

        scheduler.step()
        epoch_time  = time.time() - t0
        total_time += epoch_time

        avg_train_loss = train_loss_sum / train_batches
        avg_train_acc  = train_acc_sum  / train_batches

        # Contrastive Validation
        model.eval()
        val_loss_sum = val_acc_sum = 0.0
        val_batches  = 0

        with torch.no_grad():
            for x1, x2 in val_loader:
                x1 = x1.to(DEVICE)
                x2 = x2.to(DEVICE)
                with autocast(enabled=(DEVICE == "cuda")):
                    z1 = model(x1)
                    z2 = model(x2)
                    loss, acc = nt_xent_loss(z1, z2, temperature=TEMP)
                val_loss_sum += loss.item()
                val_acc_sum  += acc
                val_batches  += 1

        avg_val_loss = val_loss_sum / val_batches
        avg_val_acc  = val_acc_sum  / val_batches

        print(f"[{epoch+1:02d}/{args.epochs}] "
              f"loss: train {avg_train_loss:.4f} / val {avg_val_loss:.4f}  "
              f"acc: train {avg_train_acc*100:.2f}% / val {avg_val_acc*100:.2f}%  "
              f"t {epoch_time:.1f}s")

        # Save Metrics
        metrics["epoch"].append(epoch + 1)
        metrics["train_loss"].append(round(avg_train_loss, 6))
        metrics["val_loss"].append(round(avg_val_loss, 6))
        metrics["train_accuracy"].append(round(avg_train_acc, 6))
        metrics["val_accuracy"].append(round(avg_val_acc, 6))
        metrics["epoch_time"].append(round(epoch_time, 2))
        save_json(metrics, METRICS_PATH)

        # Save Best Checkpoint (based on validation loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'training_time': total_time,
                'val_loss': avg_val_loss
            }, BEST_SIMCLR)
            print(f"  -> [BEST Checkpoint] val_loss={best_val_loss:.4f}")

    # ── Update comparison.json ───────────────────────────────────────────────
    comp = load_json(COMPARISON_PATH, {})
    comp["simclr_ssl"] = {
        "accuracy":      round(avg_val_acc, 4), # final validation accuracy
        "training_time": round(total_time, 1),
        "epochs":        args.epochs,
        "macenko":       is_macenko
    }
    save_json(comp, COMPARISON_PATH)

    print(f"\n[DONE] Pre-training complete!")
    print(f"[DONE] Checkpoint saved -> {BEST_SIMCLR}")
    print(f"[DONE] metrics.json      -> {METRICS_PATH}")
    print(f"[DONE] comparison.json   -> {COMPARISON_PATH}")

if __name__ == "__main__":
    main()
