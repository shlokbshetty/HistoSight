# =============================================================================
# evaluate.py — Test Phase Evaluation
#
# Outputs ONLY: Accuracy, Loss (to console)
# No graphs. No JSON output.
# =============================================================================

import os
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

# ======================
# PATHS
# ======================
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "simclr", "ssl_splits")
CKPT_DIR        = os.path.join(BASE_DIR, "checkpoints")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
BEST_SIMCLR     = os.path.join(CKPT_DIR, "best_simclr.pth")
BEST_CLASSIFIER = os.path.join(MODELS_DIR, "best_classifier.pth")
BEST_ENCODER    = os.path.join(MODELS_DIR, "best_encoder.pth")
TEST_X_NORM     = os.path.join(DATA_DIR, "pcam_ssl_test_labeled_x_normalized.h5")
TEST_X_RAW      = os.path.join(DATA_DIR, "pcam_ssl_test_labeled_x.h5")
TEST_X          = TEST_X_NORM if os.path.exists(TEST_X_NORM) else TEST_X_RAW
TEST_Y          = os.path.join(DATA_DIR, "pcam_ssl_test_labeled_y.h5")

BATCH_SIZE = 64
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {DEVICE}")

# ======================
# DATASET
# ======================
class PCamTestDataset(Dataset):
    def __init__(self, x_path, y_path, transform=None):
        self.x_path    = x_path
        self.y_path    = y_path
        self.transform = transform
        with h5py.File(x_path, 'r') as f:
            self.length = len(f['x'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.x_path, 'r') as fx, h5py.File(self.y_path, 'r') as fy:
            img   = fx['x'][idx]
            label = fy['y'][idx]
        img   = Image.fromarray(img.astype(np.uint8))
        label = int(np.array(label).squeeze())
        
        if "normalized" not in self.x_path:
            import stain_norm
            img = stain_norm.normalize_image(img)

        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


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
        encoder_ckpt = torch.load(BEST_ENCODER, map_location=DEVICE)
        model_state = encoder_ckpt.get('model_state', encoder_ckpt.get('model', encoder_ckpt))
        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in model_state.items()
        }
        encoder.load_state_dict(encoder_state, strict=True)
    elif os.path.exists(BEST_SIMCLR):
        print(f"[INFO] Loading pre-trained SimCLR encoder from {BEST_SIMCLR}")
        simclr_ckpt = torch.load(BEST_SIMCLR, map_location=DEVICE)
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
    clf_ckpt   = torch.load(BEST_CLASSIFIER, map_location=DEVICE)
    clf_state  = clf_ckpt.get('classifier_state', clf_ckpt)
    classifier.load_state_dict(clf_state)
    classifier.eval()

    print("[INFO] Models loaded successfully.")
    return encoder, classifier

# ======================
# EVALUATE
# ======================
def main():
    for p in [TEST_X, TEST_Y]:
        if not os.path.exists(p):
            print(f"[ERROR] Test data not found: {p}")
            sys.exit(1)

    dataset = PCamTestDataset(TEST_X, TEST_Y, transform)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=True)

    encoder, classifier = load_models()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct    = 0
    total      = 0

    print(f"\nEvaluating on {len(dataset)} test samples...")

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Testing"):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with autocast(enabled=(DEVICE == "cuda")):
                features = encoder(x)
                logits   = classifier(features)
                loss     = criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == y).sum().item()
            total      += y.size(0)

    accuracy = correct / total
    avg_loss = total_loss / total

    print("\n" + "="*40)
    print(f"  Test Accuracy : {accuracy * 100:.2f}%")
    print(f"  Test Loss     : {avg_loss:.4f}")
    print("="*40)


if __name__ == "__main__":
    main()
