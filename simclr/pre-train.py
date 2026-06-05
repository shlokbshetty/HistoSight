# ======================
# 1. IMPORTS
# ======================
import os
from itertools import islice
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from PIL import Image

# ======================
# 2. CONFIG
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE = "./ssl_splits/"
SSL_SAVE = BASE + "simclr_checkpoint.pth"
TEST_FILE = BASE + "test_write.txt"

BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 1
SAVE_EVERY = 10

os.makedirs(BASE, exist_ok=True)

print("Using device:", DEVICE)

# ======================
# 3. TEST CASES (SAFE)
# ======================
with open(TEST_FILE, "w") as f:
    f.write("drive working")

print("Write test:", os.path.exists(TEST_FILE))

print("Files in folder:")
print(os.listdir(BASE))

# ======================
# 4. DATASET
# ======================
class PCamDataset(Dataset):
    def __init__(self, x_path, transform=None):
        self.x_path = x_path
        self.transform = transform

    def __len__(self):
        with h5py.File(self.x_path, 'r') as f:
            return len(f['x'])

    def __getitem__(self, idx):
        with h5py.File(self.x_path, 'r') as f:
            img = f['x'][idx]

        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img

simclr_transform = T.Compose([
    T.RandomResizedCrop(96),
    T.RandomHorizontalFlip(),
    T.ToTensor()
])

class SimCLRTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)

# ======================
# 5. MODEL
# ======================
class SimCLRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet18(weights=None)
        self.encoder.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return z

def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # 2N x D

    sim = torch.matmul(z, z.T) / temperature

    # remove self similarity
    mask = torch.eye(2 * N, dtype=torch.bool).to(z.device)
    sim = sim.masked_fill(mask, -1e9)

    # positive pairs
    positives = torch.cat([
        torch.diag(sim, N),
        torch.diag(sim, -N)
    ], dim=0)

    exp_sim = torch.exp(sim)
    denom = exp_sim.sum(dim=1)

    loss = -torch.log(torch.exp(positives) / denom)

    return loss.mean()

# ======================
# 6. TRAIN FUNCTION
# ======================
def train_simclr(model, loader, optimizer, device, save_path, start_step=0, start_epoch=0, start_batch=0):
    model.train()
    step = start_step

    for epoch in range(EPOCHS):
        if epoch < start_epoch:
            continue

        total_loss = 0

        batch_start = start_batch if epoch == start_epoch else 0
        progress = tqdm(islice(loader, batch_start, None), initial=batch_start, total=len(loader))

        for batch_idx, (x1, x2) in enumerate(progress, start=batch_start):
            x1, x2 = x1.to(device), x2.to(device)

            z1 = model(x1)
            z2 = model(x2)

            loss = nt_xent_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1
            if step % 10 == 0:
                print(f"Step {step} | Loss: {loss.item():.4f}")

            if step % SAVE_EVERY == 0:
                temp_path = save_path + ".tmp"

                torch.save({
                    'step': step,
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, temp_path)

                os.replace(temp_path, save_path)

                print("Saved at step", step)

        print("Epoch", epoch, "Loss:", total_loss / len(loader))

    temp_path = save_path + ".tmp"

    torch.save({
        'step': step,
        'epoch': EPOCHS - 1,
        'batch_idx': len(loader) - 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, temp_path)

    os.replace(temp_path, save_path)

    print("Final checkpoint saved")

# ======================
# 7. MAIN EXECUTION
# ======================
def main():
    train_path = "../../Downloads/ssl_splits/ssl_splits/pcam_ssl_train_unlabeled_x.h5"

    print("Dataset exists:", os.path.exists(train_path))

    dataset = PCamDataset(train_path, transform=SimCLRTransform(simclr_transform))

    indices = torch.randperm(len(dataset))[:30000]
    subset = Subset(dataset, indices)

    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = SimCLRModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # RESUME
    start_step = 0
    start_epoch = 0
    start_batch = 0

    if os.path.exists(SSL_SAVE):
        print("Loading checkpoint...")
        checkpoint = torch.load(SSL_SAVE, map_location=DEVICE)

        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            start_step = checkpoint.get('step', 0)
            start_epoch = checkpoint.get('epoch', 0)
            start_batch = checkpoint.get('batch_idx', checkpoint.get('step', 0) - 1) + 1
            print("Resumed from step", start_step)

    train_simclr(model, loader, optimizer, DEVICE, SSL_SAVE, start_step, start_epoch, start_batch)

    print("Checkpoint exists:", os.path.exists(SSL_SAVE))
    print(os.listdir(BASE))


if __name__ == "__main__":
    main()