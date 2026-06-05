import os
import h5py
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT = "./ssl_splits/simclr_checkpoint.pth"
TRAIN_PATH = "../../Downloads/ssl_splits/ssl_splits/pcam_train_x.h5"
LABEL_PATH = "../../Downloads/ssl_splits/ssl_splits/pcam_train_y.h5"

BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3

# ======================
# DATASET WITH LABELS
# ======================
class PCamLabelDataset(Dataset):
    def __init__(self, x_path, y_path, transform=None):
        self.x_path = x_path
        self.y_path = y_path
        self.transform = transform

    def __len__(self):
        with h5py.File(self.x_path, 'r') as f:
            return len(f['x'])

    def __getitem__(self, idx):
        with h5py.File(self.x_path, 'r') as fx, h5py.File(self.y_path, 'r') as fy:
            img = fx['x'][idx]
            label = fy['y'][idx]

        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label).long().squeeze()


transform = T.Compose([
    T.Resize((96, 96)),
    T.ToTensor()
])

# ======================
# LOAD PRETRAINED MODEL
# ======================
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)


encoder = Encoder().to(DEVICE)

checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
encoder.load_state_dict(checkpoint['model'], strict=False)

# FREEZE encoder
for param in encoder.parameters():
    param.requires_grad = False

# LINEAR CLASSIFIER
classifier = nn.Linear(512, 2).to(DEVICE)

# ======================
# TRAIN
# ======================
dataset = PCamLabelDataset(TRAIN_PATH, LABEL_PATH, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    total_loss = 0

    for x, y in tqdm(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)

        with torch.no_grad():
            features = encoder(x)

        preds = classifier(features)
        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss/len(loader)}")

torch.save(classifier.state_dict(), "./ssl_splits/linear_classifier.pth")
print("Linear evaluation training done")