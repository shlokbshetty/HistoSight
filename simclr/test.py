import torch
import h5py
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_X = "../../Downloads/ssl_splits/ssl_splits/pcam_test_x.h5"
TEST_Y = "../../Downloads/ssl_splits/ssl_splits/pcam_test_y.h5"

CHECKPOINT = "./ssl_splits/simclr_checkpoint.pth"
CLASSIFIER_PATH = "./ssl_splits/linear_classifier.pth"

# ======================
# DATASET
# ======================
class PCamDataset(Dataset):
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
# MODEL
# ======================
encoder = models.resnet18(weights=None)
encoder.fc = nn.Identity()
encoder = encoder.to(DEVICE)

checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
encoder.load_state_dict(checkpoint['model'], strict=False)

classifier = nn.Linear(512, 2).to(DEVICE)
classifier.load_state_dict(torch.load(CLASSIFIER_PATH))

encoder.eval()
classifier.eval()

# ======================
# TEST
# ======================
dataset = PCamDataset(TEST_X, TEST_Y, transform)
loader = DataLoader(dataset, batch_size=32)

correct = 0
total = 0

with torch.no_grad():
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        features = encoder(x)
        preds = classifier(features)

        _, predicted = torch.max(preds, 1)

        total += y.size(0)
        correct += (predicted == y).sum().item()

print("Test Accuracy:", correct / total)