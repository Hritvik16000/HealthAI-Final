from pathlib import Path
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "artifacts" / "cnn"
REPORTS = BASE_DIR / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def make_image(label, size=64):
    img = Image.new("L", (size, size), color=10)
    draw = ImageDraw.Draw(img)
    n_blobs = label + 1
    for _ in range(n_blobs):
        x1 = random.randint(5, 40)
        y1 = random.randint(5, 40)
        r = random.randint(8, 18)
        intensity = 120 + label * 40 + random.randint(-10, 10)
        draw.ellipse((x1, y1, x1+r, y1+r), fill=intensity)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr

class SyntheticMedicalDataset(Dataset):
    def __init__(self, n=300):
        self.X = []
        self.y = []
        for label in [0, 1, 2]:
            for _ in range(n // 3):
                self.X.append(make_image(label))
                self.y.append(label)
        self.X = np.array(self.X, dtype=np.float32)[:, None, :, :]
        self.y = np.array(self.y, dtype=np.int64)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

ds = SyntheticMedicalDataset(300)
train_len = int(0.8 * len(ds))
test_len = len(ds) - train_len
train_ds, test_ds = random_split(ds, [train_len, test_len])

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=32)

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 64), nn.ReLU(),
            nn.Linear(64, 3)
        )
    def forward(self, x):
        return self.net(x)

model = SmallCNN()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for _ in range(5):
    model.train()
    for xb, yb in train_dl:
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for xb, yb in test_dl:
        pred = model(xb).argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += len(yb)

acc = correct / total if total else 0.0
torch.save(model.state_dict(), OUT_DIR / "cnn_model.pt")
pd.DataFrame([{"module": "cnn_imaging_demo", "accuracy": round(float(acc), 4)}]).to_csv(
    REPORTS / "cnn_metrics.csv", index=False
)

print("[OK] Saved cnn_model.pt")
print("Accuracy:", round(float(acc), 4))
