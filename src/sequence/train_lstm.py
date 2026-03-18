from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import mean_absolute_error

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "artifacts" / "sequence"
REPORTS = BASE_DIR / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

def make_data(n=400, seq_len=10):
    X, y = [], []
    for _ in range(n):
        base = np.random.uniform(0.2, 0.8)
        trend = np.random.uniform(-0.03, 0.05)
        seq = np.array([base + trend*t + np.random.normal(0, 0.03) for t in range(seq_len+1)], dtype=np.float32)
        seq = np.clip(seq, 0, 1)
        X.append(seq[:-1].reshape(seq_len, 1))
        y.append(seq[-1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X, y = make_data()
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train).unsqueeze(1)
y_test = torch.tensor(y_test).unsqueeze(1)

class LSTMRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMRegressor()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for _ in range(30):
    model.train()
    opt.zero_grad()
    pred = model(X_train)
    loss = loss_fn(pred, y_train)
    loss.backward()
    opt.step()

model.eval()
with torch.no_grad():
    test_pred = model(X_test).squeeze(1).numpy()

mae = mean_absolute_error(y_test.squeeze(1).numpy(), test_pred)
torch.save(model.state_dict(), OUT_DIR / "lstm_model.pt")

pd.DataFrame([{"module": "lstm_forecast_demo", "mae": round(float(mae), 4)}]).to_csv(
    REPORTS / "lstm_metrics.csv", index=False
)

print("[OK] Saved lstm_model.pt")
print("MAE:", round(float(mae), 4))
