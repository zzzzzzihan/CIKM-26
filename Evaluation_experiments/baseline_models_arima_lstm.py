# QTFT_Step4_Baselines.py
# ARIMA + LSTM baselines (self-contained, no imports from other steps)

import math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA

# =========================================================
# Config
# =========================================================
TARGET = "gdp_growth"
HORIZON = 8
ENCODER_LEN = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DESKTOP = Path.home() / "Desktop"

# =========================================================
# Step 4.3: unified result container
# =========================================================
results = []

# =========================================================
# Unified evaluation (inlined to avoid import issues)
# =========================================================
def evaluate_multi_horizon(y_true, y_pred, horizons=(1, 4, 8)):
    metrics = {}
    for h in horizons:
        idx = h - 1
        t = y_true[:, idx]
        p = y_pred[:, idx]
        mse = torch.mean((p - t) ** 2).item()
        mae = torch.mean(torch.abs(p - t)).item()
        rmse = math.sqrt(mse)
        metrics[h] = {"RMSE": rmse, "MAE": mae}
    return metrics

# =========================================================
# Load data
# =========================================================
df = pd.read_csv(DESKTOP / "macro_quarterly_480.csv")
df["quarter"] = pd.to_datetime(df["quarter"])
df = df.sort_values(["country", "quarter"])

# =========================================================
# Build samples
# =========================================================
def build_samples(df):
    X, Y = [], []
    for _, g in df.groupby("country"):
        values = g[TARGET].values
        for i in range(len(values) - ENCODER_LEN - HORIZON):
            X.append(values[i : i + ENCODER_LEN])
            Y.append(values[i + ENCODER_LEN : i + ENCODER_LEN + HORIZON])
    return np.array(X), np.array(Y)

X, Y = build_samples(df)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

print("Test samples:", len(X_test))

Y_test_t = torch.tensor(Y_test, dtype=torch.float32)

# =========================================================
# 1. ARIMA baseline
# =========================================================
arima_preds = []

for x in X_test:
    try:
        model = ARIMA(x, order=(1, 1, 0))
        fit = model.fit()
        forecast = fit.forecast(steps=HORIZON)
    except Exception:
        forecast = np.repeat(x[-1], HORIZON)

    arima_preds.append(forecast)

arima_preds = torch.tensor(arima_preds, dtype=torch.float32)

arima_metrics = evaluate_multi_horizon(Y_test_t, arima_preds)

print("\nARIMA results:")
for h, m in arima_metrics.items():
    print(f"  {h}-step | RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}")

    # =========================
    # Step 4.4: save ARIMA results
    # =========================
    results.append({
        "country": "ALL",     # baselines are evaluated globally
        "model": "ARIMA",
        "horizon": h,
        "RMSE": m["RMSE"],
        "MAE": m["MAE"],
    })

# =========================================================
# 2. LSTM baseline
# =========================================================
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, batch_first=True)
        self.fc = nn.Linear(32, HORIZON)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

lstm = LSTMModel().to(DEVICE)
opt = torch.optim.Adam(lstm.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
Y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(DEVICE)

for epoch in range(20):
    opt.zero_grad()
    pred = lstm(X_train_t)
    loss = loss_fn(pred, Y_train_t)
    loss.backward()
    opt.step()

lstm.eval()
with torch.no_grad():
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    lstm_preds = lstm(X_test_t).cpu()

lstm_metrics = evaluate_multi_horizon(Y_test_t, lstm_preds)

print("\nLSTM results:")
for h, m in lstm_metrics.items():
    print(f"  {h}-step | RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}")

    # =========================
    # Step 4.5: save LSTM results
    # =========================
    results.append({
        "country": "ALL",
        "model": "LSTM",
        "horizon": h,
        "RMSE": m["RMSE"],
        "MAE": m["MAE"],
    })

# =========================================================
# Step 4.6: save baseline metrics to CSV
# =========================================================
df_results = pd.DataFrame(results)
csv_path = DESKTOP / "step4_metrics_baselines.csv"
df_results.to_csv(csv_path, index=False)

print(f"\n✅ Baseline metrics saved to: {csv_path}")
print(df_results)
