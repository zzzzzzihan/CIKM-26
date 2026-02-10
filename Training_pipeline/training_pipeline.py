# QTFT_Step3_Train.py
# Final stable version with publication-quality plots
# Python 3.13 compatible

import json
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from QTFT_Step2_Model import QuantumTFT

# =========================================================
# 0. Config
# =========================================================
MAX_ENCODER_LENGTH = 20
MAX_PREDICTION_LENGTH = 8
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 50
PATIENCE = 10

TARGET = "gdp_growth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DESKTOP = Path.home() / "Desktop"

# =========================================================
# 1. Load and preprocess data
# =========================================================
data_path = DESKTOP / "macro_quarterly_480.csv"
df = pd.read_csv(data_path)

# quarter: YYYYQ1 -> datetime
df["quarter"] = pd.to_datetime(df["quarter"])
df = df.sort_values(["country", "quarter"])
df["country"] = df["country"].astype(str)
df["time_idx"] = df.groupby("country").cumcount()

print("数据行数:", len(df))

# =========================================================
# 2. Build TimeSeriesDataSet
# =========================================================
training_cutoff = df["time_idx"].max() - MAX_PREDICTION_LENGTH

training = TimeSeriesDataSet(
    df[df.time_idx <= training_cutoff],
    time_idx="time_idx",
    target=TARGET,
    group_ids=["country"],
    min_encoder_length=MAX_ENCODER_LENGTH // 2,
    max_encoder_length=MAX_ENCODER_LENGTH,
    min_prediction_length=1,
    max_prediction_length=MAX_PREDICTION_LENGTH,
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=[
        "gdp_growth",
        "inflation",
        "unemployment",
    ],
    target_normalizer=GroupNormalizer(groups=["country"]),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

validation = TimeSeriesDataSet.from_dataset(
    training,
    df,
    predict=True,
    stop_randomization=True,
)

train_loader = training.to_dataloader(
    train=True, batch_size=BATCH_SIZE, num_workers=0
)
val_loader = validation.to_dataloader(
    train=False, batch_size=BATCH_SIZE, num_workers=0
)

# =========================================================
# 3. Build QuantumTFT model (matches Step2)
# =========================================================
tft_config = {
    "learning_rate": LR,
    "hidden_size": 16,
    "attention_head_size": 1,
    "dropout": 0.1,
    "hidden_continuous_size": 8,
    "output_size": 1,
    "loss": QuantileLoss(),
}

model = QuantumTFT(training, tft_config).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

print("模型构建完成")

# =========================================================
# 4. Training loop (shape-safe)
# =========================================================
def move(batch):
    x, y = batch
    x = {k: v.to(DEVICE) for k, v in x.items()}
    y = y[0].to(DEVICE)  # [B, H]
    return x, y


history = {"epoch": [], "train": [], "val": []}
best_val = float("inf")
patience = 0

best_model_path = DESKTOP / "qtft_best_model.pth"

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []

    for batch in train_loader:
        x, y = move(batch)
        optimizer.zero_grad()

        out = model(x).prediction      # [B, H, 1]
        y = y.unsqueeze(-1)            # [B, H, 1]
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            x, y = move(batch)
            out = model(x).prediction
            y = y.unsqueeze(-1)
            val_losses.append(criterion(out, y).item())

    train_loss = sum(train_losses) / len(train_losses)
    val_loss = sum(val_losses) / len(val_losses)

    history["epoch"].append(epoch)
    history["train"].append(train_loss)
    history["val"].append(val_loss)

    print(f"Epoch {epoch:03d} | Train {train_loss:.6f} | Val {val_loss:.6f}")

    if val_loss < best_val:
        best_val = val_loss
        patience = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        patience += 1
        if patience >= PATIENCE:
            print("Early stopping triggered")
            break

# =========================================================
# 5. Save history
# =========================================================
with open(DESKTOP / "qtft_training_history.json", "w") as f:
    json.dump(history, f, indent=2)

# =========================================================
# 6. Publication-quality training curve
# =========================================================
plt.figure(figsize=(7.5, 4.5))

plt.plot(
    history["epoch"],
    history["train"],
    label="Training Loss",
    linewidth=2.2,
    color="#1f77b4",
)

plt.plot(
    history["epoch"],
    history["val"],
    label="Validation Loss",
    linewidth=2.2,
    linestyle="--",
    color="#d62728",
)

best_epoch = history["epoch"][history["val"].index(min(history["val"]))]

plt.axvline(
    best_epoch,
    linestyle=":",
    linewidth=1.5,
    color="gray",
    alpha=0.8,
)

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("MSE Loss", fontsize=12)
plt.title("QTFT Training and Validation Loss", fontsize=14, pad=10)

plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

plt.legend(frameon=False, fontsize=11, loc="upper right")

plt.tight_layout()
plt.savefig(
    DESKTOP / "qtft_training_curve.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print("\n✅ Step 3 完整完成")
print("Saved to Desktop:")
print("- qtft_best_model.pth")
print("- qtft_training_history.json")
print("- qtft_training_curve.png")
