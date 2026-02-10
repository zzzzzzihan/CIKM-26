# QTFT_Step5_Explain_Attention.py
# Step 5.2: Temporal Attention Heatmap (FIXED & STABLE)

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from QTFT_Step2_Model import QuantumTFT

# =========================================================
# Config
# =========================================================
MAX_ENCODER_LENGTH = 20
MAX_PREDICTION_LENGTH = 8
BATCH_SIZE = 32
TARGET = "gdp_growth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DESKTOP = Path.home() / "Desktop"

# =========================================================
# Load data
# =========================================================
df = pd.read_csv(DESKTOP / "macro_quarterly_480.csv")
df["quarter"] = pd.to_datetime(df["quarter"])
df = df.sort_values(["country", "quarter"])
df["country"] = df["country"].astype(str)
df["time_idx"] = df.groupby("country").cumcount()

print("数据与模型准备完成")

# =========================================================
# Dataset
# =========================================================
training_cutoff = df["time_idx"].max() - MAX_PREDICTION_LENGTH

dataset = TimeSeriesDataSet(
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

loader = dataset.to_dataloader(
    train=False, batch_size=BATCH_SIZE, num_workers=0
)

# =========================================================
# Load trained QTFT
# =========================================================
model = QuantumTFT(
    dataset,
    {
        "hidden_size": 16,
        "attention_head_size": 1,
        "dropout": 0.1,
        "hidden_continuous_size": 8,
        "output_size": 1,
        "loss": QuantileLoss(),
    },
).to(DEVICE)

model.load_state_dict(
    torch.load(DESKTOP / "qtft_best_model.pth", map_location=DEVICE)
)
model.eval()

# =========================================================
# Collect attention weights
# =========================================================
attentions = []

with torch.no_grad():
    for x, _ in loader:
        x = {k: v.to(DEVICE) for k, v in x.items()}
        raw_out = model(x)
        interp = model.tft.interpret_output(raw_out, reduction="mean")

        att = interp["attention"]  # could be (encoder,) or (encoder,1)
        att = att.squeeze()        # -> (encoder_len,)
        attentions.append(att)

# Average over batches
attention_1d = torch.stack(attentions).mean(0).cpu().numpy()

# =========================================================
# Convert to 2D heatmap-friendly shape
# =========================================================
# Shape: (1, encoder_len) — one decoder horizon aggregated
attention_2d = attention_1d.reshape(1, -1)

# =========================================================
# Plot heatmap (Figure 5)
# =========================================================
plt.figure(figsize=(10, 2.5))

sns.heatmap(
    attention_2d,
    cmap="YlGnBu",
    cbar_kws={"label": "Attention weight"},
    xticklabels=list(range(-attention_2d.shape[1], 0)),
    yticklabels=["Prediction"]
)

plt.xlabel("Encoder time steps (past quarters)")
plt.ylabel("")
plt.title("Temporal Attention over Historical Quarters (QTFT)")

plt.tight_layout()

pdf_path = DESKTOP / "figure5_attention_heatmap.pdf"
png_path = DESKTOP / "figure5_attention_heatmap.png"

plt.savefig(pdf_path)
plt.savefig(png_path, dpi=300)
plt.close()

print("✅ Figure 5 generated successfully:")
print(" ", pdf_path)
print(" ", png_path)
