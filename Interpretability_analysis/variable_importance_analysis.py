# QTFT_Step5_Explain_Variables.py
# Step 5.1: Encoder Variable Importance (FINAL, STABLE)

import torch
import pandas as pd
import matplotlib.pyplot as plt
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
# Dataset (same as training)
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
# Step 5.1: Encoder variable importance
# =========================================================
importance_batches = []

with torch.no_grad():
    for x, _ in loader:
        x = {k: v.to(DEVICE) for k, v in x.items()}
        raw_out = model(x)

        interp = model.tft.interpret_output(raw_out, reduction="mean")

        # 在你的版本中，这是一个 Tensor
        importance_batches.append(interp["encoder_variables"])

# 聚合所有 batch
importance_all = torch.stack(importance_batches).mean(0).cpu()

# =========================================================
# 只保留“真实经济变量”的 importance
# TFT encoder 中后两个是 relative_time_idx 和 encoder_length
# =========================================================
variable_names = [
    "GDP growth (lagged)",
    "Inflation",
    "Unemployment",
]

importance = importance_all[: len(variable_names)]

# =========================================================
# Plot (Figure 4)
# =========================================================
plt.figure(figsize=(6.5, 4))

plt.bar(
    variable_names,
    importance.numpy(),
    color=["#9DB4AB", "#D1BEB0", "#B6C6E3"],
    edgecolor="black",
    linewidth=0.6,
)

plt.ylabel("Importance score")
plt.title("Encoder Variable Importance (QTFT)")
plt.tight_layout()

pdf_path = DESKTOP / "figure4_variable_importance.pdf"
png_path = DESKTOP / "figure4_variable_importance.png"

plt.savefig(pdf_path)
plt.savefig(png_path, dpi=300)
plt.close()

print("✅ Figure 4 generated successfully:")
print(" ", pdf_path)
print(" ", png_path)
