# QTFT_Step4_TFT_QTFT.py
# Country-level TFT vs QTFT evaluation (Figure 6)

import math
import torch
import pandas as pd
from pathlib import Path
from collections import defaultdict

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
# Metric
# =========================================================
def evaluate_rmse_mae(y_true, y_pred):
    mse = torch.mean((y_pred - y_true) ** 2).item()
    mae = torch.mean(torch.abs(y_pred - y_true)).item()
    rmse = math.sqrt(mse)
    return rmse, mae

# =========================================================
# Load data
# =========================================================
df = pd.read_csv(DESKTOP / "macro_quarterly_480.csv")
df["quarter"] = pd.to_datetime(df["quarter"])
df = df.sort_values(["country", "quarter"])
df["country"] = df["country"].astype(str)
df["time_idx"] = df.groupby("country").cumcount()

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

loader = dataset.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

# =========================================================
# Collect predictions by country (SAFE VERSION)
# =========================================================
def collect_by_country(model):
    store_pred = defaultdict(list)
    store_true = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = {k: v.to(DEVICE) for k, v in x.items()}
            country_ids = x["groups"][:, 0].cpu().numpy()

            preds = model(x).prediction.squeeze(-1).cpu()
            trues = y[0]

            for i, cid in enumerate(country_ids):
                country = dataset._group_ids[0][cid]
                store_pred[country].append(preds[i])
                store_true[country].append(trues[i])

    return store_true, store_pred

# =========================================================
# Load QTFT
# =========================================================
qtft = QuantumTFT(
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

qtft.load_state_dict(torch.load(DESKTOP / "qtft_best_model.pth", map_location=DEVICE))

# =========================================================
# Load TFT (untrained baseline)
# =========================================================
tft = QuantumTFT(
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

# =========================================================
# Run
# =========================================================
qtft_true, qtft_pred = collect_by_country(qtft)
tft_true, tft_pred = collect_by_country(tft)

rows = []

for country in qtft_true:
    yq_true = torch.stack(qtft_true[country])[:, 7]
    yq_pred = torch.stack(qtft_pred[country])[:, 7]

    yt_true = torch.stack(tft_true[country])[:, 7]
    yt_pred = torch.stack(tft_pred[country])[:, 7]

    rmse_q, mae_q = evaluate_rmse_mae(yq_true, yq_pred)
    rmse_t, mae_t = evaluate_rmse_mae(yt_true, yt_pred)

    rows.append({
        "country": country,
        "model": "QTFT",
        "horizon": 8,
        "RMSE": rmse_q,
        "MAE": mae_q,
    })
    rows.append({
        "country": country,
        "model": "TFT",
        "horizon": 8,
        "RMSE": rmse_t,
        "MAE": mae_t,
    })

# =========================================================
# Save
# =========================================================
df_out = pd.DataFrame(rows)
out_path = DESKTOP / "step4_country_metrics_models.csv"
df_out.to_csv(out_path, index=False)

print(f"\n✅ Country-level metrics saved to: {out_path}")
print(df_out)
