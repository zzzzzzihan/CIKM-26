# QTFT_Step4_6_Merge_Metrics.py
# Merge baseline + model metrics into final step4_metrics.csv

import pandas as pd
from pathlib import Path

DESKTOP = Path.home() / "Desktop"

# ---------------------------------------------------------
# Load metrics
# ---------------------------------------------------------
baseline_path = DESKTOP / "step4_metrics_baselines.csv"
model_path = DESKTOP / "step4_metrics_models.csv"

if not baseline_path.exists():
    raise FileNotFoundError(f"Missing file: {baseline_path}")

if not model_path.exists():
    raise FileNotFoundError(f"Missing file: {model_path}")

df_baselines = pd.read_csv(baseline_path)
df_models = pd.read_csv(model_path)

print("Loaded baseline metrics:", df_baselines.shape)
print("Loaded model metrics:", df_models.shape)

# ---------------------------------------------------------
# Concatenate
# ---------------------------------------------------------
df_all = pd.concat([df_baselines, df_models], ignore_index=True)

# Optional: sort for nice table order
df_all = df_all.sort_values(
    by=["horizon", "model"]
).reset_index(drop=True)

# ---------------------------------------------------------
# Save final metrics
# ---------------------------------------------------------
out_path = DESKTOP / "step4_metrics.csv"
df_all.to_csv(out_path, index=False)

print("\n✅ Final metrics saved to:")
print(out_path)
print("\nPreview:")
print(df_all.head(10))
