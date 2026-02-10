import os
from pathlib import Path
import torch
import torch.nn as nn
import pennylane as qml
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss


def find_file(filename):

    if os.path.exists(filename):
        return filename

    desktop = Path.home() / "Desktop"
    if (desktop / filename).exists():
        return str(desktop / filename)

    return input(f"Can't find{filename} , please enter the whole path")

file_path = find_file("macro_quarterly_480.csv")

df = pd.read_csv(file_path)
df["quarter"] = pd.to_datetime(df["quarter"])
df = df.sort_values(["country", "quarter"])
df["time_idx"] = df.groupby("country").cumcount()
df["country"] = df["country"].astype(str)

class QuantumFeatureLayer(nn.Module):
    def __init__(self, n_qubits=3, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            for i in range(n_qubits):
                qml.RY(inputs[i] * 3.14159, wires=i)

            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(weights[layer, i], wires=i)
                for i in range(n_qubits-1):
                    qml.CNOT(wires=[i, i+1])

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = circuit
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)

    def forward(self, x):
        batch_size = x.shape[0]
        results = []
        for i in range(batch_size):
            q_out = self.qnode(x[i, :3], self.weights)
            results.append(torch.tensor(q_out))
        return torch.stack(results).to(x.device)

class QuantumTFT(nn.Module):
    def __init__(self, training, tft_config):
        super().__init__()
        self.quantum_layer = QuantumFeatureLayer()
        self.tft = TemporalFusionTransformer.from_dataset(training, **tft_config)

    def forward(self, x):
        return self.tft(x)

max_encoder_length = 20
max_prediction_length = 8
training_cutoff = df["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    df[df.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="gdp_growth",
    group_ids=["country"],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["gdp_growth", "inflation", "unemployment"],
    target_normalizer=GroupNormalizer(groups=["country"]),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

tft_config = {
    "learning_rate": 1e-3,
    "hidden_size": 16,
    "attention_head_size": 1,
    "dropout": 0.1,
    "hidden_continuous_size": 8,
    "output_size": 1,
    "loss": QuantileLoss(),
}

model = QuantumTFT(training, tft_config)

print("Data and model preparation completed.")
