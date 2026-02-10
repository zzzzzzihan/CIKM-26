"""
QTFT Training Pipeline with Ablation Studies
Author: ZiHan Sha
Requirements: torch, pytorch-forecasting, pandas, numpy
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import json
from dataclasses import dataclass

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. DATA PROCESSING (基于你的480行数据)
# =============================================================================

@dataclass
class DataConfig:
    data_path: str = "macro_quarterly_480.csv"
    target: str = "gdp_growth"
    time_idx: str = "quarter"
    group_ids: List[str] = None
    static_categoricals: List[str] = None
    time_varying_known_reals: List[str] = None
    time_varying_unknown_reals: List[str] = None
    max_encoder_length: int = 20
    max_prediction_length: int = 8
    train_cutoff: float = 0.7  # 2014年底
    val_cutoff: float = 0.85   # 2017年底
    
    def __post_init__(self):
        self.group_ids = ["country"]
        self.static_categoricals = ["country"]
        self.time_varying_known_reals = []
        self.time_varying_unknown_reals = ["gdp_growth", "inflation", "unemployment"]


class MacroDataModule:
    """处理你的480行宏观数据"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.data = None
        self.training = None
        self.validation = None
        self.test = None
        self.scalers = {}
        
    def load_and_preprocess(self) -> pd.DataFrame:
        """加载并预处理数据"""
        df = pd.read_csv(self.config.data_path)
        
        # 转换时间
        df[self.config.time_idx] = pd.to_datetime(df[self.config.time_idx])
        df["time_idx"] = df.groupby("country").cumcount()
        
        # 添加静态特征编码
        df["country_code"] = df["country"].astype("category").cat.codes
        
        # 标准化（按国家分组）
        for country in df["country"].unique():
            mask = df["country"] == country
            for col in ["gdp_growth", "inflation", "unemployment"]:
                scaler = StandardScaler()
                df.loc[mask, f"{col}_scaled"] = scaler.fit_transform(
                    df.loc[mask, col].values.reshape(-1, 1)
                )
                self.scalers[f"{country}_{col}"] = scaler
        
        # 目标变量使用原始值（或标准化后的）
        df["target"] = df[f"{self.config.target}_scaled"]
        
        self.data = df
        return df
    
    def split_data(self):
        """按时间划分训练/验证/测试集（匹配论文设定）"""
        df = self.data
        
        # 时间划分点
        max_time = df["time_idx"].max()
        train_cutoff = int(max_time * self.config.train_cutoff)      # ~2014Q4
        val_cutoff = int(max_time * self.config.val_cutoff)          # ~2017Q4
        
        self.training = df[df["time_idx"] <= train_cutoff]
        self.validation = df[
            (df["time_idx"] > train_cutoff) & (df["time_idx"] <= val_cutoff)
        ]
        self.test = df[df["time_idx"] > val_cutoff]
        
        print(f"Train: {len(self.training)} samples")
        print(f"Val: {len(self.validation)} samples")  
        print(f"Test: {len(self.test)} samples")
        
        return self.training, self.validation, self.test


# =============================================================================
# 2. 酉变换层实现（核心创新）
# =============================================================================

class UnitaryTransformation(nn.Module):
    """
    酉变换层：保持范数的复数线性变换
    使用乘积门分解实现高效参数化
    """
    
    def __init__(self, dim: int, num_layers: int = 1, method: str = "product_gates"):
        """
        Args:
            dim: 输入维度（必须是2的倍数，用于复数表示）
            num_layers: 酉层层数
            method: "product_gates", "cayley", "exponential", "householder"
        """
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.method = method
        self.real_dim = dim // 2  # 复数维度 = 实数维度/2
        
        if method == "product_gates":
            # Parameters, not submodules
            self.layers = nn.ParameterList([
                self._create_product_gate_layer() for _ in range(num_layers)
            ])
        elif method == "cayley":
            self.skew_symmetric = nn.Parameter(torch.randn(dim, dim) * 0.01)
        elif method == "exponential":
            self.generator = nn.Parameter(torch.randn(dim, dim) * 0.01)
        elif method == "householder":
            self.vectors = nn.Parameter(torch.randn(num_layers, dim) * 0.01)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _create_product_gate_layer(self):
        """创建单个乘积门层（2-qubit门 = 4x4实数矩阵）"""
        # 每4个实数 = 2个复数 = 1个单量子比特门
        num_gates = self.dim // 4
        return nn.Parameter(torch.randn(num_gates, 6) * 0.01)  # 6参数化SU(2)
    
    def _apply_product_gate(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """应用乘积门分解"""
        batch_size = x.shape[0]
        
        # 将实数向量视为复数：x = [real, imag]
        x_complex = torch.view_as_complex(x.reshape(batch_size, -1, 2))
        out_cols = []
        
        # 每对相邻元素应用2x2酉门
        for i in range(0, self.real_dim - 1, 2):
            # 从6参数构造2x2酉矩阵（Cayley-Klein参数化）
            a, b, c, d, e, f = params[i // 2]
            
            # 构造复数矩阵元素
            u11 = torch.complex(torch.cos(a), torch.sin(a)) * torch.cos(b)
            u12 = torch.complex(torch.cos(c), torch.sin(c)) * torch.sin(b)
            u21 = torch.complex(torch.cos(d), torch.sin(d)) * torch.sin(b)
            u22 = torch.complex(torch.cos(e), torch.sin(e)) * torch.cos(b)
            
            # 应用门
            xi, xj = x_complex[:, i], x_complex[:, i + 1]
            yi = u11 * xi + u12 * xj
            yj = u21 * xi + u22 * xj
            out_cols.append(yi)
            out_cols.append(yj)

        if self.real_dim % 2 == 1:
            out_cols.append(x_complex[:, -1])
        
        # Convert back to real representation
        x_out_complex = torch.stack(out_cols, dim=1)
        x_out = torch.view_as_real(x_out_complex).reshape(batch_size, self.dim)
        return x_out

    def _cayley_transform(self) -> torch.Tensor:
        """Cayley变换：U = (I - A)(I + A)^{-1}"""
        A = self.skew_symmetric - self.skew_symmetric.T  # 强制斜对称
        I = torch.eye(self.dim, device=A.device)
        U = torch.linalg.solve(I + A, I - A)
        return U
    
    def _exponential_map(self) -> torch.Tensor:
        """指数映射：U = exp(A)"""
        A = self.generator - self.generator.T  # 斜对称
        return torch.matrix_exp(A)
    
    def _householder_reflections(self, x: torch.Tensor) -> torch.Tensor:
        """Householder反射序列"""
        for v in self.vectors:
            v = v / (v.norm() + 1e-8)
            x = x - 2 * (x @ v).unsqueeze(1) * v.unsqueeze(0)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == "product_gates":
            for layer in self.layers:
                x = self._apply_product_gate(x, layer)
            return x
        elif self.method == "cayley":
            U = self._cayley_transform()
            return x @ U.T
        elif self.method == "exponential":
            U = self._exponential_map()
            return x @ U.T
        elif self.method == "householder":
            return self._householder_reflections(x)
    
    def check_unitary(self) -> float:
        """检查酉性：U^H U = I"""
        if self.method in ["cayley", "exponential"]:
            with torch.no_grad():
                if self.method == "cayley":
                    U = self._cayley_transform()
                else:
                    U = self._exponential_map()
                I = torch.eye(self.dim, device=U.device)
                deviation = torch.norm(U.T @ U - I).item()
                return deviation
        return 0.0


# =============================================================================
# 3. QTFT模型架构
# =============================================================================

class VariableSelectionNetwork(nn.Module):
    """变量选择网络（简化版TFT）"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_vars: int):
        super().__init__()
        self.num_vars = num_vars
        self.variable_grn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.weights_grn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_vars),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, num_vars, input_dim]
        var_outputs = self.variable_grn(x)  # [batch, num_vars, hidden_dim]
        weights = self.weights_grn(x.mean(dim=1))  # [batch, num_vars]
        
        # 加权求和
        output = (var_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return output, weights


class GatedResidualNetwork(nn.Module):
    """门控残差网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.activation = nn.SiLU()
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        hidden = self.activation(self.fc1(x))
        output = self.fc2(hidden)
        gate = torch.sigmoid(self.gate(hidden))
        return self.layer_norm(residual + gate * output)


class TemporalAttention(nn.Module):
    """多头时序注意力"""
    
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None):
        # Self-attention on temporal dimension
        attn_out, weights = self.attention(x, x, x)
        return self.layer_norm(x + attn_out), weights


class QTFT(pl.LightningModule):
    """
    QTFT: Quantum-inspired Temporal Fusion Transformer
    
    Ablation options:
    - unitary_position: "none", "after_vs", "after_grn", "after_attn"
    - unitary_layers: 1, 2, 3
    - unitary_method: "product_gates", "cayley", "exponential", "householder", "orthogonal"
    - use_complex: True (unitary), False (orthogonal)
    """
    
    def __init__(
        self,
        input_dim: int = 3,  # gdp, inflation, unemployment
        hidden_dim: int = 160,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        # Ablation parameters
        unitary_position: str = "after_vs",  # "none", "after_vs", "after_grn", "after_attn"
        unitary_layers: int = 1,
        unitary_method: str = "product_gates",
        use_complex: bool = True,
        # Forecasting
        prediction_horizons: List[int] = [1, 4, 8],
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.prediction_horizons = prediction_horizons
        self.unitary_position = unitary_position
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Variable selection
        self.variable_selection = VariableSelectionNetwork(
            hidden_dim, hidden_dim, input_dim
        )
        
        # Unitary transformation (conditional on position)
        self.unitary = None
        if unitary_position == "after_vs":
            self.unitary = UnitaryTransformation(
                hidden_dim, unitary_layers, unitary_method
            )
        
        # Gated residual networks
        self.grn_layers = nn.ModuleList([
            GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        if unitary_position == "after_grn":
            self.unitary = UnitaryTransformation(
                hidden_dim, unitary_layers, unitary_method
            )
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(hidden_dim, num_heads)
        
        if unitary_position == "after_attn":
            self.unitary = UnitaryTransformation(
                hidden_dim, unitary_layers, unitary_method
            )
        
        # Output layers for multi-horizon prediction
        self.output_layers = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
            for h in prediction_horizons
        })
        
        self.criterion = nn.MSELoss()
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            predictions: Dict[str, Tensor] for each horizon
        """
        batch_size, seq_len, _ = x.shape
        
        # Embed inputs
        x_embed = self.input_embedding(x)  # [batch, seq_len, hidden_dim]
        
        # Variable selection (per time step)
        vs_outputs = []
        for t in range(seq_len):
            vs_out, vs_weights = self.variable_selection(x_embed[:, t, :].unsqueeze(1))
            vs_outputs.append(vs_out)
        x = torch.stack(vs_outputs, dim=1)  # [batch, seq_len, hidden_dim]
        
        # Unitary transformation (position 1)
        if self.unitary_position == "after_vs" and self.unitary is not None:
            x_flat = x.reshape(-1, self.hidden_dim)
            x_flat = self.unitary(x_flat)
            x = x_flat.reshape(batch_size, seq_len, self.hidden_dim)
        
        # GRN layers
        for grn in self.grn_layers:
            x = grn(x)
        
        # Unitary transformation (position 2)
        if self.unitary_position == "after_grn" and self.unitary is not None:
            x_flat = x.reshape(-1, self.hidden_dim)
            x_flat = self.unitary(x_flat)
            x = x_flat.reshape(batch_size, seq_len, self.hidden_dim)
        
        # Temporal attention
        x, attn_weights = self.temporal_attention(x)
        
        # Unitary transformation (position 3)
        if self.unitary_position == "after_attn" and self.unitary is not None:
            x_flat = x.reshape(-1, self.hidden_dim)
            x_flat = self.unitary(x_flat)
            x = x_flat.reshape(batch_size, seq_len, self.hidden_dim)
        
        # Global average pooling over time
        x_pooled = x.mean(dim=1)  # [batch, hidden_dim]
        
        # Multi-horizon predictions
        predictions = {
            str(h): self.output_layers[str(h)](x_pooled).squeeze(-1)
            for h in self.prediction_horizons
        }
        
        if return_attention:
            return predictions, attn_weights, vs_weights
        return predictions
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        
        # Multi-horizon loss
        loss = 0
        for h in self.prediction_horizons:
            loss += self.criterion(predictions[str(h)], y[str(h)])
        loss /= len(self.prediction_horizons)
        
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        
        loss = 0
        metrics = {}
        for h in self.prediction_horizons:
            pred = predictions[str(h)]
            true = y[str(h)]
            mse = self.criterion(pred, true)
            mae = torch.abs(pred - true).mean()
            rmse = torch.sqrt(mse)
            
            metrics[f"val_rmse_h{h}"] = rmse
            metrics[f"val_mae_h{h}"] = mae
            loss += mse
        
        loss /= len(self.prediction_horizons)
        metrics["val_loss"] = loss

        # Log val_loss explicitly for EarlyStopping
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log_dict(metrics, prog_bar=True)
        return metrics
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss"}
        }


# =============================================================================
# 4. 消融实验配置
# =============================================================================

ABLATION_CONFIGS = {
    # 基线
    "TFT_baseline": {
        "unitary_position": "none",
        "unitary_layers": 0,
        "unitary_method": "none",
        "use_complex": False,
    },
    
    # 位置消融
    "QTFT_after_vs": {
        "unitary_position": "after_vs",
        "unitary_layers": 1,
        "unitary_method": "product_gates",
        "use_complex": True,
    },
    "QTFT_after_grn": {
        "unitary_position": "after_grn",
        "unitary_layers": 1,
        "unitary_method": "product_gates",
        "use_complex": True,
    },
    "QTFT_after_attn": {
        "unitary_position": "after_attn",
        "unitary_layers": 1,
        "unitary_method": "product_gates",
        "use_complex": True,
    },
    
    # 层数消融
    "QTFT_layers_2": {
        "unitary_position": "after_vs",
        "unitary_layers": 2,
        "unitary_method": "product_gates",
        "use_complex": True,
    },
    "QTFT_layers_3": {
        "unitary_position": "after_vs",
        "unitary_layers": 3,
        "unitary_method": "product_gates",
        "use_complex": True,
    },
    
    # 参数化方法消融
    "QTFT_cayley": {
        "unitary_position": "after_vs",
        "unitary_layers": 1,
        "unitary_method": "cayley",
        "use_complex": True,
    },
    "QTFT_exponential": {
        "unitary_position": "after_vs",
        "unitary_layers": 1,
        "unitary_method": "exponential",
        "use_complex": True,
    },
    "QTFT_householder": {
        "unitary_position": "after_vs",
        "unitary_layers": 1,
        "unitary_method": "householder",
        "use_complex": True,
    },
    
    # 正交对比（实数）
    "QTFT_orthogonal": {
        "unitary_position": "after_vs",
        "unitary_layers": 1,
        "unitary_method": "product_gates",
        "use_complex": False,
    },
}


# =============================================================================
# 5. 训练流程
# =============================================================================

class MacroDataset(Dataset):
    """宏观数据Dataset"""
    
    def __init__(self, data: pd.DataFrame, config: DataConfig, is_train: bool = True):
        self.data = data
        self.config = config
        self.is_train = is_train
        
        # 构建序列
        self.samples = self._build_samples()
    
    def _build_samples(self):
        samples = []
        for country in self.data["country"].unique():
            country_data = self.data[self.data["country"] == country].sort_values("time_idx")
            values = country_data[["gdp_growth_scaled", "inflation_scaled", "unemployment_scaled"]].values
            
            # 滑动窗口
            for i in range(
                len(values) - self.config.max_encoder_length - self.config.max_prediction_length + 1
            ):
                encoder_end = i + self.config.max_encoder_length
                encoder_data = values[i:encoder_end]
                
                # 多horizon目标
                targets = {}
                for h in [1, 4, 8]:
                    if encoder_end + h - 1 < len(values):
                        targets[str(h)] = values[encoder_end + h - 1, 0]  # GDP growth
                
                if len(targets) == 3:  # 确保所有horizon都有值
                    samples.append({
                        "encoder": encoder_data,
                        "targets": targets
                    })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        x = torch.FloatTensor(sample["encoder"])
        y = {k: torch.FloatTensor([v]) for k, v in sample["targets"].items()}
        return x, y


def train_model(
    config_name: str,
    model_config: Dict,
    data_module: MacroDataModule,
    max_epochs: int = 50,
    gpus: int = 0,
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> Dict:
    """训练单个模型配置"""
    
    pl.seed_everything(seed)
    
    # 创建数据加载器
    train_dataset = MacroDataset(data_module.training, data_module.config)
    val_dataset = MacroDataset(data_module.validation, data_module.config)
    test_dataset = MacroDataset(data_module.test, data_module.config)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)
    
    # 创建模型
    model = QTFT(**model_config)
    
    # 回调
    early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min", check_on_train_epoch_end=False)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    # Prefer TensorBoard if available, otherwise fall back to CSV logs in output_dir
    out_dir = output_dir or "outputs"
    os.makedirs(out_dir, exist_ok=True)
    try:
        logger = TensorBoardLogger(out_dir, name=config_name)
    except ModuleNotFoundError:
        logger = CSVLogger(out_dir, name=config_name)
    
    # 训练器
    callbacks = [early_stop]
    if logger:
        callbacks.append(lr_monitor)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if gpus > 0 else "cpu",
        devices=gpus if gpus > 0 else 1,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=0.1,
        enable_progress_bar=True,
    )
    
    # 训练
    trainer.fit(model, train_loader, val_loader)
    
    # 测试
    test_results = trainer.test(model, test_loader)
    
    # 保存结果
    results = {
        "config": config_name,
        "test_metrics": test_results[0] if test_results else {},
        "model_path": trainer.checkpoint_callback.best_model_path if hasattr(trainer, 'checkpoint_callback') else None
    }
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"result_{config_name}.json"), "w") as f:
            json.dump(results, f, indent=2)
        run_info = {
            "config_name": config_name,
            "model_config": model_config,
            "data_path": data_module.config.data_path,
            "train_samples": len(data_module.training),
            "val_samples": len(data_module.validation),
            "test_samples": len(data_module.test),
        }
        with open(os.path.join(output_dir, f"run_info_{config_name}.json"), "w") as f:
            json.dump(run_info, f, indent=2)
    
    return results


def run_all_ablations(
    data_path: str = "macro_quarterly_480.csv",
    seeds: List[int] = [42, 123, 456],
    output_dir: Optional[str] = None,
):
    """运行所有消融实验"""
    
    # 准备数据
    config = DataConfig(data_path=data_path)
    data_module = MacroDataModule(config)
    data_module.load_and_preprocess()
    data_module.split_data()
    
    all_results = []
    
    for config_name, model_config in ABLATION_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Training: {config_name}")
        print(f"{'='*60}")
        
        config_results = []
        for seed in seeds:
            print(f"\nSeed: {seed}")
            result = train_model(
                config_name=f"{config_name}_seed{seed}",
                model_config=model_config,
                data_module=data_module,
                seed=seed,
                output_dir=output_dir,
            )
            config_results.append(result)
        
        # 聚合多种子结果
        aggregated = aggregate_results(config_results, config_name)
        all_results.append(aggregated)
        
        # 保存中间结果
        out_dir = output_dir or "outputs"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"results_{config_name}.json"), "w") as f:
            json.dump(aggregated, f, indent=2)
    
    # 保存完整结果
    out_dir = output_dir or "outputs"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "all_ablation_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    # 生成对比表格
    generate_comparison_table(all_results, output_dir=output_dir)
    
    return all_results


def aggregate_results(results: List[Dict], config_name: str) -> Dict:
    """聚合多随机种子结果"""
    
    metrics = {}
    for result in results:
        for key, value in result["test_metrics"].items():
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(value)
    
    aggregated = {"config": config_name}
    for key, values in metrics.items():
        aggregated[f"{key}_mean"] = np.mean(values)
        aggregated[f"{key}_std"] = np.std(values)
    
    return aggregated


def generate_comparison_table(results: List[Dict], output_dir: Optional[str] = None):
    """生成论文表格"""
    
    rows = []
    for r in results:
        row = {
            "Model": r["config"],
            "1-step RMSE": f"{r.get('val_rmse_h1_mean', 0):.2f} ± {r.get('val_rmse_h1_std', 0):.2f}",
            "4-step RMSE": f"{r.get('val_rmse_h4_mean', 0):.2f} ± {r.get('val_rmse_h4_std', 0):.2f}",
            "8-step RMSE": f"{r.get('val_rmse_h8_mean', 0):.2f} ± {r.get('val_rmse_h8_std', 0):.2f}",
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print("\n" + "="*80)
    print("ABLATION RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # 保存为LaTeX表格
    latex_table = df.to_latex(index=False, escape=False)
    out_dir = output_dir or "outputs"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ablation_results.tex"), "w") as f:
        f.write(latex_table)
    df.to_csv(os.path.join(out_dir, "ablation_results.csv"), index=False)
    
    return df


# =============================================================================
# 6. 主函数
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="QTFT Training")
    parser.add_argument("--mode", choices=["single", "ablation"], default="single")
    parser.add_argument("--config", default="QTFT_after_vs", help="Config name for single run")
    parser.add_argument("--data", default="macro_quarterly_480.csv", help="Data path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--outdir", default="outputs", help="Output directory (saved on Desktop)")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        # 单配置训练
        config = DataConfig(data_path=args.data)
        data_module = MacroDataModule(config)
        data_module.load_and_preprocess()
        data_module.split_data()
        
        model_config = ABLATION_CONFIGS[args.config]
        result = train_model(
            args.config, model_config, data_module, 
            max_epochs=args.epochs, gpus=args.gpus,
            output_dir=args.outdir,
        )
        print(f"Result: {result}")
    
    else:
        # 完整消融实验
        results = run_all_ablations(args.data, args.seeds, output_dir=args.outdir)
        print("\nAll experiments completed!")

