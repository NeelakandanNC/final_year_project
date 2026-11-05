"""
The two deep learning models (plain 1D CNNs).

BPNet   : PPG + ECG               -> SBP, DBP
RiskNet : PPG + ECG + predicted BP -> 4 risk classes
"""

import numpy as np
import torch
import torch.nn as nn

RISK_NAMES = ["No risk", "Low risk", "Moderate risk", "High risk"]

# BP is scaled with these fixed numbers so the network trains on values around 0
BP_MEAN = torch.tensor([120.0, 65.0])   # SBP, DBP
BP_STD = torch.tensor([20.0, 12.0])


def normalize(X):
    """Z-score every signal of every segment. X: (N, 2, 1000) -> float32."""
    X = np.asarray(X, dtype=np.float32)
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True) + 1e-6
    return (X - mean) / std


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel_size=7, padding=3),
        nn.BatchNorm1d(c_out),
        nn.ReLU(),
        nn.MaxPool1d(2),
    )


class Encoder(nn.Module):
    """Turns a (2, 1000) PPG+ECG segment into a 128 feature vector."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(2, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 128),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


class BPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, x):
        """Returns scaled BP. Use predict_bp() to get mmHg."""
        return self.head(self.encoder(x))

    def predict_bp(self, x):
        out = self.forward(x)
        return out * BP_STD.to(x.device) + BP_MEAN.to(x.device)


class RiskNet(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.encoder = Encoder()
        self.head = nn.Sequential(nn.Linear(128 + 2, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_classes))

    def forward(self, x, bp_mmhg):
        bp = (bp_mmhg - BP_MEAN.to(x.device)) / BP_STD.to(x.device)
        return self.head(torch.cat([self.encoder(x), bp], dim=1))
