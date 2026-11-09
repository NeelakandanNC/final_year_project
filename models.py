"""
The two deep learning models.

BPNet   : PPG + ECG (+ age, sex, BMI)                 -> SBP, DBP
RiskNet : PPG + ECG + predicted BP (+ patient info)   -> 4 risk classes

The signal part (encoder) can be:
    "cnn"  : 1D CNN                                    (try1, try2)
    "gru"  : RNN with GRU cells   (2 layers, bidirectional)
    "lstm" : RNN with LSTM cells  (2 layers, bidirectional)
The RNN reads the 10 s signal as 100 time steps of 0.1 s
(each step = 10 PPG samples + 10 ECG samples).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

RISK_NAMES = ["No risk", "Low risk", "Moderate risk", "High risk"]

# BP is scaled with these fixed numbers so the network trains on values around 0
BP_MEAN = torch.tensor([120.0, 65.0])   # SBP, DBP
BP_STD = torch.tensor([20.0, 12.0])

N_PATIENT = 3   # age, sex, BMI

# pre-operative information from VitalDB used by the risk model ("clinical" mode)
CLINICAL_NUMBERS = ["age", "sex", "bmi", "emop", "preop_htn", "preop_dm",
                    "preop_hb", "preop_plt", "preop_pt", "preop_aptt", "preop_na", "preop_k",
                    "preop_gluc", "preop_alb", "preop_ast", "preop_alt", "preop_bun", "preop_cr"]
CLINICAL_CATEGORIES = ["department", "optype", "approach", "ane_type"]


def normalize(X):
    """Z-score every signal of every segment. X: (N, 2, 1000) -> float32."""
    X = np.asarray(X, dtype=np.float32)
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True) + 1e-6
    return (X - mean) / std


def patient_features(age, sex, bmi):
    """Scale age, sex (1 = male) and BMI to about 0-1. Missing values get a typical value."""
    age = np.nan_to_num(np.asarray(age, np.float32), nan=58.0) / 100
    sex = np.nan_to_num(np.asarray(sex, np.float32), nan=0.5)
    bmi = np.nan_to_num(np.asarray(bmi, np.float32), nan=23.0) / 30
    return np.stack([age, sex, bmi], axis=1).astype(np.float32)


def clinical_features(df, stats=None):
    """Turn VitalDB clinical columns (one row per segment or patient) into numbers for the model.
    stats=None: learn medians / means / categories from df (use the TRAINING patients).
    Returns (array, stats). Save stats so predict.py can do exactly the same."""
    df = df.copy()
    if df["sex"].dtype == object:
        df["sex"] = (df["sex"] == "M").astype(float)
    # abnormal pre-op ECG / lung function test
    if "preop_ecg" in df:
        df["ecg_abnormal"] = (df["preop_ecg"].notna() & (df["preop_ecg"] != "Normal Sinus Rhythm")).astype(float)
    if "preop_pft" in df:
        df["pft_abnormal"] = (df["preop_pft"].notna() & (df["preop_pft"] != "Normal")).astype(float)
    numbers = [c for c in CLINICAL_NUMBERS + ["ecg_abnormal", "pft_abnormal"] if c in df]

    if stats is None:
        stats = {"numbers": {}, "categories": {}}
        for c in numbers:
            col = pd.to_numeric(df[c], errors="coerce")
            med = float(col.median()) if col.notna().any() else 0.0
            col = col.fillna(med)
            stats["numbers"][c] = [med, float(col.mean()), float(col.std() + 1e-6)]
        for c in CLINICAL_CATEGORIES:
            if c in df:
                top = df[c].value_counts()
                stats["categories"][c] = [str(v) for v in top[top >= 10].index]   # categories seen 10+ times

    cols = []
    for c, (med, mean, std) in stats["numbers"].items():
        col = pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(np.nan, index=df.index)
        cols.append(((col.fillna(med) - mean) / std).values)
    for c, values in stats["categories"].items():
        for v in values:
            cols.append((df[c].astype(str) == v).astype(float).values if c in df else np.zeros(len(df)))
    return np.stack(cols, axis=1).astype(np.float32), stats


def make_act(name):
    return {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[name]()


def conv_block(c_in, c_out, act):
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel_size=7, padding=3),
        nn.BatchNorm1d(c_out),
        make_act(act),
        nn.MaxPool1d(2),
    )


class Encoder(nn.Module):
    """1D CNN: (2, 1000) PPG+ECG segment -> 128 features."""

    def __init__(self, act="relu"):
        super().__init__()
        self.dim = 128
        self.net = nn.Sequential(
            conv_block(2, 32, act),
            conv_block(32, 64, act),
            conv_block(64, 128, act),
            conv_block(128, 128, act),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


class RNNEncoder(nn.Module):
    """RNN: (2, 1000) segment -> 100 steps of 0.1 s -> 2-layer bidirectional LSTM / GRU -> 512 features."""

    def __init__(self, kind="gru", hidden=128):
        super().__init__()
        rnn = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn(input_size=20, hidden_size=hidden, num_layers=2,
                       batch_first=True, bidirectional=True, dropout=0.2)
        self.dim = 4 * hidden

    def forward(self, x):
        n = x.shape[0]
        steps = x.reshape(n, 2, 100, 10).permute(0, 2, 1, 3).reshape(n, 100, 20)
        out, _ = self.rnn(steps)                                  # (n, 100, 2 * hidden)
        return torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=1)


def make_encoder(arch, act):
    return Encoder(act) if arch == "cnn" else RNNEncoder(arch)


class BPNet(nn.Module):
    """n_extra = 0 : PPG + ECG,  n_extra = 3 : PPG + ECG + age/sex/BMI"""

    def __init__(self, arch="cnn", act="relu", n_extra=0):
        super().__init__()
        self.n_extra = n_extra
        self.encoder = make_encoder(arch, act)
        self.head = nn.Sequential(nn.Linear(self.encoder.dim + n_extra, 64), make_act(act), nn.Dropout(0.3),
                                  nn.Linear(64, 2))

    def forward(self, x, extra=None):
        """Returns scaled BP. Use predict_bp() to get mmHg."""
        feats = self.encoder(x)
        if self.n_extra:
            feats = torch.cat([feats, extra], dim=1)
        return self.head(feats)

    def predict_bp(self, x, extra=None):
        out = self.forward(x, extra)
        return out * BP_STD.to(x.device) + BP_MEAN.to(x.device)


class RiskNet(nn.Module):
    """n_extra = 0 : PPG + ECG + BP                     (try1)
       n_extra = 3 : PPG + ECG + BP + age/sex/BMI       (try2)
       n_extra = k : PPG + ECG + BP + k clinical values (try3+)"""

    def __init__(self, n_extra=0, n_classes=4, arch="cnn", act="relu"):
        super().__init__()
        self.n_extra = n_extra
        self.encoder = make_encoder(arch, act)
        self.head = nn.Sequential(nn.Linear(self.encoder.dim + 2 + n_extra, 64), make_act(act), nn.Dropout(0.3),
                                  nn.Linear(64, n_classes))

    def forward(self, x, bp_mmhg, extra=None):
        bp = (bp_mmhg - BP_MEAN.to(x.device)) / BP_STD.to(x.device)
        feats = [self.encoder(x), bp]
        if self.n_extra:
            feats.append(extra)
        return self.head(torch.cat(feats, dim=1))
