"""
STEP 3 - Use the trained models

Put bp_cnn.pt and risk_cnn.pt (from Kaggle) in the weights/ folder, then:
    python predict.py                                  # a random segment from data/
    python predict.py --part data/part_000.npz --index 7
"""

import argparse
import glob

import numpy as np
import torch

from models import BPNet, RiskNet, normalize, RISK_NAMES


def load_models(folder="weights"):
    bp_model, risk_model = BPNet(), RiskNet()
    bp_model.load_state_dict(torch.load(f"{folder}/bp_cnn.pt", map_location="cpu"))
    risk_model.load_state_dict(torch.load(f"{folder}/risk_cnn.pt", map_location="cpu"))
    return bp_model.eval(), risk_model.eval()


def predict(bp_model, risk_model, ppg, ecg):
    """ppg and ecg: 10 seconds at 100 Hz (1000 samples each)."""
    x = torch.tensor(normalize(np.stack([ppg, ecg])[None]))
    with torch.no_grad():
        bp = bp_model.predict_bp(x)
        probs = torch.softmax(risk_model(x, bp), dim=1)[0].numpy()
    sbp, dbp = bp[0].tolist()
    return sbp, dbp, RISK_NAMES[probs.argmax()], probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", default=None, help="a data/part_xxx.npz file")
    parser.add_argument("--index", type=int, default=None, help="segment number inside that file")
    args = parser.parse_args()

    part = args.part or np.random.choice(sorted(glob.glob("data/part_*.npz")))
    d = np.load(part)
    i = args.index if args.index is not None else np.random.randint(len(d["X"]))

    bp_model, risk_model = load_models()
    sbp, dbp, label, probs = predict(bp_model, risk_model, d["X"][i, 0], d["X"][i, 1])

    print(f"File: {part}   segment: {i}   patient (caseid): {d['caseid'][i]}")
    print(f"Predicted BP : {sbp:.0f} / {dbp:.0f} mmHg")
    print(f"True BP      : {d['sbp'][i]:.0f} / {d['dbp'][i]:.0f} mmHg")
    print(f"Predicted risk: {label}")
    print(f"True risk     : {RISK_NAMES[d['risk'][i]]}")
    print("Probabilities :", ", ".join(f"{n} {p * 100:.0f}%" for n, p in zip(RISK_NAMES, probs)))


if __name__ == "__main__":
    main()
