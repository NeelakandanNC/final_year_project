"""
STEP 3 - Try the trained models on one segment (a quick demo / check that the weights work)

    python predict.py --folder try3        # weights of try3, random segment from its data
    python predict.py --folder try1        # older tries work too
    python predict.py --folder try3 --part try3/data/part_000.npz --index 7
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch

from models import BPNet, RiskNet, normalize, patient_features, clinical_features, RISK_NAMES


def load_config(folder):
    """try3 and newer save a config.json. try1 / try2 used the CNN with ReLU."""
    path = f"{folder}/weights/config.json"
    if os.path.exists(path):
        return json.load(open(path))
    risk_state = torch.load(f"{folder}/weights/risk_cnn.pt", map_location="cpu")
    n_extra = risk_state["head.0.weight"].shape[1] - 130          # 0 for try1, 3 for try2
    return {"data": folder, "arch": "cnn", "act": "relu", "bp_n_extra": 0,
            "risk_inputs": "basic" if n_extra else "none", "risk_n_extra": n_extra, "clinical_stats": None}


def load_models(folder, cfg):
    names = ("bp_model.pt", "risk_model.pt") if os.path.exists(f"{folder}/weights/bp_model.pt") else ("bp_cnn.pt", "risk_cnn.pt")
    bp_model = BPNet(arch=cfg["arch"], act=cfg["act"], n_extra=cfg["bp_n_extra"])
    bp_model.load_state_dict(torch.load(f"{folder}/weights/{names[0]}", map_location="cpu"))
    risk_model = RiskNet(n_extra=cfg["risk_n_extra"], arch=cfg["arch"], act=cfg["act"])
    risk_model.load_state_dict(torch.load(f"{folder}/weights/{names[1]}", map_location="cpu"))
    return bp_model.eval(), risk_model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="try3", help="try1, try2, try3, ...")
    parser.add_argument("--part", default=None, help="a part_xxx.npz file")
    parser.add_argument("--index", type=int, default=None, help="segment number inside that file")
    args = parser.parse_args()

    cfg = load_config(args.folder)
    part = args.part or np.random.choice(sorted(glob.glob(f"{cfg['data']}/data/part_*.npz")))
    d = np.load(part)
    i = args.index if args.index is not None else np.random.randint(len(d["X"]))
    cid = int(d["caseid"][i])

    x = torch.tensor(normalize(d["X"][i:i + 1]))
    pat = torch.tensor(patient_features(d["age"][i:i + 1], d["sex"][i:i + 1], d["bmi"][i:i + 1])) if "age" in d.files else None
    if cfg["risk_inputs"] == "clinical":
        clinical = pd.read_csv(f"{cfg['data']}/data/clinical.csv")
        extra, _ = clinical_features(clinical[clinical["caseid"] == cid], cfg["clinical_stats"])
        extra = torch.tensor(extra)
    else:
        extra = pat

    bp_model, risk_model = load_models(args.folder, cfg)
    with torch.no_grad():
        bp = bp_model.predict_bp(x, pat)
        probs = torch.softmax(risk_model(x, bp, extra), dim=1)[0].numpy()
    sbp, dbp = bp[0].tolist()

    print(f"Weights: {args.folder} ({cfg['arch']})   File: {part}   segment: {i}   patient (caseid): {cid}")
    if "age" in d.files:
        print(f"Patient       : age {d['age'][i]:.0f}, {'male' if d['sex'][i] == 1 else 'female'}, BMI {d['bmi'][i]:.1f}")
    print(f"Predicted BP  : {sbp:.0f} / {dbp:.0f} mmHg")
    print(f"True BP       : {d['sbp'][i]:.0f} / {d['dbp'][i]:.0f} mmHg")
    print(f"Predicted risk: {RISK_NAMES[probs.argmax()]}")
    print(f"True risk     : {RISK_NAMES[d['risk'][i]]}")
    print("Probabilities :", ", ".join(f"{n} {p * 100:.0f}%" for n, p in zip(RISK_NAMES, probs)))


if __name__ == "__main__":
    main()
