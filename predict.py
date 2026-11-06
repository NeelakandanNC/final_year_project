"""
STEP 3 - Try the trained models on one segment (a quick demo / check that the weights work)

    python predict.py                      # try2 weights, random segment from try2/data
    python predict.py --folder try1        # try1 weights and data
    python predict.py --folder try2 --part try2/data/part_000.npz --index 7
"""

import argparse
import glob

import numpy as np
import torch

from models import BPNet, RiskNet, normalize, patient_features, RISK_NAMES


def load_models(folder):
    bp_model = BPNet()
    bp_model.load_state_dict(torch.load(f"{folder}/weights/bp_cnn.pt", map_location="cpu"))
    risk_state = torch.load(f"{folder}/weights/risk_cnn.pt", map_location="cpu")
    n_extra = risk_state["head.0.weight"].shape[1] - 130      # 0 for try1, 3 for try2
    risk_model = RiskNet(n_extra=n_extra)
    risk_model.load_state_dict(risk_state)
    return bp_model.eval(), risk_model.eval()


def predict(bp_model, risk_model, ppg, ecg, age=None, sex=None, bmi=None):
    """ppg and ecg: 10 seconds at 100 Hz (1000 samples each). sex: 1 = male, 0 = female."""
    x = torch.tensor(normalize(np.stack([ppg, ecg])[None]))
    extra = torch.tensor(patient_features([age if age is not None else np.nan],
                                          [sex if sex is not None else np.nan],
                                          [bmi if bmi is not None else np.nan]))
    with torch.no_grad():
        bp = bp_model.predict_bp(x)
        probs = torch.softmax(risk_model(x, bp, extra), dim=1)[0].numpy()
    sbp, dbp = bp[0].tolist()
    return sbp, dbp, RISK_NAMES[probs.argmax()], probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="try2", help="try1, try2, ...")
    parser.add_argument("--part", default=None, help="a part_xxx.npz file")
    parser.add_argument("--index", type=int, default=None, help="segment number inside that file")
    args = parser.parse_args()

    part = args.part or np.random.choice(sorted(glob.glob(f"{args.folder}/data/part_*.npz")))
    d = np.load(part)
    i = args.index if args.index is not None else np.random.randint(len(d["X"]))
    info = {k: float(d[k][i]) for k in ["age", "sex", "bmi"] if k in d.files}

    bp_model, risk_model = load_models(args.folder)
    sbp, dbp, label, probs = predict(bp_model, risk_model, d["X"][i, 0], d["X"][i, 1], **info)

    print(f"Weights: {args.folder}   File: {part}   segment: {i}   patient (caseid): {d['caseid'][i]}")
    if info:
        print("Patient       : age {age:.0f}, {s}, BMI {bmi:.1f}".format(s="male" if info["sex"] == 1 else "female", **info))
    print(f"Predicted BP  : {sbp:.0f} / {dbp:.0f} mmHg")
    print(f"True BP       : {d['sbp'][i]:.0f} / {d['dbp'][i]:.0f} mmHg")
    print(f"Predicted risk: {label}")
    print(f"True risk     : {RISK_NAMES[d['risk'][i]]}")
    print("Probabilities :", ", ".join(f"{n} {p * 100:.0f}%" for n, p in zip(RISK_NAMES, probs)))


if __name__ == "__main__":
    main()
