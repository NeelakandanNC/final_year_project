"""
STEP 1 - Collect data from VitalDB

Downloads PPG, ECG and arterial blood pressure (ART) from the open VitalDB
dataset, cuts them into 10-second segments and saves them in the data/ folder.

For every segment we save:
    X     -> PPG and ECG signals, shape (2, 1000)  (10 s at 100 Hz)
    sbp   -> systolic BP  (mmHg)  from the arterial line   (BP label)
    dbp   -> diastolic BP (mmHg)  from the arterial line   (BP label)
    risk  -> 0 No risk, 1 Low, 2 Moderate, 3 High  from ASA status (risk label)

Run once on your laptop (needs internet):
    pip install -r requirements.txt
    python collect_data.py

If it stops in the middle just run it again, finished parts are skipped.
"""

import argparse
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import vitaldb

warnings.filterwarnings("ignore")

FS = 100                  # sampling rate we save at (VitalDB is 500 Hz, we downsample)
SEG_LEN = 10 * FS         # 10 second segments
TRACKS = ["SNUADC/PLETH", "SNUADC/ECG_II", "SNUADC/ART"]   # PPG, ECG, arterial BP
RISK_NAMES = ["No risk", "Low risk", "Moderate risk", "High risk"]
CASES_PER_PART = 100      # cases saved in each data/part_xxx.npz file


def asa_to_risk(asa):
    """ASA physical status -> 4 risk classes."""
    if pd.isna(asa):
        return -1
    asa = int(asa)
    if asa == 1:
        return 0      # No risk   (healthy patient)
    if asa == 2:
        return 1      # Low risk  (mild systemic disease)
    if asa == 3:
        return 2      # Moderate  (severe systemic disease)
    return 3          # High risk (ASA 4, 5, 6)


def get_bp(art):
    """SBP and DBP of a 10 s arterial waveform (median of beat peaks / troughs)."""
    peaks, _ = find_peaks(art, distance=int(0.3 * FS), prominence=10)
    troughs, _ = find_peaks(-art, distance=int(0.3 * FS), prominence=10)
    if len(peaks) < 5 or len(troughs) < 5:
        return None
    sbp = np.median(art[peaks])
    dbp = np.median(art[troughs])
    if np.std(art[peaks]) > 15:          # BP jumping around a lot -> skip
        return None
    return sbp, dbp, len(peaks)


def good_segment(ppg, ecg, art):
    """Simple quality check. Returns (sbp, dbp) or None if the segment is bad."""
    if np.isnan(ppg).any() or np.isnan(ecg).any() or np.isnan(art).any():
        return None
    # flat lines / disconnected sensors
    for sig in (ppg, ecg, art):
        if np.mean(np.diff(sig) == 0) > 0.3:
            return None
    if np.ptp(ecg) < 0.2:                # ECG in mV, almost no signal
        return None
    if art.min() < 20 or art.max() > 250:
        return None

    bp = get_bp(art)
    if bp is None:
        return None
    sbp, dbp, n_beats = bp
    if not (70 <= sbp <= 200 and 30 <= dbp <= 120 and sbp - dbp >= 15):
        return None

    # PPG should show roughly the same number of beats as the arterial line
    ppg_peaks, _ = find_peaks(ppg, distance=int(0.3 * FS), prominence=0.2 * np.ptp(ppg))
    if abs(len(ppg_peaks) - n_beats) > 0.3 * n_beats:
        return None
    return sbp, dbp


def process_case(caseid, n_segs):
    """Download one case and return up to n_segs good segments."""
    try:
        data = vitaldb.load_case(caseid, TRACKS, 1 / FS)
    except Exception as e:
        print(f"  case {caseid}: download failed ({e})")
        return None
    if data is None or len(data) < SEG_LEN * 10:
        return None

    ppg, ecg, art = data[:, 0], data[:, 1], data[:, 2]
    # try segments spread evenly over the surgery
    starts = np.linspace(0, len(data) - SEG_LEN, n_segs * 5).astype(int)
    X, sbp, dbp = [], [], []
    for s in starts:
        e = s + SEG_LEN
        res = good_segment(ppg[s:e], ecg[s:e], art[s:e])
        if res is None:
            continue
        X.append(np.stack([ppg[s:e], ecg[s:e]]))
        sbp.append(res[0])
        dbp.append(res[1])
        if len(X) == n_segs:
            break
    if not X:
        return None
    return np.array(X, dtype=np.float16), np.array(sbp, np.float32), np.array(dbp, np.float32)


def choose_cases(n_cases, seed=42):
    """Pick cases that have PPG + ECG + ART, trying to keep the 4 risk classes balanced."""
    print("Reading VitalDB case list ...")
    caseids = vitaldb.find_cases(TRACKS)
    clinical = pd.read_csv(f"{vitaldb.dataset.api_url}/cases")
    clinical = clinical[clinical["caseid"].isin(caseids)].copy()
    clinical["risk"] = clinical["asa"].apply(asa_to_risk)
    clinical = clinical[clinical["risk"] >= 0]
    print(f"{len(clinical)} cases have all 3 signals and an ASA score")
    print(clinical["risk"].value_counts().sort_index().rename(index=dict(enumerate(RISK_NAMES))))

    # take an equal share from each class; small classes give all they have,
    # and what is left over goes to the bigger classes
    groups = {r: g.sample(frac=1, random_state=seed) for r, g in clinical.groupby("risk")}
    order = sorted(groups, key=lambda r: len(groups[r]))     # smallest class first
    picked, left = [], n_cases
    for i, r in enumerate(order):
        take = min(len(groups[r]), left // (len(order) - i))
        picked.append(groups[r].iloc[:take])
        left -= take
    return pd.concat(picked).sort_values("caseid").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=1000, help="number of patients to download")
    parser.add_argument("--segs", type=int, default=20, help="10-second segments per patient")
    parser.add_argument("--workers", type=int, default=8, help="parallel downloads")
    parser.add_argument("--out", default="data")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cases = choose_cases(args.cases)
    # rare classes get more segments per patient so the classes are less unbalanced
    counts = cases["risk"].value_counts()
    boost = {r: int(min(4, max(1, round(counts.max() / counts[r])))) for r in counts.index}
    print("Chosen cases per class:", {RISK_NAMES[r]: int(n) for r, n in counts.sort_index().items()})
    print("Segments per case:", {RISK_NAMES[r]: args.segs * b for r, b in sorted(boost.items())})

    summary = []
    n_parts = int(np.ceil(len(cases) / CASES_PER_PART))
    for p in range(n_parts):
        path = os.path.join(args.out, f"part_{p:03d}.npz")
        chunk = cases.iloc[p * CASES_PER_PART:(p + 1) * CASES_PER_PART]
        if os.path.exists(path):
            print(f"[{p + 1}/{n_parts}] {path} already exists, skipping")
            old = np.load(path)
            for cid in np.unique(old["caseid"]):
                m = old["caseid"] == cid
                summary.append((cid, old["subjectid"][m][0], old["asa"][m][0], old["risk"][m][0], m.sum()))
            continue

        print(f"[{p + 1}/{n_parts}] downloading {len(chunk)} cases ...")
        jobs = [(row.caseid, args.segs * boost[row.risk]) for row in chunk.itertuples()]
        with ThreadPoolExecutor(args.workers) as ex:
            results = list(ex.map(lambda j: process_case(*j), jobs))

        X, sbp, dbp, risk, asa, cid, sid = [], [], [], [], [], [], []
        for row, res in zip(chunk.itertuples(), results):
            if res is None:
                continue
            x, s, d = res
            n = len(x)
            X.append(x); sbp.append(s); dbp.append(d)
            risk += [row.risk] * n; asa += [row.asa] * n
            cid += [row.caseid] * n; sid += [row.subjectid] * n
            summary.append((row.caseid, row.subjectid, row.asa, row.risk, n))
        if not X:
            continue
        np.savez_compressed(path, X=np.concatenate(X), sbp=np.concatenate(sbp), dbp=np.concatenate(dbp),
                            risk=np.array(risk, np.int64), asa=np.array(asa, np.float32),
                            caseid=np.array(cid, np.int64), subjectid=np.array(sid, np.int64))
        size = os.path.getsize(path) / 1e6
        print(f"   saved {path}: {sum(len(x) for x in X)} segments, {size:.1f} MB")

    df = pd.DataFrame(summary, columns=["caseid", "subjectid", "asa", "risk", "n_segments"])
    df.to_csv(os.path.join(args.out, "cases.csv"), index=False)
    print("\nDone.")
    print(f"Patients: {len(df)}   Segments: {df['n_segments'].sum()}")
    print("Segments per risk class:")
    print(df.groupby("risk")["n_segments"].sum().rename(index=dict(enumerate(RISK_NAMES))))


if __name__ == "__main__":
    main()
