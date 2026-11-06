"""
STEP 1 - Collect data from VitalDB

Downloads PPG, ECG and arterial blood pressure (ART) from the open VitalDB
dataset, cuts them into 10-second segments and saves them in <folder>/data.

For every segment we save:
    X       -> PPG and ECG signals, shape (2, 1000)  (10 s at 100 Hz)
    sbp/dbp -> systolic / diastolic BP (mmHg) from the arterial line   (BP labels)
    risk    -> 0 No risk, 1 Low, 2 Moderate, 3 High  from ASA status   (risk label)
    age, sex, height, weight, bmi, emop, preop_htn, preop_dm  -> patient info
    t       -> start time of the segment in the surgery (seconds)

Run on your laptop (needs internet):
    python collect_data.py --folder try2                 # all patients, 40 segments each
    python collect_data.py --folder try1 --cases 1000 --segs 20   # how try1 was made

If it stops in the middle just run it again: patients that are already saved are skipped,
even if you change --cases or --segs.
"""

import argparse
import glob
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import vitaldb

warnings.filterwarnings("ignore")

FS = 100                  # sampling rate we save at (VitalDB is 500 Hz, we downsample)
SEG_LEN = 10 * FS         # 10 second segments
TRACKS = ["SNUADC/PLETH", "SNUADC/ECG_II", "SNUADC/ART"]   # PPG, ECG, arterial BP
RISK_NAMES = ["No risk", "Low risk", "Moderate risk", "High risk"]
PATIENT_INFO = ["age", "sex", "height", "weight", "bmi", "emop", "preop_htn", "preop_dm"]
CASES_PER_PART = 100      # cases saved in each part_xxx.npz file


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
    """Download one case and return up to n_segs good segments (in time order)."""
    try:
        data = vitaldb.load_case(caseid, TRACKS, 1 / FS)
    except Exception as e:
        print(f"  case {caseid}: download failed ({e})")
        return None
    if data is None or len(data) < SEG_LEN * 10:
        return None

    ppg, ecg, art = data[:, 0], data[:, 1], data[:, 2]
    # try non-overlapping windows spread evenly over the surgery
    n_try = min(n_segs * 15, len(data) // SEG_LEN)
    starts = np.linspace(0, len(data) - SEG_LEN, n_try).astype(int)
    good = []
    for s in starts:
        res = good_segment(ppg[s:s + SEG_LEN], ecg[s:s + SEG_LEN], art[s:s + SEG_LEN])
        if res is not None:
            good.append((s, res[0], res[1]))
    if not good:
        return None
    # keep n_segs good segments spread over the whole surgery
    keep = np.unique(np.linspace(0, len(good) - 1, min(n_segs, len(good))).astype(int))
    X, sbp, dbp, t = [], [], [], []
    for k in keep:
        s, sb, db = good[k]
        X.append(np.stack([ppg[s:s + SEG_LEN], ecg[s:s + SEG_LEN]]))
        sbp.append(sb)
        dbp.append(db)
        t.append(s / FS)
    return (np.array(X, dtype=np.float16), np.array(sbp, np.float32),
            np.array(dbp, np.float32), np.array(t, np.float32))


def choose_cases(n_cases, seed=42):
    """Pick cases that have PPG + ECG + ART. n_cases=0 means all of them.
    Otherwise try to keep the 4 risk classes balanced."""
    print("Reading VitalDB case list ...")
    caseids = vitaldb.find_cases(TRACKS)
    clinical = pd.read_csv(f"{vitaldb.dataset.api_url}/cases")
    clinical = clinical[clinical["caseid"].isin(caseids)].copy()
    clinical["risk"] = clinical["asa"].apply(asa_to_risk)
    clinical = clinical[clinical["risk"] >= 0]
    clinical["sex"] = (clinical["sex"] == "M").astype(float)        # 1 = male, 0 = female
    for col in PATIENT_INFO:
        clinical[col] = pd.to_numeric(clinical[col], errors="coerce")
    print(f"{len(clinical)} cases have all 3 signals and an ASA score")
    print(clinical["risk"].value_counts().sort_index().rename(index=dict(enumerate(RISK_NAMES))))
    if n_cases <= 0 or n_cases >= len(clinical):
        return clinical.sort_values("caseid").reset_index(drop=True)

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
    parser.add_argument("--folder", default="try2", help="saves into <folder>/data")
    parser.add_argument("--cases", type=int, default=0, help="number of patients, 0 = all")
    parser.add_argument("--segs", type=int, default=40, help="10-second segments per patient")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4),
                        help="patients processed at the same time (one CPU core each)")
    args = parser.parse_args()
    out = os.path.join(args.folder, "data")
    os.makedirs(out, exist_ok=True)

    cases = choose_cases(args.cases)
    # rare classes get more segments per patient so the classes are less unbalanced
    counts = cases["risk"].value_counts()
    boost = {r: int(min(4, max(1, round(counts.max() / counts[r])))) for r in counts.index}
    print("Chosen cases per class:", {RISK_NAMES[r]: int(n) for r, n in counts.sort_index().items()})
    print("Segments per case:", {RISK_NAMES[r]: args.segs * b for r, b in sorted(boost.items())})

    # skip patients that were already saved by an earlier run
    existing = sorted(glob.glob(os.path.join(out, "part_*.npz")))
    done = set()
    for f in existing:
        done |= set(np.load(f)["caseid"].tolist())
    todo = cases[~cases["caseid"].isin(done)].reset_index(drop=True)
    first = max([int(os.path.basename(f)[5:8]) for f in existing], default=-1) + 1
    n_parts = int(np.ceil(len(todo) / CASES_PER_PART))
    print(f"Already saved: {len(done)} patients in {len(existing)} parts. To download: {len(todo)} patients "
          f"({n_parts} parts, {args.workers} at a time)")

    t_start = time.time()
    for p in range(n_parts):
        path = os.path.join(out, f"part_{first + p:03d}.npz")
        chunk = todo.iloc[p * CASES_PER_PART:(p + 1) * CASES_PER_PART]
        print(f"[{p + 1}/{n_parts}] downloading {len(chunk)} cases ...", flush=True)
        caseids = [int(c) for c in chunk["caseid"]]
        n_segs = [args.segs * boost[r] for r in chunk["risk"]]
        with ProcessPoolExecutor(args.workers) as ex:
            results = list(ex.map(process_case, caseids, n_segs))

        cols = {k: [] for k in ["X", "sbp", "dbp", "t", "risk", "asa", "caseid", "subjectid"] + PATIENT_INFO}
        for (_, row), res in zip(chunk.iterrows(), results):
            if res is None:
                continue
            x, sb, db, t = res
            n = len(x)
            cols["X"].append(x); cols["sbp"].append(sb); cols["dbp"].append(db); cols["t"].append(t)
            for k in ["risk", "asa", "caseid", "subjectid"] + PATIENT_INFO:
                cols[k].append(np.full(n, row[k]))
        if cols["X"]:
            arrays = {k: np.concatenate(v) for k, v in cols.items()}
            for k in ["risk", "caseid", "subjectid"]:
                arrays[k] = arrays[k].astype(np.int64)
            for k in ["asa"] + PATIENT_INFO:
                arrays[k] = arrays[k].astype(np.float32)
            np.savez_compressed(path, **arrays)
            print(f"   saved {path}: {len(arrays['X'])} segments, {os.path.getsize(path) / 1e6:.1f} MB")
        per_part = (time.time() - t_start) / (p + 1)
        left = per_part * (n_parts - p - 1)
        print(f"   {per_part / 60:.1f} min per part, about {left / 3600:.1f} hours left", flush=True)

    # summary of everything collected
    rows = []
    for f in sorted(os.listdir(out)):
        if f.startswith("part_") and f.endswith(".npz"):
            d = np.load(os.path.join(out, f))
            for cid in np.unique(d["caseid"]):
                m = d["caseid"] == cid
                rows.append({"caseid": cid, "subjectid": d["subjectid"][m][0], "asa": d["asa"][m][0],
                             "risk": d["risk"][m][0], "n_segments": int(m.sum()),
                             **{k: d[k][m][0] for k in PATIENT_INFO if k in d.files}})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "cases.csv"), index=False)
    print("\nDone.")
    print(f"Patients: {len(df)}   Segments: {df['n_segments'].sum()}")
    print(df.groupby("risk").agg(patients=("caseid", "count"), segments=("n_segments", "sum"))
          .rename(index=dict(enumerate(RISK_NAMES))))


if __name__ == "__main__":
    main()
