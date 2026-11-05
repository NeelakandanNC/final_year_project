# Blood Pressure and Cardiovascular Risk from PPG + ECG

Final year project, B.Tech ECE, NIT Agartala.

```
PPG + ECG  ──>  Model 1 (CNN)  ──>  Blood pressure (SBP / DBP)
PPG + ECG + BP  ──>  Model 2 (CNN)  ──>  Cardiovascular risk (No / Low / Moderate / High)
```

## Files

| File | What it does |
|---|---|
| `collect_data.py` | Step 1: downloads PPG, ECG and arterial BP from VitalDB into `data/` |
| `train_kaggle.ipynb` | Step 2: trains both models on Kaggle, saves a checkpoint every epoch |
| `predict.py` | Step 3: loads the trained weights and predicts BP + risk |
| `models.py` | The two CNN models |
| `data/` | Collected 10-second segments (`part_xxx.npz`) |
| `weights/` | Final trained weights (`bp_cnn.pt`, `risk_cnn.pt`) |

## How to run

**1. Collect data** (once, on a laptop with internet)
```
pip install -r requirements.txt
python collect_data.py
```
Then push the `data/` folder to GitHub.

**2. Train on Kaggle**: upload `train_kaggle.ipynb`, turn on GPU and Internet, Run All.
Download `bp_cnn.pt` and `risk_cnn.pt` from `/kaggle/working/weights/` and push them into `weights/`.

**3. Predict**
```
python predict.py
```

## Data and labels

- Dataset: [VitalDB](https://vitaldb.net/dataset) (open dataset of surgical patients, Seoul National University Hospital)
- Signals: `SNUADC/PLETH` (PPG), `SNUADC/ECG_II` (ECG), `SNUADC/ART` (arterial line), 10 s segments at 100 Hz
- BP label: SBP and DBP measured from the arterial line in the same 10 s
- Risk label: from the patient's ASA physical status

| ASA | Risk class |
|---|---|
| 1 | No risk |
| 2 | Low risk |
| 3 | Moderate risk |
| 4 or more | High risk |

Train / validation / test split is done by patient so no patient appears in two sets.

Reference: Lee HC et al. *VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients.* Scientific Data 9, 279 (2022).
