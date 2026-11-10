# Blood Pressure and Cardiovascular Risk from PPG + ECG

Final year project, B.Tech ECE, NIT Agartala.

```
PPG + ECG (+ age, sex, BMI)  ──>  Model 1 (RNN / CNN)  ──>  Blood pressure (SBP / DBP)
PPG + ECG + BP + patient info  ──>  Model 2 (RNN / CNN)  ──>  Cardiovascular risk (No / Low / Moderate / High)
```

## Files

| File / folder | What it is |
|---|---|
| `collect_data.py` | Step 1: downloads PPG, ECG and arterial BP from VitalDB into `<try>/data/` |
| `train_kaggle.ipynb` | Step 2: trains both models on Kaggle, saves a checkpoint every epoch |
| `predict.py` | Step 3: loads trained weights and predicts BP + risk for one segment |
| `summary.txt` | All results of try1 - try4 explained |
| `models.py` | The two models; the signal part is an RNN (GRU or LSTM) or a 1D CNN |
| `try1/`, `try2/` | One folder per experiment: `data/` (collected segments) and `weights/` (trained models + `results.txt`) |

## How to run

**1. Collect data** (on a laptop with internet)
```
pip install -r requirements.txt
python collect_data.py --folder try3 --cases 2000
```
Then push the `try3/data` folder to GitHub.

**2. Train on Kaggle**: upload `train_kaggle.ipynb`, check `TRY`, `DATA` and `ARCH` in the settings cell, turn on GPU and Internet, Save & Run All.
Download everything in `/kaggle/working/<TRY>/weights/` and push it into `<TRY>/weights/`.

**3. Predict**
```
python predict.py --folder try3
```

## Experiments

| | try1 | try2 | try3 | try4 |
|---|---|---|---|---|
| Signal model | CNN | CNN | RNN (GRU) | RNN (LSTM) |
| Patients / segments | 772 / 8,034 | 992 / 29,245 | 1,746 / 55,504 | same data as try3 |
| Risk model inputs | PPG + ECG + predicted BP | + age, sex, BMI | + all pre-op clinical info | + all pre-op clinical info |
| SBP MAE (guess-the-average baseline) | 16.5 mmHg (18.3) | 15.7 mmHg (19.5) | **14.8 mmHg** (18.1) | 15.3 mmHg (18.1) |
| DBP MAE (guess-the-average baseline) | 10.8 mmHg (12.3) | 10.7 mmHg (12.3) | 9.4 mmHg (11.3) | **9.1 mmHg** (11.3) |
| Risk accuracy, segments / patients | 32.4% / 29.2% | 35.1% / 41.1% | 53.1% / 54.2% | 54.3% / 55.0% |
| Risk macro F1, segments / patients | 0.283 / 0.210 | 0.317 / 0.350 | 0.431 / 0.456 | 0.435 / 0.450 |
| Risk, patient info only (gradient boosting, no signals) | – | – | **61.8%** (F1 0.526) | 61.8% (F1 0.526) |

**What the numbers mean** (full explanation in [`summary.txt`](summary.txt))

- **BP:** every model beats "always guess the average". Best: GRU for SBP (14.8 mmHg), LSTM for DBP (9.1 mmHg). Not yet clinical grade: the AAMI standard needs an error spread below 8 mmHg, ours is about 19 / 12 mmHg.
- **Risk:** about 55% with the deep learning model, 62% with clinical info alone (random guessing = 25%). 92-96% of patients are in the right class or the one next to it.
- **Why not higher:** extra tests (more features, more patients, heart rate, even the true measured BP) all stop at 63-65%. ASA is a judgement made by the anaesthetist, so the label itself limits the accuracy. The signals help BP prediction but add little to the risk class.
- GRU, LSTM and CNN, and the activation functions ReLU / GELU / SiLU, gave about the same results. More data helped more.

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
