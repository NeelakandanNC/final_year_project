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
| Patients / segments | 772 / 8,034 | 992 / 29,245 | try3 data | try3 data |
| Risk model inputs | PPG + ECG + predicted BP | + age, sex, BMI | + all pre-op clinical info | + all pre-op clinical info |
| SBP MAE (guess-the-average baseline) | 16.5 mmHg (18.3) | 15.7 mmHg (19.5) | | |
| DBP MAE (guess-the-average baseline) | 10.8 mmHg (12.3) | 10.7 mmHg (12.3) | | |
| Risk accuracy, segments / patients | 32.4% / 29.2% | 35.1% / 41.1% | | |
| Risk macro F1, segments / patients | 0.283 / 0.210 | 0.317 / 0.350 | | |

Random guessing for 4 risk classes = 25% accuracy. Full numbers are in each `weights/results.txt`.

**Quick model test on try2 data** (BP model, 10 segments per patient, 15 epochs, same test patients):

| Model | Parameters | SBP MAE | DBP MAE |
|---|---|---|---|
| Guess the average | – | 19.41 | 12.37 |
| RNN, GRU | 445k | 16.78 | 10.80 |
| RNN, LSTM | 582k | 16.93 | 10.86 |
| CNN, ReLU | 196k | 16.18 | 10.38 |
| CNN, SiLU | 196k | 16.50 | 10.42 |
| CNN, GELU | 196k | 16.64 | 10.37 |

Patient information alone (age, sex, BMI, emergency surgery, hypertension, diabetes, gradient boosting, 5-fold) predicts the risk class with about 50% accuracy on the try2 patients.

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
