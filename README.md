# Student Mental Stress Prediction System

A robust Python package that predicts student mental‑stress levels from behavioural, academic, and socio‑demographic data, and returns personalised coping recommendations.

## Features
* **Feature engineering** (sleep/study ratios, interactions).
* **Stacking ensemble** (RF + GBR → RidgeCV).
* **Cross‑validation** & **permutation feature‑importance** utilities.
* Optional **SHAP** visualisation.
* MIT‑licensed & dependency‑light (pandas, numpy, scikit‑learn, joblib).

## Installation
```bash
git clone <your‑fork>
cd stress_prediction_repo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start
```bash
python stress_prediction_system.py \
       --data Student_Mental_Stress_and_Coping_Mechanisms.csv
```

### With 5‑fold CV and saved artefacts
```bash
python stress_prediction_system.py \
       --data Student_Mental_Stress_and_Coping_Mechanisms.csv \
       --cv 5 \
       --save_model stress_model.joblib \
       --save_importances importance.csv
```

## Repository structure
```
stress_prediction_repo/
├── stress_prediction_system.py
├── requirements.txt
├── LICENSE
└── README.md
```

## License
This project is released under the MIT License – see `LICENSE` for details.
