from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, REPORT_DIR, RANDOM_STATE
from data_utils import build_preprocessor

print("Running Reproducibility Validation...")

X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze("columns")

def run_once():
    pipe = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("model", RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)),
    ])
    pipe.fit(X_train, y_train)
    return pipe.predict(X_test)

p1, p2 = run_once(), run_once()
max_diff = float(np.max(np.abs(p1 - p2)))
reproducible = bool(np.allclose(p1, p2, rtol=0, atol=1e-12))

report = {"execution_1_prediction_hash": hash(p1.tobytes()),
          "execution_2_prediction_hash": hash(p2.tobytes()),
          "max_absolute_difference": max_diff,
          "reproducible": reproducible}
REPORT_DIR.mkdir(parents=True, exist_ok=True)
(REPORT_DIR / "reproducibility_report.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(f"Execution 1 prediction count: {len(p1)}")
print(f"Execution 2 prediction count: {len(p2)}")
print(f"Max prediction difference: {max_diff:.12f}")
print("SUCCESS: Pipeline is reproducible." if reproducible else "FAIL: Pipeline is not reproducible.")
