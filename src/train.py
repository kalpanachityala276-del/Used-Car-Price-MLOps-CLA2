from pathlib import Path
import sys
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, MODEL_DIR, REPORT_DIR, RANDOM_STATE
from data_utils import build_preprocessor

print("Starting Model Training...")

X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze("columns")

models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=5),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
}

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

results = []
for name, estimator in models.items():
    pipe = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("model", estimator),
    ])
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, MODEL_DIR / f"{name}.joblib")
    results.append({"Model": name, "Status": "trained"})

pd.DataFrame(results).to_json(REPORT_DIR / "baseline_models.json", orient="records", indent=2)
print("Model training complete and saved to disk!")
