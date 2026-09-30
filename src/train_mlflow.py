from pathlib import Path
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, MODEL_DIR, REPORT_DIR, MLRUNS_DIR, RANDOM_STATE
from data_utils import build_preprocessor, dataset_hash

print("\n--- Starting MLflow Run: RandomForest ---")

X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze("columns")
y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")

params = {
    "n_estimators": 200,
    "random_state": RANDOM_STATE,
    "test_size": 0.20,
    "model": "RandomForestRegressor",
}
model = Pipeline([
    ("preprocessor", build_preprocessor(X_train)),
    ("model", RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)),
])
model.fit(X_train, y_train)
pred = model.predict(X_test)

metrics = {
    "MAE": float(mean_absolute_error(y_test, pred)),
    "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
    "R2": float(r2_score(y_test, pred)),
    "dataset_sha256": dataset_hash(),
}

MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
local_run = {
    "run_name": "RandomForest",
    "params": params,
    "metrics": metrics,
    "artifacts": ["RandomForest.joblib"],
}
(MLRUNS_DIR / "latest_run.json").write_text(json.dumps(local_run, indent=2), encoding="utf-8")
joblib.dump(model, MODEL_DIR / "RandomForest_mlflow.joblib")

try:
    import mlflow
    mlflow.set_tracking_uri((MLRUNS_DIR / "mlruns").as_uri())
    mlflow.set_experiment("Used_Car_Price_Prediction")
    with mlflow.start_run(run_name="RandomForest") as run:
        mlflow.log_params(params)
        mlflow.log_metrics({k:v for k,v in metrics.items() if isinstance(v, float)})
        mlflow.log_artifact(str(MODEL_DIR / "RandomForest_mlflow.joblib"))
        mlflow.set_tag("dataset_sha256", metrics["dataset_sha256"])
        local_run["mlflow_run_id"] = run.info.run_id
    print(f"Metrics logged: MAE = {metrics['MAE']:.4f} | R2 = {metrics['R2']:.4f}")
    print("MLflow run successfully tracked!")
except ImportError:
    print("MLflow is not installed in this environment.")
    print("Install it with: pip install mlflow")
    print("A local tracking manifest was still saved for reproducibility.")

