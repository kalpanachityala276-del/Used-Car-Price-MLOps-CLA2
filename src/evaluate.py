from pathlib import Path
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, MODEL_DIR, REPORT_DIR

print("Starting Model Evaluation...")

X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")

model_path = MODEL_DIR / "RandomForest.joblib"
model = joblib.load(model_path)
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

report = {
    "model": "RandomForest",
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2": float(r2),
}
REPORT_DIR.mkdir(parents=True, exist_ok=True)
(REPORT_DIR / "evaluation_report.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)

errors = X_test.copy()
errors["Actual_Selling_Price"] = y_test.values
errors["Predicted_Selling_Price"] = pred
errors["Absolute_Error"] = np.abs(y_test.values - pred)
errors.sort_values("Absolute_Error", ascending=False).to_csv(
    REPORT_DIR / "error_analysis.csv", index=False
)

print("\n--- Model Evaluation Report ---")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")
print("-------------------------------")
print("Evaluation complete and error analysis files saved!")
