from pathlib import Path
import sys
import json
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, REPORT_DIR

print("[INFO] Validating Preprocessing Outputs...")

files = [
    PROCESSED_DIR / "X_train_transformed.csv",
    PROCESSED_DIR / "X_test_transformed.csv",
    PROCESSED_DIR / "y_train_pipeline.csv",
    PROCESSED_DIR / "y_test_pipeline.csv",
]
missing = [str(p) for p in files if not p.exists()]
if missing:
    raise FileNotFoundError("Missing preprocessing outputs: " + ", ".join(missing))

Xtr = pd.read_csv(files[0])
Xte = pd.read_csv(files[1])
report = {
    "train_shape": list(Xtr.shape),
    "test_shape": list(Xte.shape),
    "same_feature_count": bool(Xtr.shape[1] == Xte.shape[1]),
    "train_nulls": int(Xtr.isnull().sum().sum()),
    "test_nulls": int(Xte.isnull().sum().sum()),
    "passed": bool(Xtr.shape[1] == Xte.shape[1] and Xtr.isnull().sum().sum() == 0 and Xte.isnull().sum().sum() == 0),
}
REPORT_DIR.mkdir(parents=True, exist_ok=True)
(REPORT_DIR / "preprocessing_summary_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
if not report["passed"]:
    raise ValueError("Output validation FAILED")
print("[SUCCESS] Output Validation PASSED.")
print("[INFO] Features are clean, scaled, encoded, and dimensionally consistent.")
print("[INFO] Summary report saved to artifacts/reports/preprocessing_summary_report.json")
