from pathlib import Path
import sys
import json
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DATA, REQUIRED_COLUMNS, CATEGORICAL_VALUES, REPORT_DIR

print(f"[INFO] Validating schema (File: {RAW_DATA.name})...")
df = pd.read_csv(RAW_DATA)

errors = []
missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing:
    errors.append(f"Missing columns: {missing}")

for col in ["Year", "Selling_Price", "Present_Price", "Kms_Driven", "Owner"]:
    if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
        errors.append(f"{col} must be numeric")

for col, allowed in CATEGORICAL_VALUES.items():
    if col in df.columns:
        bad = sorted(set(df[col].dropna().astype(str)) - allowed)
        if bad:
            errors.append(f"Invalid values in {col}: {bad}")

if df.isnull().sum().sum() > 0:
    print("[WARNING] Missing values found; preprocessing will handle them.")

report = {"records": len(df), "columns": list(df.columns), "errors": errors, "passed": not errors}
REPORT_DIR.mkdir(parents=True, exist_ok=True)
(REPORT_DIR / "data_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

if errors:
    raise ValueError("Schema Validation FAILED: " + " | ".join(errors))
print("[SUCCESS] Schema Validation PASSED. Dataset structure is valid.")
