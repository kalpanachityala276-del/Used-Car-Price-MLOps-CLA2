from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DATA, PROCESSED_DIR, REPORT_DIR, REQUIRED_COLUMNS
from data_utils import load_data, prepare_dataframe, split_data, dataset_hash

print("Starting Preprocessing Pipeline...")

df = load_data()
missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing_required:
    raise ValueError(f"Missing required columns: {missing_required}")

clean = prepare_dataframe(df)
X_train, X_test, y_train, y_test = split_data(clean)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

summary = {
    "rows_before": int(len(df)),
    "rows_after_duplicates": int(len(clean)),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "columns": list(df.columns),
    "dataset_sha256": dataset_hash(),
    "current_year_for_car_age": 2026,
}
(REPORT_DIR / "preprocessing_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print("Preprocessing completed successfully!")
