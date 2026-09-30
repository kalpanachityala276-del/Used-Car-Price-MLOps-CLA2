from pathlib import Path
import sys
import json
import joblib
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODEL_DIR, REPORT_DIR
from data_utils import dataset_hash

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

model_path = MODEL_DIR / "RandomForest.joblib"
if not model_path.exists():
    raise FileNotFoundError("Run train.py before model registration.")

registry_path = REPORT_DIR / "model_registry.json"
if registry_path.exists():
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
else:
    registry = {"model_name": "UsedCarPriceRandomForest", "versions": []}

version = len(registry["versions"]) + 1
entry = {
    "version": version,
    "model_file": str(model_path.relative_to(model_path.parents[2])),
    "stage": "Staging",
    "status": "Validated",
    "dataset_sha256": dataset_hash(),
    "registered_at": datetime.now().isoformat(timespec="seconds"),
    "preprocessing_dependency": "src/data_utils.py + scikit-learn Pipeline",
}
registry["versions"].append(entry)
registry["current_staging_version"] = version

registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
print(f"Registered model version {version} in Staging.")
print(f"Registry saved to: {registry_path}")
