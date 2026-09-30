from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import REPORT_DIR

registry_path = REPORT_DIR / "model_registry.json"
registry = json.loads(registry_path.read_text(encoding="utf-8"))

if not registry["versions"]:
    raise ValueError("No model versions are registered.")

for v in registry["versions"]:
    if v["stage"] == "Production":
        v["stage"] = "Archived"

version = registry["versions"][-1]
version["stage"] = "Production"
registry["production_version"] = version["version"]

registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
print(f"Model version {version['version']} promoted to Production.")
