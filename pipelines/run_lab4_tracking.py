import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def run(rel):
    print(f"\n[INFO] ---> Executing {rel}...")
    subprocess.run([sys.executable, str(ROOT / rel)], check=True)

print("[INFO] =========================================")
print("[INFO] Starting Lab 4: MLflow Experiment Tracking")
print("[INFO] =========================================")
run("src/preprocess.py")
run("src/train_mlflow.py")
run("src/validate_reproducibility.py")
print("\n[SUCCESS] Lab 4 Pipeline fully executed!")
