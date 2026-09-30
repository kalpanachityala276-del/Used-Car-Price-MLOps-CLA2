import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def run(rel):
    print(f"\n[INFO] ---> Executing {rel}...")
    subprocess.run([sys.executable, str(ROOT / rel)], check=True)

print("[INFO] =========================================")
print("[INFO] Starting Lab 3: Baseline ML Pipeline")
print("[INFO] =========================================")
run("src/preprocess.py")
run("src/train.py")
run("src/evaluate.py")
print("\n[SUCCESS] Lab 3 Pipeline fully executed!")
