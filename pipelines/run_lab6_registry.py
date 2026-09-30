import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def run(rel):
    print(f"\n[INFO] ---> Executing {rel}...")
    subprocess.run([sys.executable, str(ROOT / rel)], check=True)

print("[INFO] =========================================")
print("[INFO] Starting Lab 6: Model Registry & Version Management")
print("[INFO] =========================================")
run("src/preprocess.py")
run("src/train.py")
run("src/evaluate.py")
run("src/register_model.py")
run("src/promote_model.py")
print("\n[SUCCESS] Lab 6 Model Registry Pipeline fully executed!")
