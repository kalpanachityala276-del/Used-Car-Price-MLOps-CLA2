import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def run(rel):
    print(f"\n[INFO] ---> Executing {rel}...")
    subprocess.run([sys.executable, str(ROOT / rel)], check=True)

print("[INFO] =========================================")
print("[INFO] Starting Lab 5: Production Data Pipeline")
print("[INFO] =========================================")
run("src/validate_data.py")
run("src/preprocess_pipeline.py")
run("src/validate_outputs.py")
print("\n[SUCCESS] Lab 5 Production Pipeline fully executed!")
