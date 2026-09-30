from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "car data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
MODEL_DIR = ARTIFACTS / "models"
REPORT_DIR = ARTIFACTS / "reports"
MLRUNS_DIR = ARTIFACTS / "mlruns"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CURRENT_YEAR = 2026

TARGET = "Selling_Price"
REQUIRED_COLUMNS = [
    "Car_Name", "Year", "Selling_Price", "Present_Price", "Kms_Driven",
    "Fuel_Type", "Seller_Type", "Transmission", "Owner"
]

CATEGORICAL_VALUES = {
    "Fuel_Type": {"Petrol", "Diesel", "CNG"},
    "Seller_Type": {"Dealer", "Individual"},
    "Transmission": {"Manual", "Automatic"},
}
