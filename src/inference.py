from pathlib import Path
import sys
import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODEL_DIR

model = joblib.load(MODEL_DIR / "RandomForest.joblib")

def predict_car(present_price, kms_driven, fuel_type, seller_type, transmission, owner, car_age):
    row = pd.DataFrame({
        "Present_Price": [present_price],
        "Kms_Driven": [kms_driven],
        "Fuel_Type": [fuel_type],
        "Seller_Type": [seller_type],
        "Transmission": [transmission],
        "Owner": [owner],
        "Car_Age": [car_age],
    })
    return float(model.predict(row)[0])
