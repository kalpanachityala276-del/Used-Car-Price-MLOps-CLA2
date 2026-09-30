from pathlib import Path
import sys
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DATA, PROCESSED_DIR, MODEL_DIR, CURRENT_YEAR
from data_utils import load_data, prepare_dataframe, make_features_target

print("[INFO] Starting Sklearn Pipeline Preprocessing...")

df = prepare_dataframe(load_data())
X, y = make_features_target(df)

numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features),
])

print("[INFO] Fitting and transforming training data...")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
preprocessor.fit(X_train)
Xtr = preprocessor.transform(X_train)
Xte = preprocessor.transform(X_test)
print("[INFO] Transforming test data...")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(preprocessor, MODEL_DIR / "preprocessor.pkl")
joblib.dump({"feature_count": Xtr.shape[1]}, MODEL_DIR / "preprocessor_metadata.pkl")

pd.DataFrame(Xtr.toarray() if hasattr(Xtr, "toarray") else Xtr).to_csv(
    PROCESSED_DIR / "X_train_transformed.csv", index=False
)
pd.DataFrame(Xte.toarray() if hasattr(Xte, "toarray") else Xte).to_csv(
    PROCESSED_DIR / "X_test_transformed.csv", index=False
)
y_train.to_csv(PROCESSED_DIR / "y_train_pipeline.csv", index=False)
y_test.to_csv(PROCESSED_DIR / "y_test_pipeline.csv", index=False)

print("[SUCCESS] Preprocessing completed successfully! Pipeline saved as preprocessor.pkl.")
