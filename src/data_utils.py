import hashlib
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from config import (
    RAW_DATA, PROCESSED_DIR, CURRENT_YEAR, RANDOM_STATE, TEST_SIZE,
    TARGET, REQUIRED_COLUMNS
)

def load_data(path=RAW_DATA):
    return pd.read_csv(path)

def prepare_dataframe(df):
    df = df.copy()
    df = df.drop_duplicates().reset_index(drop=True)
    df["Car_Age"] = CURRENT_YEAR - pd.to_numeric(df["Year"], errors="coerce")
    return df

def make_features_target(df):
    X = df.drop(columns=[TARGET, "Car_Name", "Year"], errors="ignore")
    y = df[TARGET].astype(float)
    return X, y

def split_data(df):
    X, y = make_features_target(df)
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

def build_preprocessor(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(include=["object", "string"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])

def dataset_hash(path=RAW_DATA):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
