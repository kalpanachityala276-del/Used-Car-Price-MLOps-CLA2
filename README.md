# CLA 2 - Used Car Price Prediction with MLOps

## Student Project
This project continues the uploaded CLA 1 project: **Used Car Price Prediction using Machine Learning**.

The original notebook is preserved under `notebooks/`.

## Labs implemented

### Lab 3 - Baseline ML Pipeline
- Reproducible preprocessing
- Train/test split
- Linear Regression, Decision Tree, Random Forest and Gradient Boosting
- Model serialization with Joblib
- Evaluation using MAE, RMSE and R2
- Error analysis report

Run:
```powershell
python .\pipelines\run_lab3_baseline.py
```

### Lab 4 - MLflow Tracking & Reproducibility
- Random Forest experiment
- Parameter and metric tracking
- Model artifact logging
- Dataset SHA-256 lineage
- Repeated execution reproducibility check

Run:
```powershell
python .\pipelines\run_lab4_tracking.py
```

If MLflow is not installed:
```powershell
pip install mlflow
```

### Lab 5 - Data Validation & Production Preprocessing
- Schema validation
- Datatype validation
- Categorical-value validation
- Missing-value checks
- Reusable sklearn preprocessing pipeline
- Output validation and consistency checks

Run:
```powershell
python .\pipelines\run_lab5_pipeline.py
```

### Lab 6 - Model Registry & Version Management
- Model version registration
- Dataset lineage
- Staging/Production lifecycle
- Model promotion and traceability

Run:
```powershell
python .\pipelines\run_lab6_registry.py
```

## Important dataset note

The uploaded CLA 1 ZIP did **not** contain the original full `car data.csv`. A small sample CSV is included so the scripts are executable for demonstration.

For the actual CLA 2 submission, replace:
`data/raw/car data.csv`
with the **same full 301-record CarDekho dataset used in CLA 1**.

Do not change the column names.

Expected columns:
`Car_Name, Year, Selling_Price, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner`

## Project structure

```text
Used_Car_Price_MLOps_CLA2/
├── data/
│   ├── raw/
│   │   ├── car data.csv
│   │   └── README.txt
│   └── processed/
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── train_mlflow.py
│   ├── validate_reproducibility.py
│   ├── validate_data.py
│   ├── preprocess_pipeline.py
│   ├── validate_outputs.py
│   ├── register_model.py
│   ├── promote_model.py
│   ├── inference.py
│   ├── data_utils.py
│   └── config.py
├── pipelines/
│   ├── run_lab3_baseline.py
│   ├── run_lab4_tracking.py
│   ├── run_lab5_pipeline.py
│   └── run_lab6_registry.py
├── artifacts/
├── notebooks/
│   └── car_price_prediction_new.ipynb
├── docs/
│   └── 241FA18305 research paper.docx
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Create/activate your virtual environment and run:
```powershell
pip install -r requirements.txt
```

## GitHub

Suggested commands:
```powershell
git init
git add .
git commit -m "CLA2: add MLOps Labs 3-6 to car price prediction"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

## Viva points

Be ready to explain:
1. Why this is a regression problem.
2. Why `Car_Age` is created.
3. Why preprocessing is inside a Pipeline.
4. What MAE, RMSE and R2 mean.
5. What MLflow tracks.
6. Why a fixed random state is used.
7. How schema validation prevents bad input data.
8. What idempotent/reproducible preprocessing means.
9. Why models are serialized.
10. How model versions move from Staging to Production.
