# CLA 2 Viva Notes - Used Car Price Prediction

## Project
Used Car Price Prediction is a supervised regression project because Selling_Price is continuous.

## Lab 3
A baseline pipeline loads and preprocesses data, trains models, evaluates the Random Forest baseline, saves models and creates error analysis.

## Lab 4
MLflow records experiment parameters, metrics and model artifacts. A fixed random_state makes repeated runs reproducible.

## Lab 5
Schema validation checks required columns, numeric fields and allowed categorical values. The sklearn ColumnTransformer handles numerical and categorical preprocessing consistently.

## Lab 6
The model registry stores model version, stage, dataset hash, registration time and preprocessing dependency. A version can be promoted from Staging to Production and older production versions can be archived.

## Key metrics
MAE: average absolute prediction error.
RMSE: square-root of average squared prediction error; larger errors receive more weight.
R2: proportion of target variance explained by the model; higher is generally better for the same evaluation set.

## Important
Use the original 301-row CLA 1 dataset before the final demonstration/submission. The sample CSV included in this package is only a fallback because the uploaded ZIP did not contain the dataset.
