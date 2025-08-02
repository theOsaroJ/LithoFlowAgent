import pandas as pd
from xgboost import XGBRegressor
import joblib

def train_imputer(df: pd.DataFrame, target_cols: list[str], feature_cols: list[str], out_path: str):
    """
    Train an XGBoost regressor for each target column using the specified features.
    Saves a dict of trained models to `out_path`.
    """
    models = {}
    for tgt in target_cols:
        train = df.dropna(subset=[tgt] + feature_cols)
        X, y = train[feature_cols], train[tgt]
        model = XGBRegressor(n_estimators=100, tree_method="hist")
        model.fit(X, y)
        models[tgt] = model
    joblib.dump(models, out_path)
    print(f"Saved imputer models to {out_path}")
