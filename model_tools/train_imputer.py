import joblib
from xgboost import XGBRegressor

def train_imputer(df, target_cols, feature_cols, out_path):
    models = {}
    for tgt in target_cols:
        train = df.dropna(subset=[tgt] + feature_cols)
        X, y = train[feature_cols], train[tgt]
        m = XGBRegressor(n_estimators=100, tree_method="hist")
        m.fit(X, y)
        models[tgt] = m
    joblib.dump(models, out_path)
