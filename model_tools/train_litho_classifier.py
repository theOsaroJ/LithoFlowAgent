import joblib
from xgboost import XGBClassifier

def train_litho_classifier(df, label_col, feature_cols, out_path):
    train = df.dropna(subset=[label_col] + feature_cols)
    X, y = train[feature_cols], train[label_col]
    clf = XGBClassifier(n_estimators=100, tree_method="hist")
    clf.fit(X, y)
    joblib.dump(clf, out_path)
