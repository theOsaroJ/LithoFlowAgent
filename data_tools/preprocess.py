import pandas as pd

def align_depths(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(dfs, axis=1).sort_index()

def flag_outliers(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    df2 = df.copy()
    for col in df.columns:
        m, s = df[col].mean(), df[col].std()
        df2[f"{col}_ok"] = ((df[col] - m).abs() / s) < z_thresh
    return df2
