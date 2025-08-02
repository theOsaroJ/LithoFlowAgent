import pandas as pd

def align_depths(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge multiple depth-indexed DataFrames on their union of depths.
    Missing values are filled with NaN.
    """
    return pd.concat(dfs, axis=1).sort_index()

def flag_outliers(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    """
    Add boolean mask columns `<col>_ok` marking non-outliers by z-score.
    """
    df2 = df.copy()
    for col in df.columns:
        m, s = df[col].mean(), df[col].std()
        z = (df[col] - m).abs() / s
        df2[f"{col}_ok"] = z < z_thresh
    return df2
