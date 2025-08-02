import os
import pandas as pd
import lasio
import logging
from config import SCHEMA_MAP

logger = logging.getLogger(__name__)

def _read_file(path: str) -> pd.DataFrame:
    if path.lower().endswith(".las"):
        las = lasio.read(path)
        df = las.df()
        df.index.name = "DEPT"
        return df
    df = pd.read_csv(path)
    for src in ["DEPTH", "depth_m", "dept"]:
        if src in df.columns:
            df = df.rename(columns={src: "DEPT"})
            break
    return df.set_index("DEPT")

def apply_schema_map(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {src: cfg["name"] for src, cfg in SCHEMA_MAP.items()}
    df = df.rename(columns=rename_map)
    unmapped = set(df.columns) - {cfg["name"] for cfg in SCHEMA_MAP.values()}
    if unmapped:
        logger.warning(f"Unmapped columns: {sorted(unmapped)}")
    for src, cfg in SCHEMA_MAP.items():
        tgt = cfg["name"]
        if tgt in df.columns:
            df[tgt] = df[tgt] * cfg.get("scale", 1.0) + cfg.get("offset", 0.0)
    return df

def load_and_map_all(directory: str) -> dict[str, pd.DataFrame]:
    logs = {}
    for root, _, files in os.walk(directory):
        for fn in files:
            if fn.lower().endswith((".las", ".csv")):
                path = os.path.join(root, fn)
                df = _read_file(path)
                logs[fn] = apply_schema_map(df)
    return logs
