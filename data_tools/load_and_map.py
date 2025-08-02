import os
import pandas as pd
import lasio
import logging
from config import SCHEMA_MAP

logger = logging.getLogger(__name__)

def _read_file(path: str) -> pd.DataFrame:
    """Read a LAS or CSV file into a depth-indexed DataFrame."""
    if path.lower().endswith(".las"):
        las = lasio.read(path)
        df = las.df()
        df.index.name = "DEPT"
        return df
    else:
        df = pd.read_csv(path)
        # Normalize common depth column names
        for src in ["DEPTH", "depth_m", "dept"]:
            if src in df.columns:
                df = df.rename(columns={src: "DEPT"})
                break
        return df.set_index("DEPT")

def apply_schema_map(df: pd.DataFrame, domain: str = "well_logs") -> pd.DataFrame:
    """Rename and scale columns according to the schema map."""
    mapping = SCHEMA_MAP.get(domain, {})
    # Rename columns
    rename_dict = {src: cfg["name"] for src, cfg in mapping.items()}
    df_renamed = df.rename(columns=rename_dict)
    # Detect unmapped columns
    unmapped = set(df_renamed.columns) - set(cfg["name"] for cfg in mapping.values())
    if unmapped:
        logger.warning(f"Unmapped columns detected: {sorted(unmapped)}")
    # Apply scaling and offset
    for src, cfg in mapping.items():
        tgt = cfg["name"]
        if tgt in df_renamed.columns:
            df_renamed[tgt] = df_renamed[tgt] * cfg.get("scale", 1.0) + cfg.get("offset", 0.0)
    return df_renamed

def load_and_map_all(directory: str, domain: str = "well_logs") -> dict[str, pd.DataFrame]:
    """Recursively load and schema-map all LAS/CSV files in a directory."""
    logs = {}
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.lower().endswith((".las", ".csv")):
                path = os.path.join(root, fname)
                df = _read_file(path)
                logs[fname] = apply_schema_map(df, domain)
    return logs
