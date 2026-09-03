from pathlib import Path
from typing import Optional, List
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" /"raw" / "HTRU_2.csv"

EXPECTED_COLUMNS = [
    "mean_profile",
    "std_profile",
    "kurtosis_profile",
    "skewness_profile",
    "mean_dm_snr",
    "std_dm_snr",
    "kurtosis_dm_snr",
    "skewness_dm_snr",
    "target",
]

def load_data(filepath: Optional[Path | str] = None, columns: Optional[List[str]] = None) -> pd.DataFrame:
    target_path = Path(filepath) if filepath else DEFAULT_DATA_PATH

    if not target_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {target_path.resolve()}. "
            f"Verify that the file exists or pass a custom filepath."
        )

    if not target_path.is_file():
        raise IsADirectoryError(f"Expected a file, but found directory: {target_path}")

    if target_path.stat().st_size == 0:
        raise ValueError(f"Dataset file is empty: {target_path}")

    col_names = columns if columns is not None else EXPECTED_COLUMNS
    try:
        df = pd.read_csv(target_path, header=None, names=col_names)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse CSV file at {target_path}: {exc}") from exc

    if df.empty:
        raise ValueError("Loaded DataFrame contains 0 rows.")

    if df.shape[1] != len(col_names):
        raise ValueError(
            f"Column count mismatch: expected {len(col_names)} columns, "
            f"got {df.shape[1]}."
        )

    missing_cols = set(col_names) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df

def split_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    X = df.drop(columns="target")
    y = df["target"]

    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)