import joblib
from pathlib import Path
from typing import Optional, List, Union, Any
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" /"raw" / "HTRU_2.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "pulsar_model.joblib"

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

def save_data(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    index: bool = False,
    overwrite: bool = True,
    **kwargs,
) -> Path:

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

    if df.empty:
        raise ValueError("Cannot save an empty DataFrame.")

    target_path = Path(filepath)

    if not target_path.is_absolute():
        target_path = PROJECT_ROOT / target_path

    if target_path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {target_path.resolve()}")

    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(target_path, index=index, **kwargs)
    except Exception as exc:
        raise IOError(f"Failed to write CSV to {target_path.resolve()}: {exc}") from exc

def save_model(
    model: Any,
    filepath: Union[str, Path] = DEFAULT_MODEL_PATH,
    overwrite: bool = True,
    compress: int = 3,
) -> Path:
    if model is None:
        raise ValueError("Cannot save None as a model.")

    if not hasattr(model, "predict") and not hasattr(model, "transform"):
        raise TypeError(f"Object of type {type(model).__name__} does not look like an ML estimator.")

    target_path = Path(filepath)

    if not target_path.is_absolute():
        target_path = PROJECT_ROOT / target_path

    if target_path.exists() and not overwrite:
        raise FileExistsError(f"Model file already exists: {target_path.resolve()}")

    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        joblib.dump(model, target_path, compress=compress)
    except Exception as exc:
        raise IOError(f"Failed to serialize model to {target_path.resolve()}: {exc}") from exc

    return target_path.resolve()

def load_model(filepath: Union[str, Path] = DEFAULT_MODEL_PATH) -> Any:
    target_path = Path(filepath)
    if not target_path.is_absolute():
        target_path = PROJECT_ROOT / target_path

    if not target_path.exists():
        raise FileNotFoundError(f"Model file not found at {target_path.resolve()}")

    return joblib.load(target_path)