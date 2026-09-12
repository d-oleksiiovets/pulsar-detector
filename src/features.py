import pandas as pd

def add_coefficient_of_variation(df: pd.DataFrame) -> pd.DataFrame:
    """Add the coefficient of variation as a derived feature."""

    df = df.copy()
    df["cv_profile"] = df["std_profile"] / df["mean_profile"]
    return df