from typing import List, Optional

import pandas as pd


def standardize_model_output(
    df: pd.DataFrame,
    required_cols: Optional[List[str]] = None,
    column_dtypes: Optional[dict] = None,
    name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Ensures model outputs are consistent in schema and types.

    Parameters:
        df: The model output DataFrame
        required_cols: List of required column names (e.g. ["symbol", "alpha_score"])
        column_dtypes: Dict of expected column dtypes (e.g. {"symbol": str})
        name: Optional name for logging or debugging context

    Returns:
        Cleaned DataFrame with enforced types

    Raises:
        ValueError if required columns are missing
    """
    name = name or "model"
    df = df.copy()

    if required_cols:
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")

    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str)

    if column_dtypes:
        for col, dtype in column_dtypes.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

    return df
