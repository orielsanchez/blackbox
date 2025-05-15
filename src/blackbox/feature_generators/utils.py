import pandas as pd

from blackbox.utils.context import get_logger


def validate_feature_output(
    feature_name: str, df: pd.DataFrame, current_date: pd.Timestamp = None
) -> pd.DataFrame:
    logger = get_logger()

    if df is None or df.empty:
        logger.warning(f"{feature_name}: returned empty DataFrame")
        return pd.DataFrame()

    # Drop all-NaN rows
    if df.isna().all(axis=1).any():
        logger.warning(f"{feature_name}: dropping rows where all features are NaN")
        df = df.dropna(how="all")

    # If current_date is provided, validate that it's in the index
    if current_date is not None:
        if "date" not in df.index.names:
            logger.warning(f"{feature_name}: index missing 'date' level")
            return pd.DataFrame()
        if current_date not in df.index.get_level_values("date"):
            logger.warning(
                f"{feature_name}: current_date {current_date} missing from index"
            )
            return pd.DataFrame()

    return df
