import pandas as pd

from blackbox.utils.context import get_logger


def validate_feature_matrix(
    feature_matrix: pd.DataFrame, expected_dates: list[pd.Timestamp] = None
) -> None:
    logger = get_logger()

    # 1. Detect all-NaN dates
    all_nan_dates = (
        feature_matrix.groupby(level="date")
        .apply(lambda df: df.dropna(how="all").empty)
        .loc[lambda x: x]
    )
    if not all_nan_dates.empty:
        logger.warning(
            f"⚠️ {len(all_nan_dates)} dates have all-NaN features:\n"
            + ", ".join(str(d.date()) for d in all_nan_dates.index[:5])
            + ("..." if len(all_nan_dates) > 5 else "")
        )

    # 2. Detect missing dates (if expected list provided)
    if expected_dates is not None:
        actual_dates = feature_matrix.index.get_level_values("date").unique()
        missing_dates = sorted(set(expected_dates) - set(actual_dates))
        if missing_dates:
            logger.warning(
                f"⚠️ {len(missing_dates)} dates missing from feature matrix:\n"
                + ", ".join(str(d.date()) for d in missing_dates[:5])
                + ("..." if len(missing_dates) > 5 else "")
            )
