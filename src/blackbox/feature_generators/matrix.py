from typing import List, Optional

import pandas as pd
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from blackbox.config.schema import FeatureSpec
from blackbox.feature_generators.pipeline import FeaturePipeline
from blackbox.utils.context import get_logger


class FeatureMatrixGenerator:
    def __init__(self, feature_spec: List[FeatureSpec]):
        self.logger = get_logger()
        self.pipeline = FeaturePipeline(
            [{"name": f.name, "params": f.params} for f in feature_spec]
        )

    def run(
        self,
        ohlcv: pd.DataFrame,
        dates: List[pd.Timestamp],
        start_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        # Ensure proper indexing
        if ohlcv.index.names != ["date", "symbol"]:
            self.logger.warning(
                f"⚠️ Reindexing OHLCV: expected ['date', 'symbol'], got {ohlcv.index.names}"
            )
            ohlcv = ohlcv.reset_index().set_index(["date", "symbol"]).sort_index()

        self.logger.info(f"🔄 Running feature pipeline over {len(ohlcv)} rows...")
        full_features = self.pipeline.run(ohlcv)

        earliest_feature_date = full_features.index.get_level_values("date").min()
        latest_feature_date = full_features.index.get_level_values("date").max()
        self.logger.info(
            f"📐 Full feature frame range: {earliest_feature_date} → {latest_feature_date}"
        )

        # Log the earliest date for each feature to show warmup requirements
        first_dates = {}
        for column in full_features.columns:
            # Find first non-NaN date for this feature
            valid_data = full_features[column].dropna()
            if not valid_data.empty:
                first_dates[column] = valid_data.index.get_level_values("date").min()

        if first_dates:
            self.logger.info(f"🏁 Feature earliest valid dates: {first_dates}")

        if (
            not isinstance(full_features.index, pd.MultiIndex)
            or "date" not in full_features.index.names
        ):
            raise ValueError("❌ Feature output missing 'date' level in index")

        full_features = full_features[~full_features.index.duplicated(keep="first")]
        total_symbols = ohlcv.index.get_level_values("symbol").nunique()

        earliest_requested_date = min(dates) if dates else None

        # Show why dates might be missing due to warmup
        if earliest_requested_date and earliest_feature_date > earliest_requested_date:
            warmup_days = (earliest_feature_date - earliest_requested_date).days
            self.logger.warning(
                f"⚠️ First {warmup_days} days ({earliest_requested_date} to {earliest_feature_date}) "
                f"don't have feature data due to warmup requirements"
            )

        feature_frames = []
        skipped_dates = 0
        warmup_dates = 0
        missing_dates = 0

        with Progress(
            TextColumn("[bold blue]📊 Slicing features"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("slice", total=len(dates))

            for date in dates:
                progress.advance(task)

                if start_date and date < start_date:
                    skipped_dates += 1
                    continue

                # Check if date is in warmup period
                if date < earliest_feature_date:
                    warmup_dates += 1
                    self.logger.debug(
                        f"{date.date()} is in warmup period (before {earliest_feature_date.date()})"
                    )
                    continue

                mask = full_features.index.get_level_values("date") == date
                daily_features = full_features.loc[mask]

                if daily_features.empty:
                    missing_dates += 1
                    self.logger.warning(
                        f"{date.date()} | ⚠️ No features found for this date"
                    )
                    continue

                valid_symbols = daily_features.index.get_level_values(
                    "symbol"
                ).nunique()
                self.logger.debug(
                    f"{date.date()} | ✅ Valid symbols: {valid_symbols} / {total_symbols}"
                )

                feature_frames.append(daily_features)

        if skipped_dates > 0 and start_date:
            self.logger.info(
                f"⏩ Skipped {skipped_dates} dates before {start_date.date()}"
            )

        if warmup_dates > 0:
            self.logger.info(
                f"⏳ Skipped {warmup_dates} dates in warmup period (before {earliest_feature_date.date()})"
            )

        if missing_dates > 0:
            self.logger.warning(
                f"⚠️ Missing features for {missing_dates} dates after warmup period"
            )

        if not feature_frames:
            msg = "❌ No feature frames generated — check data or feature pipeline."
            self.logger.error(msg)
            raise RuntimeError(msg)

        result = pd.concat(feature_frames).sort_index()
        result.index = pd.MultiIndex.from_tuples(
            [
                (pd.to_datetime(date).normalize(), symbol)
                for date, symbol in result.index
            ],
            names=["date", "symbol"],
        )
        return result
