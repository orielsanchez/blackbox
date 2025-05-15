from typing import List

import pandas as pd
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from blackbox.config.schema import FeatureSpec
from blackbox.feature_generators.pipeline import FeaturePipeline
from blackbox.utils.context import get_logger


class FeatureMatrixGenerator:
    def __init__(self, feature_spec: List[FeatureSpec]):
        self.logger = get_logger()

        self.pipeline = FeaturePipeline(
            [{"name": f.name, "params": f.params} for f in feature_spec]
        )

    def run(self, data: List[dict]) -> pd.DataFrame:
        feature_frames = []

        with Progress(
            TextColumn("[bold blue]📊 Generating features"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("feature-gen", total=len(data))

            for snapshot in data:
                date = pd.Timestamp(snapshot["date"])
                ohlcv = snapshot["ohlcv"]

                self.logger.debug(f"{date.date()} | OHLCV shape: {ohlcv.shape}")
                self.logger.debug(
                    f"{date.date()} | Available symbols: {ohlcv.index.get_level_values('symbol').nunique()}"
                )

                try:
                    full_feature_df = self.pipeline.run(ohlcv)
                except Exception as e:
                    self.logger.warning(f"{date.date()} | ⚠️ Pipeline error: {e}")
                    progress.advance(task)
                    continue

                # Select features for the current date
                if "date" not in full_feature_df.index.names:
                    self.logger.warning(f"{date.date()} | ⚠️ Feature output missing 'date' level")
                    progress.advance(task)
                    continue

                mask = full_feature_df.index.get_level_values("date") == date
                today_df = full_feature_df[mask]

                if today_df.empty:
                    self.logger.warning(f"{date.date()} | ⚠️ No features found for this date")
                    progress.advance(task)
                    continue

                today_df = today_df.reset_index()
                today_df["date"] = date  # Reassert consistent date if dropped
                today_df = today_df.set_index(["date", "symbol"]).sort_index()
                today_df = today_df[~today_df.index.duplicated(keep="first")]

                valid_symbols = today_df.index.get_level_values("symbol").nunique()
                total_symbols = ohlcv.index.get_level_values("symbol").nunique()
                self.logger.debug(
                    f"{date.date()} | ✅ Valid symbols: {valid_symbols} / {total_symbols}"
                )

                feature_frames.append(today_df)
                progress.advance(task)

        if not feature_frames:
            msg = "❌ No feature frames generated — check data and feature definitions."
            self.logger.error(msg)
            raise RuntimeError(msg)

        full_matrix = pd.concat(feature_frames).sort_index()
        full_matrix = full_matrix[~full_matrix.index.duplicated(keep="first")]

        return full_matrix
