import argparse
import os
from datetime import datetime

from dotenv import load_dotenv

from blackbox.data.ohlcv_updater import PolygonOHLCVUpdater
from blackbox.utils.logger import RichLogger


def main():
    if not os.path.exists(".env"):
        raise RuntimeError("❌ .env file not found — please create one and set POLYGON_API_KEY")

    load_dotenv()

    if not os.getenv("POLYGON_API_KEY"):
        raise RuntimeError(
            "❌ POLYGON_API_KEY not found in environment. Please check your .env file."
        )

    parser = argparse.ArgumentParser(description="Fetch and update OHLCV data from Polygon.io")
    parser.add_argument("--db-path", type=str, default="db/ohlcv.duckdb")
    parser.add_argument("--universe", type=str, default="universe/top_2000_by_volume.csv")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--sleep", type=float, default=1.0, help="Sleep between batches (seconds)"
    )  # ignored now
    parser.add_argument("--lookback-days", type=int, default=2, help="How many days to look back")
    parser.add_argument(
        "--refetch-days", type=int, default=3, help="How many recent days to re-fetch"
    )
    parser.add_argument(
        "--full-backfill", action="store_true", help="Download entire 5-year history"
    )
    parser.add_argument("--log-file", type=str, default=None, help="Optional log file path")
    parser.add_argument("--dry-run", action="store_true", help="Simulate only, do not insert")
    parser.add_argument("--max-workers", type=int, default=40, help="Parallel thread count")
    args = parser.parse_args()

    # Create logger
    log_file = args.log_file or f"logs/fetch_ohlcv_{datetime.utcnow().date()}.log"
    logger = RichLogger(log_to_console=True, log_to_file=log_file, level="INFO")

    logger.info("🚀 Starting OHLCV update job")

    updater = PolygonOHLCVUpdater(
        db_path=args.db_path,
        table="daily_data",
        universe_path=args.universe,
        batch_size=args.batch_size,
        full_backfill=args.full_backfill,
        lookback_days=args.lookback_days,
        refetch_days=args.refetch_days,
        dry_run=args.dry_run,
        logger=logger,
        max_workers=args.max_workers,
    )

    updater.run()
    logger.info("✅ OHLCV update complete")


if __name__ == "__main__":
    main()
