import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_symbols_from_hive(
    data_dir: str = "data/ohlcv/minute/hive_parquet",
) -> list[str]:
    base_path = Path(data_dir)
    symbol_dirs = base_path.glob("symbol=*")

    symbols = {p.name.split("=")[1] for p in symbol_dirs if "=" in p.name}
    logger.info(f"Found {len(symbols)} unique symbols from directory structure")
    return sorted(symbols)


def save_symbols_to_csv(symbols: list[str], out_path: str = "universe/top_symbols.csv"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(symbols, name="symbol").to_csv(out_path, index=False)
    logger.info(f"Saved {len(symbols)} symbols to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and save symbol universe from Hive-style parquet data"
    )
    parser.add_argument(
        "--data-dir",
        default="data/ohlcv/minute/hive_parquet",
        help="Path to Hive-style parquet root",
    )
    parser.add_argument(
        "--output", default="universe/top_symbols.csv", help="Output CSV file path"
    )
    parser.add_argument(
        "--top-n", type=int, default=None, help="Limit to top N symbols"
    )
    args = parser.parse_args()

    symbols = extract_symbols_from_hive(args.data_dir)
    if args.top_n:
        symbols = symbols[: args.top_n]
        logger.info(f"Trimmed to top {args.top_n} symbols")

    save_symbols_to_csv(symbols, args.output)


if __name__ == "__main__":
    main()
