import argparse
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

import blackbox.feature_generators  # trigger registration
from blackbox.config.loader import load_config
from blackbox.core.live import LiveTradingEngine  # explicit live engine import
from blackbox.core.runner import run_backtest  # explicit runner import
from blackbox.core.types.context import BacktestConfig
from blackbox.models.factory import build_models
from blackbox.utils.context import set_value, validate
from blackbox.utils.logger import RichLogger


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run strategy in backtest or live mode")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/strategies/sinclair_mean_reversion.yaml",
    )
    parser.add_argument("--mode", choices=["backtest", "live"], default="backtest")
    parser.add_argument("--api-key", type=str, help="Alpaca API key")
    parser.add_argument("--secret-key", type=str, help="Alpaca API secret")
    parser.add_argument("--no-cache", action="store_true", help="Disable feature matrix cache")
    parser.add_argument("--refresh-data", action="store_true", help="Refetch OHLCV data")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()

    config: BacktestConfig = load_config(args.config)
    if args.log_level:
        config.log_level = args.log_level

    # 👇 Create output directory for this run
    run_id = getattr(config, "run_id", "default_run")
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    output_dir = Path("runs") / f"{run_id}_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(config)

    try:
        if args.mode == "backtest":
            logger.info("🔁 Mode: backtest")
            run_backtest(
                config_path=args.config,
                use_cached_features=not args.no_cache,
                refresh_data=args.refresh_data,
                plot_equity=config.plot_equity,
                output_dir=output_dir,
            )
        else:
            logger.info("📡 Mode: live")
            run_live(config, args.api_key, args.secret_key, logger)
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        raise


def setup_logger(config: BacktestConfig) -> RichLogger:
    logger = RichLogger(
        level=config.log_level,
        log_to_console=config.log_to_console,
        log_to_file=config.log_to_file,
        structured=config.structured_logging,
    )
    set_value("logger", logger)
    validate(["logger"])
    blackbox.feature_generators.set_context({"logger": logger})
    return logger


def run_live(
    config: BacktestConfig,
    api_key_arg: str | None,
    secret_key_arg: str | None,
    logger: RichLogger,
) -> None:
    api_key = api_key_arg or os.getenv("ALPACA_API_KEY")
    secret_key = secret_key_arg or os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError("❌ Alpaca credentials missing (--api-key/--secret-key or .env)")

    models = build_models(config)
    alpha = models.alpha
    risk = models.risk
    cost = models.cost
    portfolio = models.portfolio
    execution = models.execution

    universe = config.alpha_model.params.get("universe", [])
    if not universe:
        raise ValueError("❌ Missing `universe` in alpha_model.params")

    from blackbox.data.alpaca_stream import AlpacaLiveDataStream

    stream = AlpacaLiveDataStream(
        api_key=api_key,
        secret_key=secret_key,
        symbols=universe,
        timeframe="1Day",
    )

    engine = LiveTradingEngine(
        alpha=alpha,
        risk=risk,
        cost=cost,
        portfolio=portfolio,
        execution=execution,
        logger=logger,
        verbose=config.verbose,
    )
    engine.run(stream)


if __name__ == "__main__":
    main()
