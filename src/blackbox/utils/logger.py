import blackbox.feature_generators
from blackbox.core.types.dataclasses import BacktestConfig
from blackbox.utils.context import set_value, validate
from blackbox.utils.rich_logger import RichLogger


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
