import logging
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler


class RichLogger:
    def __init__(
        self,
        name: str = "blackbox",
        level: str = "INFO",
        log_to_console: bool = True,
        log_to_file: bool = True,
        log_file_path: str = "results/logs/blackbox.log",
        structured: bool = True,
        module_filter: str = "",
    ):
        self.console = Console()
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.upper()))
        self._logger.handlers.clear()  # prevent duplicate logs on reload

        if module_filter:

            class ModuleFilter(logging.Filter):
                def filter(self, record: logging.LogRecord) -> bool:
                    return record.name.startswith(module_filter)

            self._logger.addFilter(ModuleFilter())

        # Formatter
        formatter = (
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            if structured
            else logging.Formatter("%(message)s")
        )

        # Console output via rich
        if log_to_console:
            console_handler = RichHandler(
                console=self.console,
                markup=True,
                rich_tracebacks=True,
                show_path=False,
                show_level=True,
                show_time=False,
            )
            console_handler.setLevel(self._logger.level)
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

        # Optional file logging
        if log_to_file:
            Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            full_name = Path(log_file_path).with_name(
                f"{Path(log_file_path).stem}_{timestamp}.log"
            )
            file_handler = logging.FileHandler(full_name)
            file_handler.setLevel(self._logger.level)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

    # Drop-in replacements with args/kwargs
    def debug(self, msg: str, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        self._logger.exception(msg, *args, **kwargs)

    def get_logger(self) -> logging.Logger:
        return self._logger
