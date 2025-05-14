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
        structured: bool = False,
        module_filter: str = "",  # e.g., "blackbox.core"
    ):
        self.console = Console()
        self.logger = logging.getLogger(name)

        level_enum = getattr(logging, level.upper())
        self.logger.setLevel(level_enum)
        self.logger.handlers.clear()  # prevent duplicates

        # Optional module name filtering
        if module_filter:

            class ModuleFilter(logging.Filter):
                def filter(self, record: logging.LogRecord) -> bool:
                    return record.name.startswith(module_filter)

            self.logger.addFilter(ModuleFilter())

        # Formatter
        formatter = (
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            if structured
            else logging.Formatter("%(message)s")
        )

        # Console handler
        if log_to_console:
            console_handler = RichHandler(
                console=self.console,
                markup=True,
                rich_tracebacks=True,
                show_path=False,
                log_time_format="%H:%M:%S",
            )
            console_handler.setLevel(level_enum)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if log_to_file:
            Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = Path(log_file_path).with_name(
                f"{Path(log_file_path).stem}_{timestamp}.log"
            )
            file_handler = logging.FileHandler(file_path)
            file_handler.setLevel(level_enum)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)

    def exception(self, msg: str):
        self.logger.exception(msg)

    def get_logger(self):
        return self.logger
