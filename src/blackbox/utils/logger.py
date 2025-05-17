import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class RichLogger:
    def __init__(
        self,
        name: str = "blackbox",
        level: str = "INFO",
        log_to_console: bool = True,
        log_to_file: bool = True,
        log_file_path: Optional[str] = "results/logs/blackbox.log",
        module_filter: str = "",
        structured: bool = False,
    ):
        self.console = Console()
        self.structured = structured

        # Configure logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.upper()))
        self._logger.handlers.clear()

        if module_filter:
            self._logger.addFilter(self._build_module_filter(module_filter))

        formatter = self._get_formatter(structured)

        if log_to_console:
            self._add_console_handler(formatter)

        if log_to_file and log_file_path:
            self._add_file_handler(log_file_path, formatter)

    def _build_module_filter(self, prefix: str) -> logging.Filter:
        class ModuleFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                return record.name.startswith(prefix)

        return ModuleFilter()

    def _get_formatter(self, structured: bool) -> logging.Formatter:
        if structured:
            return logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        return logging.Formatter("%(message)s")

    def _add_console_handler(self, formatter: logging.Formatter) -> None:
        console_handler = RichHandler(
            console=self.console,
            markup=True,
            rich_tracebacks=True,
            show_path=False,
            show_level=False,
            show_time=False,
        )
        console_handler.setLevel(self._logger.level)
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

    def _add_file_handler(self, base_path: str, formatter: logging.Formatter) -> None:
        base = Path(base_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_path = base.with_name(f"{base.stem}_{timestamp}.log")
        file_handler = logging.FileHandler(full_path)
        file_handler.setLevel(self._logger.level)
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

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

    def progress(self) -> Progress:
        return Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=True,
        )
