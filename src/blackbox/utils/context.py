from contextlib import contextmanager
from copy import deepcopy
from typing import Any

import pandas as pd

from blackbox.utils.rich_logger import RichLogger

_context: dict[str, Any] = {}
_feature_matrix: pd.DataFrame | None = None


def set_value(key: str, value: Any):
    _context[key] = value


def get(key: str, default: Any = None) -> Any:
    return _context.get(key, default)


def has(key: str) -> bool:
    return key in _context


def clear():
    _context.clear()


def validate(required: list[str]):
    missing = [key for key in required if key not in _context]
    if missing:
        raise RuntimeError(f"❌ Context missing required keys: {missing}")


def get_logger() -> RichLogger:
    logger = get("logger")
    if not isinstance(logger, RichLogger):
        raise RuntimeError("Logger is not set or is of the wrong type.")
    return logger


@contextmanager
def scoped_context(overrides: dict[str, Any]):
    original = deepcopy(_context)
    try:
        _context.update(overrides)
        yield
    finally:
        _context.clear()
        _context.update(original)


def set_feature_matrix(matrix: pd.DataFrame):
    global _feature_matrix
    _feature_matrix = matrix


def get_feature_matrix() -> pd.DataFrame:
    if _feature_matrix is None:
        raise RuntimeError("Feature matrix not set.")
    return _feature_matrix
