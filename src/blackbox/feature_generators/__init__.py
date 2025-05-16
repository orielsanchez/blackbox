import importlib
import pkgutil
from typing import Any

# Module-level logger (populated by set_context)
logger = None


def set_context(ctx: dict[str, Any]) -> None:
    """
    Sets global context (e.g., logger) for use in feature generators.
    Should be called once during application startup.
    """
    global logger
    logger = ctx.get("logger")


def import_all_feature_modules() -> None:
    """
    Dynamically import all submodules under this package to trigger
    side effects like class registration (via decorators).
    """
    for finder, module_name, is_pkg in pkgutil.walk_packages(
        __path__, prefix=__name__ + "."
    ):
        try:
            importlib.import_module(module_name)
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Failed to import {module_name}: {e}")
            else:
                print(f"⚠️ Failed to import {module_name}: {e}")


# Automatically import all feature modules when package is loaded
import_all_feature_modules()
