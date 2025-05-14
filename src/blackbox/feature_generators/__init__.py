import importlib
import pkgutil
from typing import Any

logger = None


def set_context(ctx: dict[str, Any]):
    global logger
    logger = ctx.get("logger", None)


def import_all_feature_modules():
    for _, module_name, _ in pkgutil.walk_packages(__path__, prefix=__name__ + "."):
        importlib.import_module(module_name)


import_all_feature_modules()
