import importlib
import inspect
import pkgutil
from typing import Any, Type


def discover_models(package: str, interface: Type[Any]) -> dict[str, type]:
    """
    Dynamically discover all classes in a package that implement a given interface.

    Args:
        package (str): Python module path (e.g., "blackbox.models.alpha")
        interface (Type): Protocol or ABC interface to match against

    Returns:
        dict[str, type]: Map from lowercase model name attribute → class reference
    """
    registry = {}
    module = importlib.import_module(package)
    package_path = module.__path__

    for _, modname, ispkg in pkgutil.walk_packages(package_path, prefix=package + "."):
        if ispkg:
            continue
        submod = importlib.import_module(modname)

        for name, obj in inspect.getmembers(submod, inspect.isclass):
            if (
                not name.startswith("_")
                and issubclass(obj, interface)
                and obj is not interface
            ):
                try:
                    # instantiate without arguments if possible (default arguments needed)
                    instance = obj()
                    key = instance.name.lower()
                except Exception as e:
                    raise ValueError(
                        f"Failed to instantiate {obj} for registry discovery: {e}"
                    )

                registry[key] = obj

    return registry
