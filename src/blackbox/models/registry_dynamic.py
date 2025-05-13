import importlib
import inspect
import pkgutil
import sys
from pathlib import Path
from typing import Any, Type

_registry_cache: dict[str, dict[str, type]] = {}


def discover_models(module_path: str, interface: Type[Any]) -> dict[str, type]:
    """
    Dynamically discover all model classes in a module directory that match a given interface.

    Args:
        module_path (str): Filesystem path (e.g. 'src/blackbox/models/alpha')
        interface (Type): Base class or Protocol (e.g. AlphaModel)

    Returns:
        dict[str, type]: model_name → class reference
    """
    if module_path in _registry_cache:
        return _registry_cache[module_path]

    path = Path(module_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"❌ Module path does not exist: {path}")

    # Guess import path (e.g. blackbox.models.alpha)
    import_base = infer_import_base(path)

    # Add project root/src to sys.path
    project_root = find_project_root(path)
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    discovered: dict[str, type] = {}

    for _, module_name, _ in pkgutil.iter_modules([str(path)]):
        full_import_path = f"{import_base}.{module_name}"
        try:
            module = importlib.import_module(full_import_path)
        except Exception as e:
            print(f"⚠️ Could not import {full_import_path}: {e}")
            continue

        for _, cls in inspect.getmembers(module, inspect.isclass):
            if not hasattr(cls, "name") or not isinstance(cls.name, str):
                continue

            # Try structural or subclass match
            if _matches_interface(cls, interface):
                discovered[cls.name] = cls

    _registry_cache[module_path] = discovered
    return discovered


def _matches_interface(cls: type, interface: Type) -> bool:
    """Returns True if cls matches interface by subclass or method presence."""
    try:
        return issubclass(cls, interface)
    except TypeError:
        # For Protocols with non-method members (e.g. 'name'), use duck typing
        required = [m for m in dir(interface) if not m.startswith("_")]
        return all(hasattr(cls, attr) for attr in required)


def infer_import_base(path: Path) -> str:
    parts = list(path.parts)
    if "src" in parts:
        parts = parts[parts.index("src") + 1 :]

    try:
        idx = parts.index("blackbox")
        return ".".join(parts[idx:])
    except ValueError:
        raise ValueError(f"❌ Could not infer import path from {path}")


def find_project_root(path: Path) -> Path:
    for parent in path.parents:
        if (parent / "src").exists():
            return parent
    raise FileNotFoundError("❌ Could not find project root containing 'src/'")
