"""
Adapter registry — maps model names to adapter classes.

Built-in adapters: ``flownets``
Register your own with :func:`register_adapter`.
"""

from core.base_adapter import ModelAdapter
from core.adapters import FlowNetSAdapter, RaftAdapter


# ═══════════════════════════════════════════════════════════════════════════════
# Built-in adapters for well-known models
# ═══════════════════════════════════════════════════════════════════════════════

ADAPTER_REGISTRY: dict[str, type[ModelAdapter]] = {
    "flownets": FlowNetSAdapter,
    "raft": RaftAdapter,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════


def list_adapters() -> list[str]:
    """Return all registered adapter names."""
    return list(ADAPTER_REGISTRY.keys())


def register_adapter(name: str, adapter_class: type[ModelAdapter]) -> None:
    """Register a new adapter class.

    Args:
        name:          Lookup key (case-insensitive at retrieval time).
        adapter_class: A :class:`ModelAdapter` **subclass** (not an instance).
    """
    ADAPTER_REGISTRY[name] = adapter_class


def get_adapter(name: str) -> ModelAdapter:
    """Look up an adapter by name and return a ready-to-use instance.

    Args:
        name: Registered model name (e.g. ``"raft"``, ``"pwcnet"``).

    Returns:
        A freshly constructed :class:`ModelAdapter` instance.

    Raises:
        KeyError:  If *name* is not registered.
        TypeError: If the registry entry is not a ``ModelAdapter`` subclass.
    """
    name = name.lower()
    if name not in ADAPTER_REGISTRY:
        raise KeyError(
            f"Unknown adapter '{name}'. "
            f"Available: {list_adapters()}. "
            f"Register with register_adapter()."
        )

    entry = ADAPTER_REGISTRY[name]

    if isinstance(entry, type) and issubclass(entry, ModelAdapter):
        return entry()
    else:
        raise TypeError(
            f"Registry entry for '{name}' must be a ModelAdapter subclass, "
            f"got {type(entry)}"
        )
