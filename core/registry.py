"""
Adapter registry - maps model names to adapters.

Usage:
    adapter = get_adapter("flownets")      # -> DefaultAdapter with FlowNetS config
    adapter = get_adapter("raft")          # -> DefaultAdapter with RAFT config
    adapter = get_adapter("my_custom", **overrides)
"""

from dataclasses import replace

from core.base_adapter import ModelAdapter
from core.adapter_config import AdapterConfig
from core.default_adapter import DefaultAdapter


# ═══════════════════════════════════════════════════════════════════════════════
# Built-in adapter configs for well-known models
# ═══════════════════════════════════════════════════════════════════════════════

ADAPTER_REGISTRY: dict[str, AdapterConfig | type[ModelAdapter]] = {
    # ── FlowNetS ─────────────────────────────
    "flownets": AdapterConfig(
        input_names=["input"],
        input_format="concat",
        normalization="unit",
        padding_factor=64,
        output_scale=20.0,
        output_resolution="quarter",
    ),
    # ── PWC-Net ──────────────────────────────
    "pwcnet": AdapterConfig(
        input_names=["input"],
        input_format="concat",
        normalization="unit",
        padding_factor=64,
        output_scale=20.0,
        output_resolution="quarter",
    ),
    # ── RAFT ─────────────────────────────────
    "raft": AdapterConfig(
        input_names=["image1", "image2"],
        normalization="none",
        padding_factor=8,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════


def list_adapters() -> list[str]:
    """Return all registered adapter names."""
    return list(ADAPTER_REGISTRY.keys())


def register_adapter(
    name: str, entry: AdapterConfig | type[ModelAdapter]
) -> None:
    """
    Register a new adapter (config or custom class).

    Args:
        name:  Lookup key (e.g. "my_model").
        entry: Either an AdapterConfig (for DefaultAdapter) or a ModelAdapter subclass.
    """
    ADAPTER_REGISTRY[name] = entry


def get_adapter(name: str, **overrides) -> ModelAdapter:
    """
    Look up an adapter by name and return a ready-to-use instance.

    Args:
        name:      Registered model name (e.g. "raft", "pwcnet").
        overrides: Keyword args to override fields in AdapterConfig.
                   Ignored for custom ModelAdapter classes.

    Returns:
        A ModelAdapter instance.

    Raises:
        KeyError: If name is not registered.
    """
    name = name.lower()
    if name not in ADAPTER_REGISTRY:
        raise KeyError(
            f"Unknown adapter '{name}'. "
            f"Available: {list_adapters()}. "
            f"Register with register_adapter()."
        )

    entry = ADAPTER_REGISTRY[name]

    if isinstance(entry, AdapterConfig):
        if overrides:
            config = replace(entry, **overrides)
        else:
            config = entry
        return DefaultAdapter(config)
    elif isinstance(entry, type) and issubclass(entry, ModelAdapter):
        return entry(**overrides)
    else:
        raise TypeError(
            f"Registry entry for '{name}' must be AdapterConfig or ModelAdapter subclass, "
            f"got {type(entry)}"
        )
