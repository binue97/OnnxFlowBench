"""
Model-specific adapters for optical flow models.

Each adapter subclasses :class:`~core.base_adapter.ModelAdapter` and
composes utilities from :mod:`core.adapter_utils`.

Built-in adapters:
    - :class:`FlowNetSAdapter` — FlowNetS
    - :class:`PWCNetAdapter`   — PWC-Net
    - :class:`RAFTAdapter`     — RAFT
"""

from core.adapters.flownets_adapter import FlowNetSAdapter
from core.adapters.pwcnet_adapter import PWCNetAdapter
from core.adapters.raft_adapter import RAFTAdapter
