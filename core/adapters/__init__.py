"""
Model-specific adapters for optical flow models.

Each adapter subclasses :class:`core.base_adapter.ModelAdapter` and
composes utilities from :mod:`core.adapter_utils`.
"""

from core.adapters.dis_adapter import DISAdapter
from core.adapters.flownets_adapter import FlowNetSAdapter
from core.adapters.raft_adapter import RaftAdapter
from core.adapters.ofnet_adapter import OFNetAdapter
