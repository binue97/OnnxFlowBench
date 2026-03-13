"""
FlowModel - the single user-facing entry point.

Composes OnnxEngine (raw ONNX inference) + ModelAdapter (pre/post processing)
API:
    model = FlowModel("raft.onnx", adapter="raft")
    flow  = model.predict(img1, img2)
"""

import numpy as np

from core.onnx_engine import OnnxEngine
from core.base_adapter import ModelAdapter
from core.registry import get_adapter


class FlowModel:
    """Optical flow model that combines ONNXEngine + ModelAdapter."""

    def __init__(
        self,
        onnx_path: str,
        adapter: str | ModelAdapter = "raft",
        device: str = "cuda",
    ):
        """
        Args:
            onnx_path: Path to the .onnx model file.
            adapter:   One of:
                       - str: registered adapter name (e.g. "flownets", "raft")
                       - ModelAdapter: a ready-to-use adapter instance
            device:    "cuda" or "cpu".
        """
        self.engine = OnnxEngine(onnx_path, device=device)
        self.adapter = self._resolve_adapter(adapter)

    @staticmethod
    def _resolve_adapter(adapter) -> ModelAdapter:
        """Convert the adapter argument into a ModelAdapter instance."""
        if isinstance(adapter, str):
            return get_adapter(adapter)
        elif isinstance(adapter, ModelAdapter):
            return adapter
        else:
            raise TypeError(
                f"adapter must be str or ModelAdapter, "
                f"got {type(adapter).__name__}"
            )

    def predict(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
    ) -> np.ndarray:
        """
        Run optical flow prediction on an image pair.

        Args:
            img1: First frame,  (H, W, 3) uint8, range [0, 255].
            img2: Second frame, (H, W, 3) uint8, range [0, 255].

        Returns:
            flow: (H, W, 2) float32 in pixel units at original resolution.
        """
        feed = self.adapter.preprocess(img1, img2)

        self._save_tensor("/home/bnu/workspace/FlowEval/resources/raft/", "inputs", feed)

        outputs = self.engine(feed)

        self._save_tensor("/home/bnu/workspace/FlowEval/resources/raft/", "outputs", outputs)
        
        flow = self.adapter.postprocess(outputs)
        return flow

    @staticmethod
    def _save_tensor(dir: str, tag: str, tensors: dict[str, np.ndarray]):
        """Save each tensor in *tensors* as  <dir>/<tag>_<name>.npy."""
        import os
        os.makedirs(dir, exist_ok=True)
        for name, arr in tensors.items():
            path = os.path.join(dir, f"{tag}_{name}.npy")
            np.save(path, arr)
            print(f"[debug] saved {path}  shape={arr.shape}  dtype={arr.dtype}")

    def __repr__(self) -> str:
        adapter_name = type(self.adapter).__name__
        return f"FlowModel(\n  engine={self.engine!r},\n  adapter={adapter_name}\n)"
