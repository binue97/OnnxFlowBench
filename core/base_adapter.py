"""
Abstract base class for model adapters.

A ModelAdapter encapsulates ALL model-specific logic:
    - How to preprocess raw images into ONNX-ready tensors
    - How to postprocess raw ONNX outputs into standard flow format

The contract:
    preprocess:  (H,W,3) uint8 x 2   ->  dict[str, ndarray]  (ONNX input feed)
    postprocess: dict[str, ndarray]  ->  (H,W,2) float32     (flow in pixel units)
"""

from abc import ABC, abstractmethod
import numpy as np


class ModelAdapter(ABC):
    """
    Abstract adapter that bridges raw images <-> ONNX model.

    Subclass this for models whose pre/post processing cannot
    be expressed via AdapterConfig + DefaultAdapter.
    """

    @abstractmethod
    def preprocess(self, img1: np.ndarray, img2: np.ndarray) -> dict[str, np.ndarray]:
        """
        Convert two raw images into an ONNX input feed.

        Args:
            img1: First frame,  (H, W, 3) uint8, range [0, 255].
            img2: Second frame, (H, W, 3) uint8, range [0, 255].

        Returns:
            Dict of {onnx_input_name: ndarray} ready for OnnxEngine.__call__.

        Note:
            Implementations should store any state needed by postprocess()
            (e.g. original resolution, pad amounts) as instance attributes.
        """
        ...

    @abstractmethod
    def postprocess(self, outputs: dict[str, np.ndarray]) -> np.ndarray:
        """
        Convert raw ONNX outputs into standard flow.

        Args:
            outputs: Dict from OnnxEngine.__call__, keyed by output tensor name.

        Returns:
            flow: (H, W, 2) float32 in pixel units at the original input resolution.
        """
        ...
