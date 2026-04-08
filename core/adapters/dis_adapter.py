"""
OpenCV DIS optical flow adapter (standalone — no ONNX model needed).
"""

import numpy as np

from core.base_adapter import ModelAdapter
from core.registry import register


@register("dis")
class DISAdapter(ModelAdapter):
    """Dense Inverse Search optical flow via OpenCV.

    This is a standalone adapter: it handles prediction directly
    without an ONNX engine.  Use ``--adapter dis`` with no ``--model``.
    """

    PRESETS = ("ultrafast", "fast", "medium")

    def __init__(self, preset: str = "ultrafast") -> None:
        import cv2

        preset_map = {
            "ultrafast": cv2.DISOpticalFlow_PRESET_ULTRAFAST,
            "fast": cv2.DISOpticalFlow_PRESET_FAST,
            "medium": cv2.DISOpticalFlow_PRESET_MEDIUM,
        }
        if preset not in preset_map:
            raise ValueError(
                f"Unknown DIS preset {preset!r}. Choose from {self.PRESETS}"
            )
        self._dis = cv2.DISOpticalFlow_create(preset_map[preset])

    def predict(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        import cv2

        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        return self._dis.calc(gray1, gray2, None)

    # preprocess / postprocess are unused for standalone adapters
    def preprocess(self, img1: np.ndarray, img2: np.ndarray) -> dict[str, np.ndarray]:
        raise NotImplementedError("DIS is standalone; use predict() directly")

    def postprocess(self, outputs: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError("DIS is standalone; use predict() directly")
