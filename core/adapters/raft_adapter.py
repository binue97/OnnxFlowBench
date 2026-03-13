"""
FlowNetS adapter.

Reference: https://github.com/hmorimitsu/ptlflow
"""

import numpy as np

from core.base_adapter import ModelAdapter
from core import adapter_utils as utils


class RaftAdapter(ModelAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padder = utils.Padder(factor=8, mode="replicate", two_side_pad=True)

    def preprocess(self, img1: np.ndarray, img2: np.ndarray) -> dict[str, np.ndarray]:
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)

        img1 = utils.hwc_to_chw(img1)
        img2 = utils.hwc_to_chw(img2)

        self.padder.reset()
        img1 = self.padder.pad(img1)
        img2 = self.padder.pad(img2)

        img1 = utils.add_batch_dim(img1)
        img2 = utils.add_batch_dim(img2)

        return {"image1": img1, "image2": img2}

    def postprocess(self, outputs: dict[str, np.ndarray]) -> np.ndarray:
        flow = utils.select_output(outputs, "flow_up")
        flow = utils.remove_batch_dim(flow)
        flow = self.padder.unpad(flow)
        flow = utils.chw_to_hwc(flow)
        return flow
