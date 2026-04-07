"""
FlowNetS adapter.

Reference: https://github.com/ClementPinard/FlowNetPytorch
"""

import numpy as np

from core.base_adapter import ModelAdapter
from core import adapter_utils as utils


class FlowNetSAdapter(ModelAdapter):
    MEAN = [0.411, 0.432, 0.45]
    STD = [1.0, 1.0, 1.0]
    DIVFLOW = 20.0

    def preprocess(self, img1: np.ndarray, img2: np.ndarray) -> dict[str, np.ndarray]:
        img1 = utils.normalize_meanstd(img1, self.MEAN, self.STD)
        img2 = utils.normalize_meanstd(img2, self.MEAN, self.STD)

        img1 = utils.hwc_to_chw(img1)
        img2 = utils.hwc_to_chw(img2)

        img1 = utils.add_batch_dim(img1)
        img2 = utils.add_batch_dim(img2)

        concat = np.concatenate([img1, img2], axis=1)  # (1, 6, H, W)
        return {"input": concat}

    def postprocess(self, outputs: dict[str, np.ndarray]) -> np.ndarray:
        flow = utils.select_output(outputs, "output")
        flow = utils.remove_batch_dim(flow)
        flow = self.DIVFLOW * flow
        return flow
