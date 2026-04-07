"""OFNet adapter for the fixed-size ONNX export."""

import numpy as np

from core.base_adapter import ModelAdapter
from core import adapter_utils as utils
from core.registry import register


@register("ofnet")
class OFNetAdapter(ModelAdapter):
    def __init__(self) -> None:
        self.original_size: tuple[int, int] | None = None
        self.MEAN = [0.5, 0.5, 0.5]
        self.STD = [0.5, 0.5, 0.5]
        self.ALIGN = 32

    def preprocess(self, img1: np.ndarray, img2: np.ndarray) -> dict[str, np.ndarray]:
        self.original_size = img1.shape[:2]

        img1 = utils.normalize_meanstd(img1, self.MEAN, self.STD)
        img2 = utils.normalize_meanstd(img2, self.MEAN, self.STD)

        img1 = utils.hwc_to_chw(img1)
        img2 = utils.hwc_to_chw(img2)

        img1 = utils.interpolate_to_divisible(img1, self.ALIGN)
        img2 = utils.interpolate_to_divisible(img2, self.ALIGN)

        img1 = utils.add_batch_dim(img1)
        img2 = utils.add_batch_dim(img2)
        
        return {
            "image1": img1,
            "image2": img2,
        }

    def postprocess(self, outputs: dict[str, np.ndarray]) -> np.ndarray:
        if self.original_size is None:
            raise RuntimeError("OFNetAdapter.postprocess() called before preprocess()")

        flow = utils.select_output(outputs, "flows")
        flow = utils.remove_batch_dim(flow)
        flow = utils.chw_to_hwc(flow)
        return utils.resize_flow(
            flow,
            self.original_size[0],
            self.original_size[1],
            scale_flow=True,
        )
