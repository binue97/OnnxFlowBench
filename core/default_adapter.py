"""
Config-driven ModelAdapter that handles ~80% of optical flow models.

All model-specific behaviour is expressed via AdapterConfig.
For models that need custom logic, subclass ModelAdapter directly.
"""

import numpy as np
import cv2

from core.base_adapter import ModelAdapter
from core.adapter_config import AdapterConfig


class DefaultAdapter(ModelAdapter):
    """
    Generic adapter driven entirely by an AdapterConfig.

    Preprocessing pipeline:
        1. Convert uint8 -> float32
        2. Normalize (none / unit / imagenet)
        3. Transpose HWC -> CHW, add batch dim -> (1, 3, H, W)
        4. Pad to multiple of padding_factor
        5. Format inputs (separate / concat)

    Postprocessing pipeline:
        1. Select the flow output tensor
        2. Remove batch dim
        3. Reformat to (H, W, 2)
        4. Apply output_scale
        5. Upsample if output_resolution != "full"
        6. Remove padding (crop to original size)
    """

    _RESOLUTION_FACTOR = {
        "full": 1,
        "quarter": 4,
        "eighth": 8,
    }

    def __init__(self, config: AdapterConfig | None = None):
        self.config = config or AdapterConfig()
        # State set by preprocess, consumed by postprocess
        self._original_h: int = 0
        self._original_w: int = 0
        self._padded_h: int = 0
        self._padded_w: int = 0

    # ── Preprocess ────────────────────────────────────────────────────────────

    def preprocess(self, img1: np.ndarray, img2: np.ndarray) -> dict[str, np.ndarray]:
        self._original_h, self._original_w = img1.shape[:2]

        img1 = self._normalize(img1.astype(np.float32))
        img2 = self._normalize(img2.astype(np.float32))

        # HWC -> CHW
        img1 = np.transpose(img1, (2, 0, 1))
        img2 = np.transpose(img2, (2, 0, 1))

        # Pad
        img1 = self._pad(img1)
        img2 = self._pad(img2)

        self._padded_h = img1.shape[1]
        self._padded_w = img1.shape[2]

        # Add batch dim -> (1, C, H, W)
        img1 = img1[np.newaxis]
        img2 = img2[np.newaxis]

        # Format inputs
        cfg = self.config
        if cfg.input_format == "concat":
            concat = np.concatenate([img1, img2], axis=1)  # (1, 6, H, W)
            return {cfg.input_names[0]: concat}
        else:  # "separate"
            return {
                cfg.input_names[0]: img1,
                cfg.input_names[1]: img2,
            }

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Apply normalization to a (H, W, 3) float32 image."""
        mode = self.config.normalization
        if mode == "none":
            return img
        elif mode == "unit":
            return img / 255.0
        elif mode == "imagenet":
            mean = np.array(self.config.imagenet_mean, dtype=np.float32).reshape(
                1, 1, 3
            )
            std = np.array(self.config.imagenet_std, dtype=np.float32).reshape(1, 1, 3)
            return (img / 255.0 - mean) / std
        else:
            raise ValueError(f"Unknown normalization mode: {mode!r}")

    def _pad(self, img: np.ndarray) -> np.ndarray:
        """Pad a (C, H, W) array so H and W are divisible by padding_factor."""
        factor = self.config.padding_factor
        if factor <= 1:
            return img
        _, h, w = img.shape
        new_h = int(np.ceil(h / factor) * factor)
        new_w = int(np.ceil(w / factor) * factor)
        pad_h = new_h - h
        pad_w = new_w - w
        if pad_h == 0 and pad_w == 0:
            return img

        if self.config.padding_mode == "zero":
            return np.pad(
                img,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=0,
            )
        elif self.config.padding_mode == "replicate":
            return np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
        else:
            raise ValueError(f"Unknown padding mode: {self.config.padding_mode!r}")

    # ── Postprocess ───────────────────────────────────────────────────────────

    def postprocess(self, outputs: dict[str, np.ndarray]) -> np.ndarray:
        flow = self._select_output(outputs)
        flow = self._remove_batch_dim(flow)
        flow = self._to_hwc(flow)
        flow = flow * self.config.output_scale
        flow = self._upsample(flow)
        flow = self._unpad(flow)
        return flow

    def _select_output(self, outputs: dict[str, np.ndarray]) -> np.ndarray:
        """Pick the flow tensor from possibly multiple outputs."""
        key = self.config.output_name
        if isinstance(key, int):
            names = list(outputs.keys())
            if key >= len(names):
                raise IndexError(
                    f"output_name index {key} out of range, model has {len(names)} outputs"
                )
            return outputs[names[key]]
        else:
            if key not in outputs:
                raise KeyError(
                    f"output_name '{key}' not found. Available: {list(outputs.keys())}"
                )
            return outputs[key]

    @staticmethod
    def _remove_batch_dim(x: np.ndarray) -> np.ndarray:
        """(1, ...) -> (...)"""
        if x.ndim >= 1 and x.shape[0] == 1:
            return x[0]
        return x

    def _to_hwc(self, flow: np.ndarray) -> np.ndarray:
        """Convert to (H, W, 2) regardless of input layout."""
        layout = self.config.output_layout
        if layout == "CHW":
            # (2, H, W) -> (H, W, 2)
            return np.transpose(flow, (1, 2, 0))
        elif layout == "HWC":
            return flow
        else:
            raise ValueError(f"Unknown output_layout: {layout!r}")

    def _upsample(self, flow: np.ndarray) -> np.ndarray:
        """Upsample flow if model outputs at sub-resolution."""
        res = self.config.output_resolution
        scale = self._RESOLUTION_FACTOR.get(res)
        if scale is None:
            raise ValueError(f"Unknown output_resolution: {res!r}")
        if scale == 1:
            return flow
        h, w = flow.shape[:2]
        new_h, new_w = h * scale, w * scale
        # Upsample each channel and scale flow magnitudes
        flow_up = cv2.resize(
            flow,
            (new_w, new_h),
            interpolation=(
                cv2.INTER_LINEAR
                if self.config.upsample_mode == "bilinear"
                else cv2.INTER_NEAREST
            ),
        )
        if self.config.scale_flow_with_upsample:
            flow_up *= scale
        return flow_up

    def _unpad(self, flow: np.ndarray) -> np.ndarray:
        """Crop back to original spatial dimensions."""
        return flow[: self._original_h, : self._original_w]
