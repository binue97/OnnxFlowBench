"""
Config-driven ModelAdapter that handles general optical flow models.

All model-specific behaviour is configured via AdapterConfig.
For models that need custom pre/post processing, create new Adapter that subclass ModelAdapter directly.
"""

import numpy as np
import cv2

from core.base_adapter import ModelAdapter
from core.adapter_config import AdapterConfig


class DefaultAdapter(ModelAdapter):
    """Generic adapter driven entirely by an AdapterConfig."""

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

    def preprocess(self, img1: np.ndarray, img2: np.ndarray) -> dict[str, np.ndarray]:
        """Preprocess two raw images into an ONNX input feed according to the config."""
        self._original_h, self._original_w = img1.shape[:2]

        img1 = self._normalize(img1.astype(np.float32))
        img2 = self._normalize(img2.astype(np.float32))

        # Color channel reorder (input is RGB from loader)
        if self.config.input_color_order == "bgr":
            img1 = img1[:, :, ::-1].copy()
            img2 = img2[:, :, ::-1].copy()

        # HWC -> CHW
        img1 = np.transpose(img1, (2, 0, 1))
        img2 = np.transpose(img2, (2, 0, 1))

        # Resize input images (pad or interpolation)
        img1 = self._resize_input(img1)
        img2 = self._resize_input(img2)

        # Add batch dim -> (1, C, H, W)
        img1 = img1[np.newaxis]
        img2 = img2[np.newaxis]

        # Format inputs
        cfg = self.config
        if cfg.input_format == "concat":
            # (1, 6, H, W)
            concat = np.concatenate([img1, img2], axis=1)
            return {cfg.input_names[0]: concat}
        elif cfg.input_format == "stacked":
            # (1, 2, 3, H, W)
            stacked = np.stack([img1, img2], axis=1)
            return {cfg.input_names[0]: stacked}
        else:
            # (1, 3, H, W) x 2
            return {
                cfg.input_names[0]: img1,
                cfg.input_names[1]: img2,
            }

    def postprocess(self, outputs: dict[str, np.ndarray]) -> np.ndarray:
        """Postprocess raw ONNX outputs into standard flow format according to the config."""
        flow = self._select_output(outputs)
        flow = self._remove_batch_dim(flow)
        flow = self._to_hwc(flow)
        flow = flow * self.config.output_scale
        flow = self._resize_output(flow)
        return flow

    # ── Utilities ─────────────────────────────────────────────────────────────────

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Apply normalization to a (H, W, 3) float32 image."""
        mode = self.config.normalization
        if mode == "none":
            return img
        elif mode == "unit":
            return img / 255.0
        elif mode == "meanstd":
            mean = np.array(self.config.normalize_mean, dtype=np.float32).reshape(
                1, 1, 3
            )
            std = np.array(self.config.normalize_std, dtype=np.float32).reshape(1, 1, 3)
            return (img / 255.0 - mean) / std
        else:
            raise ValueError(f"Unknown normalization mode: {mode!r}")

    def _resize_input(self, img: np.ndarray) -> np.ndarray:
        """Resize a (C, H, W) array so H and W are divisible by resizing_factor."""
        if self.config.resize_mode == "interpolation":
            return self._interpolate_input(img)
        elif self.config.resize_mode == "pad":
            return self._pad(img)
        else:
            raise ValueError(f"Unknown resize_mode: {self.config.resize_mode!r}")

    def _pad(self, img: np.ndarray) -> np.ndarray:
        """Pad a (C, H, W) array so H and W are divisible by resizing_factor."""
        factor = self.config.resizing_factor
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

    def _interpolate_input(self, img: np.ndarray) -> np.ndarray:
        """Resize a (C, H, W) array via interpolation so dims are divisible by resizing_factor."""
        factor = self.config.resizing_factor
        if factor <= 1:
            return img
        _, h, w = img.shape
        new_h = int(np.ceil(h / factor) * factor)
        new_w = int(np.ceil(w / factor) * factor)
        if new_h == h and new_w == w:
            return img
        # Transpose to (H, W, C) for cv2.resize, then back to (C, H, W)
        hwc = np.transpose(img, (1, 2, 0))
        hwc_resized = cv2.resize(hwc, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return np.transpose(hwc_resized, (2, 0, 1))

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

    def _resize_output(self, flow: np.ndarray) -> np.ndarray:
        """Restore flow to original spatial dimensions (upsample + undo resize/pad)."""
        h, w = flow.shape[:2]
        target_h, target_w = self._original_h, self._original_w

        res = self.config.output_resolution
        scale = self._RESOLUTION_FACTOR.get(res)
        if scale is None:
            raise ValueError(f"Unknown output_resolution: {res!r}")

        if self.config.resize_mode == "interpolation":
            # Single resize: upsample sub-resolution and undo input interpolation
            # in one step to avoid an extra interpolation pass.
            flow = self._resize_flow(
                flow,
                target_h,
                target_w,
                scale_flow=self.config.scale_flow_with_upsample,
            )
        elif self.config.resize_mode == "pad":
            # Upsample first (resize), then crop
            if scale > 1:
                flow = self._resize_flow(
                    flow,
                    h * scale,
                    w * scale,
                    scale_flow=self.config.scale_flow_with_upsample,
                )
            # Crop to original size
            flow = flow[:target_h, :target_w]
        else:
            raise ValueError(f"Unknown resize_mode: {self.config.resize_mode!r}")

        return flow

    @staticmethod
    def _remove_batch_dim(x: np.ndarray) -> np.ndarray:
        """Squeeze all leading size-1 dims, e.g. (1,1,2,H,W) -> (2,H,W)."""
        while x.ndim > 3 and x.shape[0] == 1:
            x = x[0]
        return x

    @staticmethod
    def _resize_flow(
        flow: np.ndarray,
        target_h: int,
        target_w: int,
        scale_flow: bool = True,
    ) -> np.ndarray:
        """Resize (H, W, 2) flow to target size, optionally scaling values proportionally."""
        h, w = flow.shape[:2]
        if h == target_h and w == target_w:
            return flow
        flow_resized = cv2.resize(
            flow,
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR,
        )
        if scale_flow:
            flow_resized[..., 0] *= target_w / w  # horizontal
            flow_resized[..., 1] *= target_h / h  # vertical
        return flow_resized
