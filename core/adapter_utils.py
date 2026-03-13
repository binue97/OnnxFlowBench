"""
Reusable pre/post-processing utilities for optical flow adapters.

These functions are building blocks that any ModelAdapter subclass
can compose to avoid reimplementing common logic.
"""

import numpy as np
import cv2


# ── Padder ─────────────────────────────────────────────────────────────────────
class Padder:
    """Pad images so spatial dims are divisible by *factor*, then unpad outputs.

    Remembers the padding applied during :meth:`pad` so that :meth:`unpad`
    can restore the original spatial dimensions without any bookkeeping
    on the caller's side.
    """

    def __init__(
        self,
        factor: int = 8,
        mode: str = "replicate",
        two_side_pad: bool = True,
    ) -> None:
        self.factor = factor
        self.mode = mode
        self.two_side_pad = two_side_pad
        # (top, bottom, left, right) — set by the first call to pad()
        self._pad: tuple[int, int, int, int] | None = None

    def pad(self, img: np.ndarray) -> np.ndarray:
        """Pad a (C, H, W) array so H and W are divisible by *factor*."""
        _, h, w = img.shape

        if self._pad is None:
            self._pad = self._compute_pad(h, w)

        top, bottom, left, right = self._pad
        if top == 0 and bottom == 0 and left == 0 and right == 0:
            return img

        if self.mode == "zero":
            return np.pad(
                img,
                ((0, 0), (top, bottom), (left, right)),
                mode="constant",
                constant_values=0,
            )
        elif self.mode == "replicate":
            return np.pad(
                img,
                ((0, 0), (top, bottom), (left, right)),
                mode="edge",
            )
        else:
            raise ValueError(f"Unknown padding mode: {self.mode!r}")

    def unpad(self, x: np.ndarray) -> np.ndarray:
        """Remove the padding from the last two spatial dims of *x*."""
        if self._pad is None:
            raise RuntimeError("Padder.unpad() called before pad()")

        top, bottom, left, right = self._pad
        if top == 0 and bottom == 0 and left == 0 and right == 0:
            return x
        h = x.shape[-2]
        w = x.shape[-1]
        return x[..., top : h - bottom, left : w - right]

    def reset(self) -> None:
        """Clear stored padding so the next :meth:`pad` recomputes it."""
        self._pad = None

    def _compute_pad(
        self, h: int, w: int
    ) -> tuple[int, int, int, int]:
        """Return (top, bottom, left, right) padding."""
        if self.factor <= 1:
            return (0, 0, 0, 0)

        new_h = int(np.ceil(h / self.factor) * self.factor)
        new_w = int(np.ceil(w / self.factor) * self.factor)
        pad_h = new_h - h
        pad_w = new_w - w

        if self.two_side_pad:
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
        else:
            top, bottom = 0, pad_h
            left, right = 0, pad_w

        return (top, bottom, left, right)


# ── Normalization ─────────────────────────────────────────────────────────────


def normalize_unit(img: np.ndarray) -> np.ndarray:
    """Scale pixel values from [0, 255] to [0, 1].

    Args:
        img: (H, W, 3) uint8 or float32.

    Returns:
        (H, W, 3) float32 in [0, 1].
    """
    return img.astype(np.float32) / 255.0


def normalize_meanstd(
    img: np.ndarray,
    mean: list[float] = (0.485, 0.456, 0.406),
    std: list[float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Apply (img/255 − mean) / std normalization.

    Args:
        img:  (H, W, 3) uint8 or float32.
        mean: Per-channel mean (length 3).
        std:  Per-channel std  (length 3).

    Returns:
        (H, W, 3) float32.
    """
    img = img.astype(np.float32) / 255.0
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    return (img - mean) / std


# ── Color channel reorder ────────────────────────────────────────────────────


def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    """Reverse channel order: RGB ↔ BGR. Works on (H, W, 3)."""
    return img[:, :, ::-1].copy()


# ── Layout conversion ────────────────────────────────────────────────────────


def hwc_to_chw(img: np.ndarray) -> np.ndarray:
    """(H, W, C) → (C, H, W)."""
    return np.transpose(img, (2, 0, 1))


def chw_to_hwc(tensor: np.ndarray) -> np.ndarray:
    """(C, H, W) → (H, W, C)."""
    return np.transpose(tensor, (1, 2, 0))


# ── Batch dimension ──────────────────────────────────────────────────────────


def add_batch_dim(img: np.ndarray) -> np.ndarray:
    """Add a leading batch dimension: (C, H, W) → (1, C, H, W)."""
    return img[np.newaxis]


def remove_batch_dim(x: np.ndarray) -> np.ndarray:
    """Squeeze all leading size-1 dims, e.g. (1, 1, 2, H, W) → (2, H, W)."""
    while x.ndim > 3 and x.shape[0] == 1:
        x = x[0]
    return x


# ── Spatial resizing ─────────────────────────────────────────────────────────


def interpolate_to_divisible(img: np.ndarray, factor: int) -> np.ndarray:
    """Resize a (C, H, W) array via bilinear interpolation so H and W
    are divisible by *factor*.

    Args:
        img:    (C, H, W) tensor.
        factor: Divisibility factor.

    Returns:
        Resized (C, H', W') array.
    """
    if factor <= 1:
        return img
    _, h, w = img.shape
    new_h = int(np.ceil(h / factor) * factor)
    new_w = int(np.ceil(w / factor) * factor)
    if new_h == h and new_w == w:
        return img
    hwc = np.transpose(img, (1, 2, 0))
    hwc_resized = cv2.resize(hwc, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return np.transpose(hwc_resized, (2, 0, 1))


# ── Output selection ─────────────────────────────────────────────────────────


def select_output(outputs: dict[str, np.ndarray], key: str | int = 0) -> np.ndarray:
    """Pick a tensor from the ONNX output dict by name (``str``) or index (``int``).

    Args:
        outputs: Dict from OnnxEngine.__call__.
        key:     Output tensor name or positional index.

    Raises:
        IndexError: If index is out of range.
        KeyError:   If name is not found.
    """
    if isinstance(key, int):
        names = list(outputs.keys())
        if key >= len(names):
            raise IndexError(
                f"Output index {key} out of range, model has {len(names)} outputs"
            )
        return outputs[names[key]]
    else:
        if key not in outputs:
            raise KeyError(
                f"Output '{key}' not found. Available: {list(outputs.keys())}"
            )
        return outputs[key]


# ── Flow resizing ────────────────────────────────────────────────────────────


def resize_flow(
    flow: np.ndarray,
    target_h: int,
    target_w: int,
    scale_flow: bool = True,
) -> np.ndarray:
    """Resize (H, W, 2) flow to target size.

    Args:
        flow:       (H, W, 2) float32 flow field.
        target_h:   Target height.
        target_w:   Target width.
        scale_flow: If True, scale flow magnitudes proportionally
                    to the resize ratio (assumes flow values are in
                    source-resolution pixel units).

    Returns:
        Resized (target_h, target_w, 2) float32 flow.
    """
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
