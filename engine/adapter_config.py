"""
Configuration dataclass for the DefaultAdapter.

Covers the common axes of variation across optical flow models:
    - Input normalization, layout, and naming
    - Padding strategy
    - Output selection, layout, scale, and resolution
"""

from dataclasses import dataclass, field


@dataclass
class AdapterConfig:
    """
    Declarative configuration for DefaultAdapter.

    Covers ~80% of optical flow models without writing custom code.
    For models that need special logic (tiling, multi-pass, etc.),
    subclass ModelAdapter directly instead.
    """

    # ── Input ─────────────────────────────────────────────────────────────────

    input_names: list[str] = field(default_factory=lambda: ["image1", "image2"])
    """ONNX input tensor names, in order [img1, img2] or [concatenated]."""

    input_format: str = "separate"
    """
    How images are fed to the model:
        "separate" -> two tensors (most models: RAFT, GMA, FlowFormer, ...)
        "concat"   -> one 6-channel tensor (B, 6, H, W)  (PWC-Net, LiteFlowNet, ...)
    """

    normalization: str = "none"
    """
    Pixel normalization before inference:
        "none"     -> keep [0, 255]      (RAFT, GMA, ...)
        "unit"     -> scale to [0, 1]    (PWC-Net, ...)
        "imagenet" -> ImageNet mean/std  (FlowFormer, ...)
    """

    imagenet_mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    imagenet_std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    padding_factor: int = 8
    """Pad spatial dims to be divisible by this factor (8, 16, 32, ...)."""

    padding_mode: str = "replicate"
    """Padding mode: "zero" or "replicate"."""

    # ── Output ────────────────────────────────────────────────────────────────

    output_name: str | int = 0
    """
    Which ONNX output tensor contains the flow.
    String -> match by name.  Int -> index into output list.
    """

    output_layout: str = "CHW"
    """
    Layout of the flow output tensor (after removing batch dim):
        "CHW" -> (2, H, W)  - most models
        "HWC" -> (H, W, 2)
    """

    output_scale: float = 1.0
    """
    Multiplier applied to raw flow values.
    Most models output pixel-unit flow (scale=1.0).
    """

    output_resolution: str = "full"
    """
    Resolution of the model's flow output relative to input:
        "full"    -> same as input (most models)
        "quarter" -> 1/4 resolution (PWC-Net, ...)
        "eighth"  -> 1/8 resolution
    """

    upsample_mode: str = "bilinear"
    """Interpolation mode when upsampling sub-resolution output: "bilinear" or "nearest"."""
