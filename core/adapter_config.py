"""
Configuration dataclass for the DefaultAdapter.
"""

from dataclasses import dataclass, field


@dataclass
class AdapterConfig:
    """
    Declarative configuration for DefaultAdapter.

    Covers general optical flow models without writing custom code for every model.
    For models that need special logic (tiling, multi-pass, etc.),
    create a new adapter class that subclasses ModelAdapter.
    """

    # ── Input ─────────────────────────────────────────────────────────────────

    input_names: list[str] = field(default_factory=lambda: ["image1", "image2"])
    """ONNX input tensor names, in order [img1, img2] or [concatenated]."""

    input_format: str = "separate"
    """
    How images are fed to the model:
        "separate" -> two tensors
        "concat"   -> one 6-channel tensor (B, 6, H, W)
        "stacked"  -> one tensor (B, 2, 3, H, W)
    """

    normalization: str = "unit"
    """
    Pixel normalization before inference:
        "none"     -> keep [0, 255]
        "unit"     -> scale to [0, 1]
        "imagenet" -> ImageNet mean/std
    """

    imagenet_mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    imagenet_std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    input_color_order: str = "rgb"
    """
    Expected color channel order of the model input:
        "rgb" -> no conversion needed (default, most models)
        "bgr" -> flip RGB to BGR before feeding to model
    """

    resizing_factor: int = 8
    """Resize spatial dims to be divisible by this factor (8, 16, 32, ...)."""

    padding_mode: str = "replicate"
    """Padding mode: "zero" or "replicate". Only used when resize_mode="pad"."""

    resize_mode: str = "interpolation"
    """
    How to make spatial dims divisible by resizing_factor:
        "interpolation" -> bilinear resize to nearest multiple, resize-back in postprocess
        "pad"           -> pad with padding_mode, crop in postprocess
    """

    interpolation_align_corners: bool = True
    """align_corners for interpolation resize."""

    # ── Output ────────────────────────────────────────────────────────────────

    output_name: str | int = 0
    """
    Which ONNX output tensor contains the flow.
    String -> match by name.  Int -> index into output list.
    """

    output_layout: str = "CHW"
    """
    Layout of the flow output tensor (after removing batch dim):
        "CHW" -> (2, H, W)
        "HWC" -> (H, W, 2)
    """

    output_scale: float = 1.0
    """
    Multiplier applied to raw output flow.
    """

    output_resolution: str = "full"
    """
    Resolution of the model's flow output relative to input:
        "full"    -> same as input
        "quarter" -> 1/4 resolution
        "eighth"  -> 1/8 resolution
    """

    upsample_mode: str = "bilinear"
    """Interpolation mode when upsampling sub-resolution output: "bilinear" or "nearest"."""

    scale_flow_with_upsample: bool = True
    """
    Whether to scale flow magnitudes by the upsample factor.
        True  -> flow values are in sub-resolution pixel units
        False -> flow values are already in full-resolution pixel units
    """
