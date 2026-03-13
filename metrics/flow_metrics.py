"""
Optical flow evaluation metrics.

All functions follow a consistent interface:
    - pred:  (H, W, 2)  float32  - predicted flow in pixel units
    - gt:    (H, W, 2)  float32  - ground-truth flow in pixel units
    - valid: (H, W)     float32  - binary mask (1 = valid pixel, 0 = ignore)

All functions return a single float scalar.
"""

import numpy as np


def _validate_inputs(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> None:
    """Check shapes and types before computing metrics."""
    if pred.ndim != 3 or pred.shape[2] != 2:
        raise ValueError(f"pred must be (H, W, 2), got {pred.shape}")
    if gt.ndim != 3 or gt.shape[2] != 2:
        raise ValueError(f"gt must be (H, W, 2), got {gt.shape}")
    if valid.ndim != 2:
        raise ValueError(f"valid must be (H, W), got {valid.shape}")
    if pred.shape != gt.shape:
        raise ValueError(
            f"pred and gt must have the same shape, got {pred.shape} vs {gt.shape}"
        )
    if pred.shape[:2] != valid.shape:
        raise ValueError(
            f"spatial dims must match: pred {pred.shape[:2]} vs valid {valid.shape}"
        )


def _endpoint_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-pixel endpoint error. Returns (H, W) array."""
    return np.sqrt(np.sum((pred - gt) ** 2, axis=-1))


# ═══════════════════════════════════════════════════════════════════════════════
# Core metrics
# ═══════════════════════════════════════════════════════════════════════════════


def epe(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    """
    End-Point Error - mean L2 distance over valid pixels.

    This is the primary metric for Sintel, FlyingChairs, FlyingThings, Spring.

    Returns:
        Mean EPE (float). Returns 0.0 if no valid pixels.
    """
    _validate_inputs(pred, gt, valid)
    mask = valid > 0.5
    if not mask.any():
        return 0.0
    err = _endpoint_error(pred, gt)
    return float(np.mean(err[mask]))


def fl_all(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    epe_thresh: float = 3.0,
    rel_thresh: float = 0.05,
) -> float:
    """
    Fl-all - fraction of outlier pixels (KITTI metric).

    A pixel is an outlier if:
        EPE > epe_thresh  AND  EPE > rel_thresh * ||gt||

    Returns:
        Outlier percentage in [0, 100] (float). Returns 0.0 if no valid pixels.
    """
    _validate_inputs(pred, gt, valid)
    mask = valid > 0.5
    if not mask.any():
        return 0.0
    err = _endpoint_error(pred, gt)
    gt_mag = np.sqrt(np.sum(gt**2, axis=-1))
    outlier = (err > epe_thresh) & (err > rel_thresh * gt_mag)
    return float(np.mean(outlier[mask])) * 100.0


def n_pixel(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    n: float = 1.0,
) -> float:
    """
    N-pixel error - fraction of pixels with EPE > n.

    Common thresholds: 1px (Spring), 3px, 5px.

    Returns:
        Error percentage in [0, 100] (float). Returns 0.0 if no valid pixels.
    """
    _validate_inputs(pred, gt, valid)
    mask = valid > 0.5
    if not mask.any():
        return 0.0
    err = _endpoint_error(pred, gt)
    return float(np.mean(err[mask] > n)) * 100.0


# ═══════════════════════════════════════════════════════════════════════════════
# Per-dataset metric dispatch
# ═══════════════════════════════════════════════════════════════════════════════

# Which metrics to compute for each dataset family
# (TODO): This should be moved to a config file
DATASET_METRICS = {
    "sintel": ["epe", "fl_all", "1px", "3px", "5px"],
    "chairs": ["epe", "fl_all", "1px", "3px", "5px"],
    "things": ["epe", "fl_all", "1px", "3px", "5px"],
    "kitti": ["epe", "fl_all", "1px", "3px", "5px"],
    "spring": ["epe", "fl_all", "1px", "3px", "5px"],
    "hd1k": ["epe", "fl_all", "1px", "3px", "5px"],
    "tartanair": ["epe", "fl_all", "1px", "3px", "5px"],
}


def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    dataset: str | None = None,
) -> dict[str, float]:
    """
    Compute the standard metrics for a given dataset.

    If dataset is None, all metrics are computed.

    Args:
        pred:    (H, W, 2) predicted flow
        gt:      (H, W, 2) ground-truth flow
        valid:   (H, W)    binary validity mask
        dataset: dataset name (e.g. "sintel", "kitti") or None for all

    Returns:
        Dict of metric_name -> value.
    """
    _validate_inputs(pred, gt, valid)

    if dataset is not None:
        dataset = dataset.lower()

    # Determine which metrics to compute
    if dataset is not None and dataset in DATASET_METRICS:
        metric_names = DATASET_METRICS[dataset]
    else:
        metric_names = ["epe", "fl_all", "1px", "3px", "5px"]

    results = {}
    for name in metric_names:
        if name == "epe":
            results["epe"] = epe(pred, gt, valid)
        elif name == "fl_all":
            results["fl_all"] = fl_all(pred, gt, valid)
        elif name.endswith("px"):
            n = float(name.replace("px", ""))
            results[name] = n_pixel(pred, gt, valid, n=n)

    return results
