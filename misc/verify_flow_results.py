"""
Compare two optical flow fields and print a detailed analysis.

Supports .flo and .npy formats.

Usage:
    python misc/verify_flow_results.py flow_a.npy flow_b.npy
    python misc/verify_flow_results.py flow_a.flo flow_b.flo
    python misc/verify_flow_results.py flow_a.flo flow_b.npy
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.frame_utils import readFlow


# ═══════════════════════════════════════════════════════════════════════════════
# I/O
# ═══════════════════════════════════════════════════════════════════════════════


def load_flow(path: str) -> np.ndarray:
    """Load a flow field from .flo or .npy file. Returns (H, W, 2) float32."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".flo":
        flow = readFlow(path)
    elif ext == ".npy":
        flow = np.load(path)
    else:
        raise ValueError(f"Unsupported format: {ext!r}. Use .flo or .npy")

    if flow is None:
        raise RuntimeError(f"Failed to load flow from {path}")

    flow = flow.astype(np.float32)

    # Handle (2, H, W) -> (H, W, 2)
    if flow.ndim == 3 and flow.shape[0] == 2:
        flow = np.transpose(flow, (1, 2, 0))

    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"Expected (H, W, 2) flow, got shape {flow.shape}")

    return flow


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════════════


def compute_magnitude(flow: np.ndarray) -> np.ndarray:
    """Per-pixel flow magnitude. Returns (H, W)."""
    return np.sqrt(np.sum(flow ** 2, axis=-1))


def percentile_str(arr: np.ndarray, percentiles: list[int]) -> str:
    """Format percentile values as a compact string."""
    vals = np.percentile(arr, percentiles)
    parts = [f"P{p}={v:.4f}" for p, v in zip(percentiles, vals)]
    return "  ".join(parts)


def analyze_single(flow: np.ndarray, label: str) -> dict:
    """Print stats for a single flow field. Returns summary dict."""
    mag = compute_magnitude(flow)
    u, v = flow[..., 0], flow[..., 1]

    stats = {
        "shape": flow.shape,
        "dtype": flow.dtype,
        "mag_mean": float(np.mean(mag)),
        "mag_std": float(np.std(mag)),
        "mag_min": float(np.min(mag)),
        "mag_max": float(np.max(mag)),
        "u_mean": float(np.mean(u)),
        "v_mean": float(np.mean(v)),
        "nan_count": int(np.isnan(flow).sum()),
        "inf_count": int(np.isinf(flow).sum()),
    }

    print(f"  ┌── {label}")
    print(f"  │ Shape: {stats['shape']}   dtype: {stats['dtype']}")
    print(f"  │ NaN: {stats['nan_count']}   Inf: {stats['inf_count']}")
    print(f"  │")
    print(f"  │ Magnitude")
    print(f"  │   mean: {stats['mag_mean']:.4f}   std: {stats['mag_std']:.4f}")
    print(f"  │   min:  {stats['mag_min']:.4f}   max: {stats['mag_max']:.4f}")
    print(f"  │   {percentile_str(mag, [25, 50, 75, 95, 99])}")
    print(f"  │")
    print(f"  │ Components")
    print(f"  │   u (horizontal):  mean={np.mean(u):.4f}  std={np.std(u):.4f}  range=[{np.min(u):.4f}, {np.max(u):.4f}]")
    print(f"  │   v (vertical):    mean={np.mean(v):.4f}  std={np.std(v):.4f}  range=[{np.min(v):.4f}, {np.max(v):.4f}]")
    print(f"  └──")

    return stats


def compare_flows(flow_a: np.ndarray, flow_b: np.ndarray) -> None:
    """Print comparison metrics between two flow fields."""
    diff = flow_a - flow_b
    epe = compute_magnitude(diff)
    mag_a = compute_magnitude(flow_a)
    mag_b = compute_magnitude(flow_b)

    # Relative error (avoid division by zero)
    max_mag = np.maximum(mag_a, mag_b)
    safe_denom = np.where(max_mag > 1e-6, max_mag, 1.0)
    relative_err = epe / safe_denom

    # Angular error
    u_a, v_a = flow_a[..., 0], flow_a[..., 1]
    u_b, v_b = flow_b[..., 0], flow_b[..., 1]
    # Use (u, v, 1) representation for angular error
    dot = u_a * u_b + v_a * v_b + 1.0
    norm_a = np.sqrt(u_a ** 2 + v_a ** 2 + 1.0)
    norm_b = np.sqrt(u_b ** 2 + v_b ** 2 + 1.0)
    cos_angle = np.clip(dot / (norm_a * norm_b), -1.0, 1.0)
    angular_err = np.degrees(np.arccos(cos_angle))

    # Outlier rates
    n_pixels = flow_a.shape[0] * flow_a.shape[1]
    outlier_3px = np.sum(epe > 3.0)
    outlier_1px = np.sum(epe > 1.0)
    outlier_05px = np.sum(epe > 0.5)

    # Correlation
    corr_u = np.corrcoef(flow_a[..., 0].ravel(), flow_b[..., 0].ravel())[0, 1]
    corr_v = np.corrcoef(flow_a[..., 1].ravel(), flow_b[..., 1].ravel())[0, 1]

    W = 60

    print()
    print(f"  ╔{'═' * W}╗")
    print(f"  ║{'COMPARISON':^{W}}║")
    print(f"  ╠{'═' * W}╣")

    # EPE
    print(f"  ║{'':2}{'End-Point Error (EPE)':<{W - 2}}║")
    print(f"  ║{'':4}{'mean:':<12}{np.mean(epe):>12.4f}{'':>32}║")
    print(f"  ║{'':4}{'std:':<12}{np.std(epe):>12.4f}{'':>32}║")
    print(f"  ║{'':4}{'median:':<12}{np.median(epe):>12.4f}{'':>32}║")
    print(f"  ║{'':4}{'max:':<12}{np.max(epe):>12.4f}{'':>32}║")
    print(f"  ║{'':4}{percentile_str(epe, [90, 95, 99]):<{W - 4}}║")
    print(f"  ╟{'─' * W}╢")

    # Relative Error
    print(f"  ║{'':2}{'Relative Error':<{W - 2}}║")
    print(f"  ║{'':4}{'mean:':<12}{np.mean(relative_err):>12.4f}{'':>32}║")
    print(f"  ║{'':4}{'median:':<12}{np.median(relative_err):>12.4f}{'':>32}║")
    print(f"  ╟{'─' * W}╢")

    # Angular Error
    print(f"  ║{'':2}{'Angular Error (degrees)':<{W - 2}}║")
    print(f"  ║{'':4}{'mean:':<12}{np.mean(angular_err):>12.4f}{'':>32}║")
    print(f"  ║{'':4}{'median:':<12}{np.median(angular_err):>12.4f}{'':>32}║")
    print(f"  ║{'':4}{'max:':<12}{np.max(angular_err):>12.4f}{'':>32}║")
    print(f"  ╟{'─' * W}╢")

    # Outliers
    print(f"  ║{'':2}{'Outlier Rates':<{W - 2}}║")
    print(f"  ║{'':4}{'EPE > 0.5px:':<20}{outlier_05px:>8} / {n_pixels:<8} ({100 * outlier_05px / n_pixels:>6.2f}%){'':>5}║")
    print(f"  ║{'':4}{'EPE > 1.0px:':<20}{outlier_1px:>8} / {n_pixels:<8} ({100 * outlier_1px / n_pixels:>6.2f}%){'':>5}║")
    print(f"  ║{'':4}{'EPE > 3.0px:':<20}{outlier_3px:>8} / {n_pixels:<8} ({100 * outlier_3px / n_pixels:>6.2f}%){'':>5}║")
    print(f"  ╟{'─' * W}╢")

    # Correlation
    print(f"  ║{'':2}{'Correlation (Pearson)':<{W - 2}}║")
    print(f"  ║{'':4}{'u channel:':<12}{corr_u:>12.6f}{'':>32}║")
    print(f"  ║{'':4}{'v channel:':<12}{corr_v:>12.6f}{'':>32}║")
    print(f"  ╟{'─' * W}╢")

    # Verdict
    mean_epe = np.mean(epe)
    if mean_epe < 0.01:
        verdict = "✓ IDENTICAL (numerically equivalent)"
    elif mean_epe < 0.1:
        verdict = "✓ VERY CLOSE (minor numerical differences)"
    elif mean_epe < 1.0:
        verdict = "~ SIMILAR (small but noticeable differences)"
    elif mean_epe < 5.0:
        verdict = "✗ DIFFERENT (significant differences)"
    else:
        verdict = "✗ VERY DIFFERENT (large deviations)"

    print(f"  ║{'':2}{'Verdict':<{W - 2}}║")
    print(f"  ║{'':4}{verdict:<{W - 4}}║")
    print(f"  ╚{'═' * W}╝")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Compare two optical flow fields (.flo or .npy).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python misc/verify_flow_results.py flow_a.npy flow_b.npy\n"
            "  python misc/verify_flow_results.py flow_a.flo flow_b.npy\n"
        ),
    )
    parser.add_argument("flow_a", help="Path to first flow file (.flo or .npy)")
    parser.add_argument("flow_b", help="Path to second flow file (.flo or .npy)")
    args = parser.parse_args()

    # Load
    print()
    print(f"  Loading: {args.flow_a}")
    flow_a = load_flow(args.flow_a)
    print(f"  Loading: {args.flow_b}")
    flow_b = load_flow(args.flow_b)
    print()

    # Individual stats
    name_a = os.path.basename(args.flow_a)
    name_b = os.path.basename(args.flow_b)

    analyze_single(flow_a, f"A: {name_a}")
    print()
    analyze_single(flow_b, f"B: {name_b}")

    # Shape check
    if flow_a.shape != flow_b.shape:
        print()
        print(f"  ✗ Shape mismatch: {flow_a.shape} vs {flow_b.shape}")
        print(f"    Cannot compute comparison metrics.")
        sys.exit(1)

    # Compare
    compare_flows(flow_a, flow_b)


if __name__ == "__main__":
    main()
