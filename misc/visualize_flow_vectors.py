"""
Visualize optical flow vectors as arrows on an animated GIF.

Flow arrows are drawn on a background that smoothly transitions
between image 1 and image 2 in a loop.

Usage examples:

    # Homogeneous grid, 200 points
    python viz_flow.py --flow flow.flo --img1 frame1.png --img2 frame2.png \
        --mode hom --pt-num 200

    # FAST feature points, 300 points, custom output
    python viz_flow.py --flow flow.flo --img1 frame1.png --img2 frame2.png \
        --mode fast --pt-num 300 --output viz.gif
"""

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.flow_viz import make_colorwheel
from utils.frame_utils import read_gen


# ── Point selection ───────────────────────────────────────────────────────────


def _sample_homogeneous(h: int, w: int, n: int) -> np.ndarray:
    """
    Sample n points on a uniform grid over (h, w).

    Returns (n, 2) array of (y, x) coordinates.
    """
    # Compute grid spacing to get ~n points
    aspect = w / h
    rows = max(1, int(np.sqrt(n / aspect)))
    cols = max(1, int(n / rows))

    ys = np.linspace(0, h - 1, rows + 2)[1:-1]  # skip border
    xs = np.linspace(0, w - 1, cols + 2)[1:-1]
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    pts = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.float32)

    # If we have more than n, subsample randomly
    if len(pts) > n:
        idx = np.random.default_rng(42).choice(len(pts), n, replace=False)
        pts = pts[idx]
    return pts


def _sample_fast(img_gray: np.ndarray, n: int, threshold: int = 20) -> np.ndarray:
    """
    Detect FAST features on a grayscale image, return the n strongest points.

    Args:
        img_gray:  (H, W) uint8 grayscale image.
        n:         Number of points to return.
        threshold: FAST corner detection threshold (higher = stricter).

    Returns:
        (N, 2) array of (y, x) coordinates, sorted by strength (strongest first).
    """
    fast = cv2.FastFeatureDetector_create(threshold=threshold)
    fast.setNonmaxSuppression(True)
    keypoints = fast.detect(img_gray, None)

    if len(keypoints) == 0:
        raise RuntimeError("No FAST features detected. Try --mode hom or lower --fast-threshold.")

    # Sort by response strength (strongest first), keep top n
    keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)
    keypoints = keypoints[:n]

    print(f"  Keypoint strength range: [{keypoints[-1].response:.1f}, {keypoints[0].response:.1f}]")

    pts = np.array([[kp.pt[1], kp.pt[0]] for kp in keypoints], dtype=np.float32)
    return pts


def select_points(
    img1: np.ndarray, mode: str, n: int, fast_threshold: int = 20
) -> np.ndarray:
    """
    Select points to visualize.

    Args:
        img1:           (H, W, 3) uint8 RGB image.
        mode:           "hom" for homogeneous grid, "fast" for FAST features.
        n:              Number of points.
        fast_threshold: FAST detector threshold (only used when mode='fast').

    Returns:
        (N, 2) array of (y, x) coordinates (float32).
    """
    h, w = img1.shape[:2]
    if mode == "hom":
        return _sample_homogeneous(h, w, n)
    elif mode == "fast":
        gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        return _sample_fast(gray, n, threshold=fast_threshold)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'hom' or 'fast'.")


# ── Drawing ───────────────────────────────────────────────────────────────────


def _flow_vector_color(dx: float, dy: float, max_rad: float) -> tuple[int, int, int]:
    """
    Map a single flow vector (dx, dy) to an RGB color using the Middlebury colorwheel.

    Args:
        dx, dy:   Flow components.
        max_rad:  Maximum flow magnitude (for normalization).

    Returns:
        (R, G, B) tuple of ints in [0, 255].
    """
    colorwheel = make_colorwheel()  # (55, 3)
    ncols = colorwheel.shape[0]

    rad = np.sqrt(dx * dx + dy * dy)
    norm_rad = min(rad / (max_rad + 1e-5), 1.0)

    a = np.arctan2(-dy, -dx) / np.pi  # [-1, 1]
    fk = (a + 1.0) / 2.0 * (ncols - 1)
    k0 = int(np.floor(fk)) % ncols
    k1 = (k0 + 1) % ncols
    f = fk - np.floor(fk)

    color = []
    for ch in range(3):
        c0 = colorwheel[k0, ch] / 255.0
        c1 = colorwheel[k1, ch] / 255.0
        c = (1 - f) * c0 + f * c1
        c = 1 - norm_rad * (1 - c)  # whiter for small flow
        color.append(int(np.floor(255 * c)))

    return tuple(color)


def _draw_arrows(
    bg: np.ndarray,
    pts: np.ndarray,
    flow: np.ndarray,
    thickness: int = 1,
    dot_radius: int = 3,
) -> np.ndarray:
    """
    Draw flow vectors as lines with a dot at the endpoint.
    Color is determined per-vector by the Middlebury colorwheel.

    Args:
        bg:          (H, W, 3) uint8 image.
        pts:         (N, 2) array of (y, x) source points.
        flow:        (H, W, 2) float32 optical flow.
        thickness:   Line thickness.
        dot_radius:  Radius of the endpoint dot.

    Returns:
        (H, W, 3) uint8 image with flow vectors drawn.
    """
    canvas = bg.copy()
    h, w = flow.shape[:2]

    # Compute max magnitude across selected points for normalization
    magnitudes = []
    for i in range(len(pts)):
        y, x = pts[i]
        yi = int(np.clip(round(y), 0, h - 1))
        xi = int(np.clip(round(x), 0, w - 1))
        dx, dy = flow[yi, xi]
        magnitudes.append(np.sqrt(dx * dx + dy * dy))
    max_rad = max(magnitudes) if magnitudes else 1.0

    for i in range(len(pts)):
        y, x = pts[i]
        yi = int(np.clip(round(y), 0, h - 1))
        xi = int(np.clip(round(x), 0, w - 1))

        dx, dy = flow[yi, xi]
        x2 = x + dx
        y2 = y + dy

        color = _flow_vector_color(dx, dy, max_rad)

        pt1 = (int(round(x)), int(round(y)))
        pt2 = (int(round(x2)), int(round(y2)))

        cv2.line(canvas, pt1, pt2, color=color, thickness=thickness)
        cv2.circle(canvas, pt2, dot_radius, color=color, thickness=-1)

    return canvas


# ── GIF generation ────────────────────────────────────────────────────────────


def make_gif(
    img1: np.ndarray,
    img2: np.ndarray,
    flow: np.ndarray,
    pts: np.ndarray,
    output_path: str,
    n_frames: int = 30,
    hold_frames: int = 10,
    duration_ms: int = 50,
    thickness: int = 1,
):
    """
    Create an animated GIF: arrows stay fixed, background transitions
    img1 -> img2 -> img1 in a loop, with a pause on each image.

    Args:
        img1, img2:  (H, W, 3) uint8 RGB images.
        flow:        (H, W, 2) float32 optical flow.
        pts:         (N, 2) source points (y, x).
        output_path: Where to save the GIF.
        n_frames:    Number of frames per transition (1->2 or 2->1).
        hold_frames: Number of extra frames to hold on each static image.
        duration_ms: Duration per frame in milliseconds.
        thickness:   Arrow line thickness.
    """
    frames = []
    frame_img1 = _draw_arrows(img1, pts, flow, thickness=thickness)
    frame_img2 = _draw_arrows(img2, pts, flow, thickness=thickness)

    # Hold on img1
    for _ in range(hold_frames):
        frames.append(Image.fromarray(frame_img1))

    # Transition: img1 -> img2
    for i in range(n_frames):
        alpha = i / (n_frames - 1)
        blended = (
            img1.astype(np.float32) * (1 - alpha)
            + img2.astype(np.float32) * alpha
        ).astype(np.uint8)
        frame = _draw_arrows(blended, pts, flow, thickness=thickness)
        frames.append(Image.fromarray(frame))

    # Hold on img2
    for _ in range(hold_frames):
        frames.append(Image.fromarray(frame_img2))

    # Transition: img2 -> img1
    for i in range(n_frames):
        alpha = i / (n_frames - 1)
        blended = (
            img2.astype(np.float32) * (1 - alpha)
            + img1.astype(np.float32) * alpha
        ).astype(np.uint8)
        frame = _draw_arrows(blended, pts, flow, thickness=thickness)
        frames.append(Image.fromarray(frame))

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration_ms,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Visualize optical flow vectors as arrows on an animated GIF"
    )
    # Required
    parser.add_argument("--flow", required=True, help="Path to flow file (.flo, .pfm, .npy, ...)")
    parser.add_argument("--img1", required=True, help="Path to first image")
    parser.add_argument("--img2", required=True, help="Path to second image")

    # Point selection
    parser.add_argument("--mode", default="hom", choices=["fast", "hom"],
                        help="Point selection: 'fast' (FAST features) or 'hom' (uniform grid)")
    parser.add_argument("--pt-num", type=int, default=200,
                        help="Number of flow vectors to visualize")
    parser.add_argument("--fast-threshold", type=int, default=20,
                        help="FAST detector threshold (higher = only stronger corners, mode=fast only)")

    # Output
    parser.add_argument("--output", default=None,
                        help="Output GIF path (default: same dir as --flow)")

    # Visual tuning
    parser.add_argument("--n-frames", type=int, default=10,
                        help="Frames per transition (1->2 or 2->1)")
    parser.add_argument("--hold-frames", type=int, default=10,
                        help="Extra frames to hold on each static image")
    parser.add_argument("--duration", type=int, default=100,
                        help="Frame duration in milliseconds")
    parser.add_argument("--thickness", type=int, default=1,
                        help="Arrow line thickness")

    args = parser.parse_args()

    # ── Resolve output path ───────────────────────────────────────────────
    if args.output is None:
        flow_dir = os.path.dirname(os.path.abspath(args.flow))
        flow_stem = os.path.splitext(os.path.basename(args.flow))[0]
        args.output = os.path.join(flow_dir, f"{flow_stem}_viz.gif")

    # ── Load inputs ───────────────────────────────────────────────────────
    print(f"Loading flow: {args.flow}")
    flow = np.array(read_gen(args.flow), dtype=np.float32)
    assert flow.ndim == 3 and flow.shape[2] == 2, \
        f"Expected flow (H,W,2), got {flow.shape}"
    print(f"  Flow shape: {flow.shape}")

    print(f"Loading images: {args.img1}, {args.img2}")
    img1 = np.array(Image.open(args.img1).convert("RGB"))
    img2 = np.array(Image.open(args.img2).convert("RGB"))
    assert img1.shape == img2.shape, \
        f"Image shape mismatch: {img1.shape} vs {img2.shape}"
    assert img1.shape[:2] == flow.shape[:2], \
        f"Image/flow shape mismatch: img {img1.shape[:2]} vs flow {flow.shape[:2]}"
    print(f"  Image shape: {img1.shape}")

    # ── Select points ─────────────────────────────────────────────────────
    print(f"Selecting {args.pt_num} points (mode={args.mode})")
    pts = select_points(img1, args.mode, args.pt_num, fast_threshold=args.fast_threshold)
    print(f"  Selected {len(pts)} points")

    # ── Generate GIF ──────────────────────────────────────────────────────
    total = (args.n_frames + args.hold_frames) * 2
    print(f"Generating GIF ({total} frames, {args.duration}ms/frame)...")
    make_gif(
        img1, img2, flow, pts,
        output_path=args.output,
        n_frames=args.n_frames,
        hold_frames=args.hold_frames,
        duration_ms=args.duration,
        thickness=args.thickness,
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
