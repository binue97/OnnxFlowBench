"""
Single-pair inference and verification.

Runs one image pair through the FlowModel pipeline and optionally
compares the output against a reference for bit-exactness.

Usage examples:

    # Infer from two image paths, save visualization
    python infer.py --model raft.onnx --adapter raft --img1 frame1.png --img2 frame2.png --output results/

    # Save predicted flow as .npy
    python infer.py --model raft.onnx --adapter raft \
        --img1 frame1.png --img2 frame2.png --output results/ --save-npy
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.flow_model import FlowModel
from core.registry import list_adapters
from utils.flow_viz import flow_to_image
from utils.frame_utils import writeFlow


def load_image(path: str) -> np.ndarray:
    """Load an image as (H, W, 3) uint8 RGB."""
    img = Image.open(path)
    mode = img.mode  # e.g. "L", "RGB", "RGBA", "I", "P", ...
    w, h = img.size
    print(f"  Image path: {path}: {w}x{h}, mode={mode}")

    img = np.array(img.convert("RGB"))
    print(f"  Converted from {mode} to RGB, shape: {img.shape}, dtype: {img.dtype}\n")
    return img


def save_results(
    output_dir: str,
    flow_pred: np.ndarray,
    name: str = "flow_pred",
    png: bool = False,
    flo: bool = False,
    npy: bool = False,
):
    """Save predicted flow in the requested formats."""
    os.makedirs(output_dir, exist_ok=True)

    if png:
        path = os.path.join(output_dir, f"{name}.png")
        flow_viz = flow_to_image(flow_pred)
        Image.fromarray(flow_viz).save(path)
        print(f"  Saved: {path}")

    if flo:
        path = os.path.join(output_dir, f"{name}.flo")
        writeFlow(path, flow_pred)
        print(f"  Saved: {path}")

    if npy:
        path = os.path.join(output_dir, f"{name}.npy")
        np.save(path, flow_pred)
        print(f"  Saved: {path}")


def main():
    # ── Argparse ───────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Single-pair optical flow inference and verification"
    )
    # Model
    parser.add_argument("--model", required=True, help="Path to .onnx model")
    parser.add_argument(
        "--adapter",
        default="raft",
        help=f"Adapter name ({', '.join(list_adapters())})",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    # Input
    parser.add_argument("--img1", required=True, help="Path to first image")
    parser.add_argument("--img2", required=True, help="Path to second image")
    # Output
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--png", action="store_true", help="Save color visualization (.png)")
    parser.add_argument("--flo", action="store_true", help="Save Middlebury .flo format")
    parser.add_argument("--npy", action="store_true", help="Save raw numpy .npy")
    args = parser.parse_args()

    # ── Load images ───────────────────────────────────────────────────────────
    print(f"Loading images: {args.img1}, {args.img2}")
    img1 = load_image(args.img1)
    img2 = load_image(args.img2)
    print(f"  Image shape: {img1.shape}")

    # ── Run inference ─────────────────────────────────────────────────────────
    print(f"Loading model: {args.model} (adapter={args.adapter}, device={args.device})")
    model = FlowModel(args.model, adapter=args.adapter, device=args.device)
    print(f"  {model.engine}")

    print("Running inference...")
    flow_pred = model.predict(img1, img2)
    print(f"  Flow shape: {flow_pred.shape}, dtype: {flow_pred.dtype}")
    print(f"  Flow range: [{flow_pred.min():.2f}, {flow_pred.max():.2f}]")

    # ── Save outputs ──────────────────────────────────────────────────────────
    if args.output:
        assert args.png or args.flo or args.npy, "Specify at least one output format with --png, --flo, or --npy"
        name = os.path.splitext(os.path.basename(args.img1))[0]
        print(f"Saving results to {args.output}")
        save_results(
            args.output,
            flow_pred,
            name=name,
            png=args.png,
            flo=args.flo,
            npy=args.npy,
        )

    print("Inference Done.")


if __name__ == "__main__":
    main()
