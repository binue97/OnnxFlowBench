"""Compare two tensor files (.npy or .pt) element-wise."""

import argparse
import numpy as np


def load_tensor(path: str) -> np.ndarray:
    """Load a tensor from .npy or .pt file and return as numpy array."""
    if path.endswith(".npy"):
        return np.load(path)
    elif path.endswith(".pt"):
        import torch
        return torch.load(path, map_location="cpu").numpy()
    else:
        raise ValueError(f"Unsupported file format: {path}")


def print_info(name: str, arr: np.ndarray):
    """Print shape, dtype, and basic stats of a tensor."""
    print(f"  shape : {arr.shape}")
    print(f"  dtype : {arr.dtype}")
    print(f"  min   : {arr.min():.6f}")
    print(f"  max   : {arr.max():.6f}")
    print(f"  mean  : {arr.mean():.6f}")
    print(f"  std   : {arr.std():.6f}")


def main():
    parser = argparse.ArgumentParser(description="Compare two tensor files (.npy / .pt)")
    parser.add_argument("file_a", help="Path to first tensor file")
    parser.add_argument("file_b", help="Path to second tensor file")
    args = parser.parse_args()

    a = load_tensor(args.file_a)
    b = load_tensor(args.file_b)

    print(f"\n[A] {args.file_a}")
    print_info("A", a)

    print(f"\n[B] {args.file_b}")
    print_info("B", b)

    # --- Comparison ---
    print("\n[Comparison]")
    if a.shape != b.shape:
        print(f"  ⚠ Shape mismatch: {a.shape} vs {b.shape}")
        return

    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    print(f"  abs diff  max  : {diff.max():.8f}")
    print(f"  abs diff  mean : {diff.mean():.8f}")
    print()
    print(f"  allclose(1e-1) : {np.allclose(a, b, atol=1e-1)}")
    print(f"  allclose(1e-2) : {np.allclose(a, b, atol=1e-2)}")
    print(f"  allclose(1e-3) : {np.allclose(a, b, atol=1e-3)}")
    print(f"  allclose(1e-4) : {np.allclose(a, b, atol=1e-4)}")
    print(f"  allclose(1e-5) : {np.allclose(a, b, atol=1e-5)}")
    print()


if __name__ == "__main__":
    main()
