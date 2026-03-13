"""Inspect a .pt or .npy file and print its contents/metadata."""

import argparse
import os

import numpy as np
import torch


def _format_mem(nbytes: int) -> str:
    if nbytes < 1024:
        return f"{nbytes} B"
    elif nbytes < 1024 ** 2:
        return f"{nbytes / 1024:.2f} KB"
    else:
        return f"{nbytes / 1024 ** 2:.2f} MB"


def print_tensor_info(name, tensor, indent=2):
    prefix = " " * indent
    print(f"{prefix}{name}:")
    print(f"{prefix}  shape   = {list(tensor.shape)}")
    print(f"{prefix}  dtype   = {tensor.dtype}")
    print(f"{prefix}  device  = {tensor.device}")
    print(f"{prefix}  stride  = {tensor.stride()}")
    print(f"{prefix}  contiguous = {tensor.is_contiguous()}")
    print(f"{prefix}  requires_grad = {tensor.requires_grad}")
    print(f"{prefix}  min={tensor.min().item():.6g}  max={tensor.max().item():.6g}  mean={tensor.float().mean().item():.6g}")
    print(f"{prefix}  numel   = {tensor.numel()}")
    print(f"{prefix}  memory  = {_format_mem(tensor.numel() * tensor.element_size())}")


def print_ndarray_info(name, arr, indent=2):
    prefix = " " * indent
    print(f"{prefix}{name}:")
    print(f"{prefix}  shape   = {list(arr.shape)}")
    print(f"{prefix}  dtype   = {arr.dtype}")
    print(f"{prefix}  strides = {arr.strides}")
    print(f"{prefix}  contiguous = {arr.flags['C_CONTIGUOUS']}")
    print(f"{prefix}  min={arr.min():.6g}  max={arr.max():.6g}  mean={arr.astype(np.float64).mean():.6g}")
    print(f"{prefix}  size    = {arr.size}")
    print(f"{prefix}  memory  = {_format_mem(arr.nbytes)}")


def print_info(obj, name="root", indent=0):
    prefix = " " * indent
    if isinstance(obj, torch.Tensor):
        print_tensor_info(name, obj, indent)
    elif isinstance(obj, np.ndarray):
        print_ndarray_info(name, obj, indent)
    elif isinstance(obj, dict):
        print(f"{prefix}{name}: dict with {len(obj)} keys")
        for key in obj:
            print_info(obj[key], name=str(key), indent=indent + 2)
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{name}: {type(obj).__name__} with {len(obj)} elements")
        for i, item in enumerate(obj):
            print_info(item, name=f"[{i}]", indent=indent + 2)
    else:
        print(f"{prefix}{name}: {type(obj).__name__} = {obj}")


def load_file(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path, allow_pickle=False)
    elif ext == ".npz":
        return dict(np.load(path, allow_pickle=False))
    elif ext in (".pt", ".pth"):
        return torch.load(path, map_location="cpu", weights_only=False)
    else:
        raise ValueError(f"Unsupported file extension: {ext!r}")


def main():
    parser = argparse.ArgumentParser(description="Inspect a .pt / .npy file and print its metadata.")
    parser.add_argument("file", type=str, help="Path to the .pt, .pth, .npy, or .npz file")
    args = parser.parse_args()

    print(f"Loading: {args.file}")
    data = load_file(args.file)
    print(f"Top-level type: {type(data).__name__}\n")
    print_info(data)


if __name__ == "__main__":
    main()
