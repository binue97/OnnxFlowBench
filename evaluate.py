"""
Evaluate an ONNX optical flow model on a dataset.

Iterates over all samples, runs inference, computes per-sample metrics,
and reports aggregated results.

Usage examples:

    # Evaluate on Sintel clean
    python evaluate.py --model raft.onnx --adapter raft --dataset sintel --dstype clean

    # Evaluate on multiple datasets
    python evaluate.py --model raft.onnx --adapter raft --dataset sintel kitti

    # Evaluate on KITTI training set, save per-sample CSV
    python evaluate.py --model raft.onnx --adapter raft --dataset kitti --output results/

    # Evaluate first 50 samples only
    python evaluate.py --model raft.onnx --adapter raft --dataset sintel --max-samples 50
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.flow_model import FlowModel
from core.registry import list_adapters
from metrics.flow_metrics import compute_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset loading
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_NAMES = ["sintel", "chairs", "kitti", "things", "spring", "hd1k", "tartanair"]


def positive_int(value: str) -> int:
    """argparse type that accepts only positive integers."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def effective_sample_count(ds, max_samples: int | None) -> int:
    """Return how many samples will actually be evaluated."""
    total = len(ds)
    if max_samples is None:
        return total
    return min(total, max_samples)


def build_dataset(name: str, dstype: str = "clean", split: str | None = None):
    """
    Build a dataset by name.

    Returns:
        A FlowDataset instance.
    """
    from dataloader import (
        MpiSintel,
        FlyingChairs,
        KITTI,
        FlyingThings,
        Spring,
        HD1K,
        TartanAir,
    )

    name = name.lower()

    if name == "sintel":
        ds = MpiSintel(dstype=dstype, split=split or "training")
    elif name == "chairs":
        ds = FlyingChairs(split=split or "validation")
    elif name == "kitti":
        ds = KITTI(split=split or "training")
    elif name == "things":
        ds = FlyingThings(dstype=dstype)
    elif name == "spring":
        ds = Spring(split=split or "train")
    elif name == "hd1k":
        ds = HD1K()
    elif name == "tartanair":
        ds = TartanAir()
    else:
        raise ValueError(f"Unknown dataset: {name!r}. Available: {DATASET_NAMES}")

    return ds


# ═══════════════════════════════════════════════════════════════════════════════
# Sample reading
# ═══════════════════════════════════════════════════════════════════════════════


def read_sample(ds, index: int):
    """
    Read a single sample (images + ground truth) from a dataset.

    Returns:
        img1:    (H, W, 3) uint8 RGB
        img2:    (H, W, 3) uint8 RGB
        flow_gt: (H, W, 2) float32
        valid:   (H, W) float32
    """
    img1 = np.array(Image.open(ds.image_list[index][0]).convert("RGB"))
    img2 = np.array(Image.open(ds.image_list[index][1]).convert("RGB"))

    flow_gt, valid = ds.read_flow(index)
    flow_gt = np.array(flow_gt, dtype=np.float32)
    valid = np.array(valid, dtype=np.float32)

    return img1, img2, flow_gt, valid


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation loop
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate(
    model: FlowModel,
    ds,
    dataset_name: str,
    max_samples: int | None = None,
) -> tuple[list[dict], dict]:
    """
    Run evaluation: iterate dataset, predict, compute metrics.

    Args:
        model:        FlowModel instance.
        ds:           FlowDataset instance (must have ground truth).
        dataset_name: Name for metric dispatch (e.g. "sintel").
        max_samples:  Limit number of samples (None = all).

    Returns:
        (per_sample_results, aggregated_results)
        per_sample_results: list of dicts, each with "index", "extra_info", and metric values.
        aggregated_results: dict of metric_name -> mean value.
    """
    assert not ds.is_test, "Cannot evaluate on test split (no ground truth)."

    n = effective_sample_count(ds, max_samples)

    per_sample = []
    metric_accum = {}
    total_time = 0.0

    for i in range(n):
        img1, img2, flow_gt, valid = read_sample(ds, i)

        t0 = time.perf_counter()
        flow_pred = model.predict(img1, img2)
        t1 = time.perf_counter()
        total_time += t1 - t0

        metrics = compute_metrics(flow_pred, flow_gt, valid, dataset=dataset_name)

        # Store per-sample result
        extra = ds.extra_info[i] if i < len(ds.extra_info) else ""
        row = {"index": i, "extra_info": str(extra)}
        row.update(metrics)
        per_sample.append(row)

        # Accumulate
        for k, v in metrics.items():
            if k not in metric_accum:
                metric_accum[k] = []
            metric_accum[k].append(v)

        # Progress
        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = total_time
            avg_ms = elapsed / (i + 1) * 1000
            epe_so_far = np.mean(metric_accum.get("epe", [0]))
            print(f"  [{i + 1}/{n}] epe={epe_so_far:.4f}, avg={avg_ms:.1f}ms/sample")

    # Aggregate
    aggregated = {k: float(np.mean(v)) for k, v in metric_accum.items()}
    aggregated["avg_time_ms"] = total_time / n * 1000
    aggregated["total_samples"] = n

    return per_sample, aggregated


# ═══════════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════════


def print_results(aggregated: dict, dataset_name: str):
    """Pretty-print aggregated results."""
    print(f"\n{'=' * 50}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Samples: {int(aggregated['total_samples'])}")
    print(f"  Avg inference time: {aggregated['avg_time_ms']:.1f} ms")
    print(f"{'=' * 50}")

    for k, v in aggregated.items():
        if k in ("avg_time_ms", "total_samples"):
            continue
        print(f"  {k:>10s}: {v:.4f}")

    print(f"{'=' * 50}\n")


def save_csv(per_sample: list[dict], path: str):
    """Save per-sample results as CSV."""
    if not per_sample:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys = per_sample[0].keys()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(per_sample)
    print(f"Saved per-sample results: {path}")


def save_summary(aggregated: dict, path: str):
    """Save aggregated results as a text file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for k, v in aggregated.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved summary: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an ONNX optical flow model on a dataset"
    )

    # Model
    parser.add_argument("--model", required=True, help="Path to .onnx model")
    parser.add_argument(
        "--adapter",
        default="raft",
        help=f"Adapter name ({', '.join(list_adapters())})",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    # Dataset
    parser.add_argument(
        "--dataset",
        required=True,
        nargs="+",
        choices=DATASET_NAMES,
        help="Dataset(s) to evaluate on",
    )
    parser.add_argument(
        "--dstype", default="clean", help="Dataset type (e.g. clean/final for Sintel)"
    )
    parser.add_argument(
        "--split", default=None, help="Dataset split (default: training/validation)"
    )
    parser.add_argument(
        "--max-samples",
        type=positive_int,
        default=None,
        help="Evaluate only the first N samples per dataset",
    )

    # Output
    parser.add_argument(
        "--output", default=None, help="Output directory for results CSV and summary"
    )

    args = parser.parse_args()

    # ── Build model ───────────────────────────────────────────────────────
    print(f"Loading model: {args.model} (adapter={args.adapter}, device={args.device})")
    model = FlowModel(args.model, adapter=args.adapter, device=args.device)
    print(f"  {model.engine}")

    # ── Evaluate each dataset ─────────────────────────────────────────────
    for dataset_name in args.dataset:
        print(f"\nLoading dataset: {dataset_name}")
        ds = build_dataset(dataset_name, dstype=args.dstype, split=args.split)
        print(f"  Samples: {len(ds)}")
        sample_count = effective_sample_count(ds, args.max_samples)

        if args.max_samples is None or sample_count == len(ds):
            print(f"Evaluating all {sample_count} samples...")
        else:
            print(f"Evaluating first {sample_count} of {len(ds)} samples...")
        per_sample, aggregated = evaluate(
            model, ds, dataset_name, max_samples=args.max_samples
        )

        print_results(aggregated, dataset_name)

        if args.output:
            ds_dir = os.path.join(args.output, dataset_name)
            save_csv(per_sample, os.path.join(ds_dir, "per_sample.csv"))
            save_summary(aggregated, os.path.join(ds_dir, "summary.txt"))

    print("Evaluation Done.")


if __name__ == "__main__":
    main()
