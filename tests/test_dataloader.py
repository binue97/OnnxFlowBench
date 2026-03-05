"""
Tests for the FlowEval dataloader package using real datasets.

Usage:
    python -m pytest tests/test_dataloader.py -v
"""

import sys, os
import pytest
import numpy as np
import torch

# ── ensure project root is on sys.path ──────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dataloader.template import FlowDataset
from dataloader.sintel import MpiSintel
from dataloader.chairs import FlyingChairs
from dataloader.kitti import KITTI
from dataloader.things import FlyingThings
from dataloader.spring import Spring
from dataloader.hd1k import HD1K
from dataloader.tartanair import TartanAir


# ═══════════════════════════════════════════════════════════════════════════════
# FlowDataset base class
# ═══════════════════════════════════════════════════════════════════════════════

class TestFlowDatasetBase:
    def test_empty_dataset_has_zero_length(self):
        ds = FlowDataset()
        assert len(ds) == 0

    def test_rmul_repeats_lists(self):
        ds = FlowDataset()
        ds.image_list = [["a.png", "b.png"]]
        ds.flow_list = ["f.flo"]
        ds = 3 * ds
        assert len(ds.image_list) == 3
        assert len(ds.flow_list) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# MPI Sintel
# ═══════════════════════════════════════════════════════════════════════════════

class TestMpiSintel:
    def test_discovery(self):
        ds = MpiSintel(split="training", dstype="clean")
        assert len(ds) > 0
        assert len(ds.flow_list) == len(ds.image_list)
        assert ds.is_test is False

    def test_both_dstypes_load(self):
        clean = MpiSintel(split="training", dstype="clean")
        final = MpiSintel(split="training", dstype="final")
        assert len(clean) > 0
        assert len(final) > 0

    def test_getitem_returns_tensors(self):
        ds = MpiSintel(split="training", dstype="clean")
        img1, img2, flow, valid = ds[0]
        assert isinstance(img1, torch.Tensor) and img1.shape[0] == 3
        assert isinstance(flow, torch.Tensor) and flow.shape[0] == 2
        assert valid.dtype == torch.float32

    def test_extra_info_populated(self):
        ds = MpiSintel(split="training", dstype="clean")
        scene, frame_id = ds.extra_info[0]
        assert isinstance(scene, str)
        assert isinstance(frame_id, int)


# ═══════════════════════════════════════════════════════════════════════════════
# FlyingChairs
# ═══════════════════════════════════════════════════════════════════════════════

class TestFlyingChairs:
    def test_discovery_training(self):
        ds = FlyingChairs(split="training")
        assert len(ds) > 0
        assert len(ds.flow_list) == len(ds.image_list)

    def test_discovery_validation(self):
        ds = FlyingChairs(split="validation")
        assert len(ds) > 0

    def test_getitem(self):
        ds = FlyingChairs(split="training")
        img1, img2, flow, valid = ds[0]
        assert img1.shape[0] == 3
        assert flow.shape[0] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# KITTI
# ═══════════════════════════════════════════════════════════════════════════════

class TestKITTI:
    def test_discovery(self):
        ds = KITTI(split="training")
        assert len(ds) > 0
        assert len(ds.flow_list) == len(ds.image_list)

    def test_extra_info(self):
        ds = KITTI(split="training")
        assert len(ds.extra_info) > 0
        assert ds.extra_info[0][0].endswith("_10.png")

    def test_getitem(self):
        ds = KITTI(split="training")
        img1, img2, flow, valid = ds[0]
        assert img1.ndim == 3 and img1.shape[0] == 3
        assert flow.shape[0] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# FlyingThings3D
# ═══════════════════════════════════════════════════════════════════════════════

class TestFlyingThings:
    def test_discovery(self):
        ds = FlyingThings(dstype="frames_cleanpass")
        assert len(ds) > 0
        assert len(ds.image_list) == len(ds.flow_list)

    def test_getitem(self):
        ds = FlyingThings(dstype="frames_cleanpass")
        img1, img2, flow, valid = ds[0]
        assert img1.shape[0] == 3
        assert flow.shape[0] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Spring
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpring:
    def test_discovery(self):
        ds = Spring(split="train")
        assert len(ds) > 0
        assert len(ds.flow_list) == len(ds.image_list)
        assert ds.is_test is False

    def test_invalid_split_raises(self):
        with pytest.raises(AssertionError):
            Spring(split="invalid")

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            Spring(root=str(tmp_path), split="train")

    def test_getitem(self):
        ds = Spring(split="train")
        img1, img2, flow, valid = ds[0]
        assert isinstance(img1, torch.Tensor) and img1.shape[0] == 3


# ═══════════════════════════════════════════════════════════════════════════════
# HD1K
# ═══════════════════════════════════════════════════════════════════════════════

class TestHD1K:
    def test_discovery(self):
        ds = HD1K()
        assert len(ds) > 0
        assert len(ds.flow_list) == len(ds.image_list)

    def test_getitem(self):
        ds = HD1K()
        img1, img2, flow, valid = ds[0]
        assert flow.shape[0] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# TartanAir
# ═══════════════════════════════════════════════════════════════════════════════

class TestTartanAir:
    def test_discovery(self):
        ds = TartanAir()
        assert len(ds) > 0
        assert len(ds.flow_list) == len(ds.image_list)
        assert len(ds.mask_list) == len(ds.image_list)

    def test_getitem(self):
        ds = TartanAir()
        img1, img2, flow, valid = ds[0]
        assert img1.shape[0] == 3
        assert flow.shape[0] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# DataLoader integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataLoaderIntegration:
    def test_dataloader_batch(self):
        ds = MpiSintel(split="training", dstype="clean")
        loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
        img1, img2, flow, valid = next(iter(loader))
        assert img1.shape[0] == 2
        assert flow.shape[1] == 2

    def test_concat_datasets(self):
        ds1 = MpiSintel(split="training", dstype="clean")
        ds2 = MpiSintel(split="training", dstype="final")
        combined = ds1 + ds2
        assert len(combined) == len(ds1) + len(ds2)

    def test_rmul_dataset(self):
        ds = KITTI(split="training")
        original_len = len(ds)
        ds = 5 * ds
        assert len(ds) == 5 * original_len


# ═══════════════════════════════════════════════════════════════════════════════
# Output quality checks
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutputShapesAndTypes:
    def test_image_range(self):
        ds = MpiSintel(split="training", dstype="clean")
        img1, img2, flow, valid = ds[0]
        assert img1.min() >= 0
        assert img1.max() <= 255

    def test_valid_mask_is_binary(self):
        ds = MpiSintel(split="training", dstype="clean")
        _, _, _, valid = ds[0]
        assert set(valid.unique().tolist()).issubset({0.0, 1.0})

    def test_flow_no_nan_inf(self):
        ds = MpiSintel(split="training", dstype="clean")
        _, _, flow, _ = ds[0]
        assert not torch.isnan(flow).any()
        assert not torch.isinf(flow).any()


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset statistics (run with -s to see printed output)
#   python -m pytest tests/test_dataloader.py -v -s -k "test_print_statistics"
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatasetStatistics:
    """Print summary statistics for every dataset."""

    DATASETS = [
        ("Sintel (clean)",       lambda: MpiSintel(split="training", dstype="clean")),
        ("Sintel (final)",       lambda: MpiSintel(split="training", dstype="final")),
        ("FlyingChairs (train)", lambda: FlyingChairs(split="training")),
        ("FlyingChairs (val)",   lambda: FlyingChairs(split="validation")),
        ("KITTI",                lambda: KITTI(split="training")),
        ("FlyingThings",         lambda: FlyingThings(dstype="frames_cleanpass")),
        ("Spring (train)",       lambda: Spring(split="train")),
        ("HD1K",                 lambda: HD1K()),
        ("TartanAir",            lambda: TartanAir()),
    ]

    def test_print_statistics(self):
        header = f"{'Dataset':<25} {'Image Pairs':>12} {'Flow Files':>12} {'Match':>6}"
        sep = "─" * len(header)
        print(f"\n{sep}")
        print(header)
        print(sep)

        for name, loader_fn in self.DATASETS:
            ds = loader_fn()
            n_pairs = len(ds.image_list)
            n_flows = len(ds.flow_list)
            match = "✓" if n_pairs == n_flows else "✗"
            print(f"{name:<25} {n_pairs:>12,} {n_flows:>12,} {match:>6}")
            assert n_pairs > 0, f"{name}: no image pairs found"
            assert n_pairs == n_flows, f"{name}: mismatch — {n_pairs} pairs vs {n_flows} flows"

        print(sep)
