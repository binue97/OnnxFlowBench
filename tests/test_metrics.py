"""
Tests for optical flow evaluation metrics.

All tests use synthetic flow data with known expected values - no model or dataset needed.

Usage:
    python -m pytest tests/test_metrics.py -v
"""

import sys
import os
import pytest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from metrics.flow_metrics import (
    epe,
    fl_all,
    n_pixel,
    compute_metrics,
    _validate_inputs,
    _endpoint_error,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _constant_flow(h=10, w=10, dx=1.0, dy=0.0):
    """Create a constant flow field."""
    flow = np.zeros((h, w, 2), dtype=np.float32)
    flow[..., 0] = dx
    flow[..., 1] = dy
    return flow


def _all_valid(h=10, w=10):
    return np.ones((h, w), dtype=np.float32)


def _all_invalid(h=10, w=10):
    return np.zeros((h, w), dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# EPE
# ═══════════════════════════════════════════════════════════════════════════════


class TestEPE:
    def test_perfect_prediction(self):
        """EPE should be 0 when pred == gt."""
        flow = _constant_flow(dx=3.0, dy=4.0)
        assert epe(flow, flow, _all_valid()) == pytest.approx(0.0)

    def test_known_constant_error(self):
        """Constant offset of (3,4) -> EPE = 5.0."""
        pred = _constant_flow(dx=0.0, dy=0.0)
        gt = _constant_flow(dx=3.0, dy=4.0)
        assert epe(pred, gt, _all_valid()) == pytest.approx(5.0)

    def test_unit_error(self):
        """Offset of (1,0) -> EPE = 1.0."""
        pred = _constant_flow(dx=0.0, dy=0.0)
        gt = _constant_flow(dx=1.0, dy=0.0)
        assert epe(pred, gt, _all_valid()) == pytest.approx(1.0)

    def test_one_valid_pixel(self):
        """Only valid pixels should contribute."""
        H, W = 4, 4
        pred = np.zeros((H, W, 2), dtype=np.float32)
        gt = np.zeros((H, W, 2), dtype=np.float32)
        gt[0, 0] = [3.0, 4.0]  # EPE = 5 at this pixel
        gt[2, 2] = [100.0, 0.0]  # EPE = 100 at this pixel (will be masked out)

        valid = np.zeros((H, W), dtype=np.float32)
        valid[0, 0] = 1.0  # Only pixel at [0, 0] is valid
        assert epe(pred, gt, valid) == pytest.approx(5.0)

    def test_all_invalid(self):
        """EPE should be 0 when no pixels are valid."""
        assert epe(_constant_flow(), _constant_flow(dx=5.0), _all_invalid()) == 0.0

    def test_partial_mask(self):
        """Half pixels with EPE=2, half with EPE=4 -> mean=3."""
        H, W = 2, 2
        pred = np.zeros((H, W, 2), dtype=np.float32)
        gt = np.zeros((H, W, 2), dtype=np.float32)
        gt[0, 0] = [2.0, 0.0]  # EPE = 2
        gt[0, 1] = [2.0, 0.0]  # EPE = 2
        gt[1, 0] = [4.0, 0.0]  # EPE = 4
        gt[1, 1] = [4.0, 0.0]  # EPE = 4
        valid = _all_valid(H, W)
        assert epe(pred, gt, valid) == pytest.approx(3.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Fl-all (KITTI outlier metric)
# ═══════════════════════════════════════════════════════════════════════════════


class TestFlAll:
    def test_perfect_prediction(self):
        """No outliers when pred == gt."""
        flow = _constant_flow(dx=10.0, dy=10.0)
        assert fl_all(flow, flow, _all_valid()) == pytest.approx(0.0)

    def test_all_outliers(self):
        """Every pixel has EPE=10 with gt_mag=0 -> all outliers."""
        pred = _constant_flow(dx=10.0, dy=0.0)
        gt = _constant_flow(dx=0.0, dy=0.0)
        assert fl_all(pred, gt, _all_valid()) == pytest.approx(100.0)

    def test_below_epe_threshold(self):
        """EPE=2, epe_threshold=3"""
        pred = _constant_flow(dx=2.0, dy=0.0)
        gt = _constant_flow(dx=0.0, dy=0.0)
        assert fl_all(pred, gt, _all_valid()) == pytest.approx(0.0)

    def test_below_relative_threshold(self):
        """EPE=4, relative_threshold = 100 * 0.05 = 5.0"""
        pred = _constant_flow(dx=104.0, dy=0.0)
        gt = _constant_flow(dx=100.0, dy=0.0)
        print(epe(pred, gt, _all_valid()))
        assert fl_all(pred, gt, _all_valid()) == pytest.approx(0.0)

    def test_mixed_outliers(self):
        """2 out of 4 pixels are outliers -> Fl-all = 0.5."""
        H, W = 2, 2
        pred = np.zeros((H, W, 2), dtype=np.float32)
        gt = np.zeros((H, W, 2), dtype=np.float32)

        # Pixel (0,0): EPE=10, gt_mag=0 -> outlier
        pred[0, 0] = [10.0, 0.0]
        gt[0, 0] = [0.0, 0.0]

        # Pixel (0,1): EPE=10, gt_mag=0 -> outlier
        pred[0, 1] = [10.0, 0.0]
        gt[0, 1] = [0.0, 0.0]

        # Pixel (1,0): EPE=1, gt_mag=0 -> inlier
        pred[1, 0] = [1.0, 0.0]
        gt[1, 0] = [0.0, 0.0]

        # Pixel (1,1): EPE=0, gt_mag=0 -> inlier
        pred[1, 1] = [0.0, 0.0]
        gt[1, 1] = [0.0, 0.0]

        valid = _all_valid(H, W)
        assert fl_all(pred, gt, valid) == pytest.approx(50.0)

    def test_all_invalid(self):
        pred = _constant_flow(dx=100.0)
        gt = _constant_flow(dx=0.0)
        assert fl_all(pred, gt, _all_invalid()) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# N-pixel metric
# ═══════════════════════════════════════════════════════════════════════════════


class TestNPixel:
    def test_perfect_prediction(self):
        flow = _constant_flow(dx=5.0, dy=5.0)
        assert n_pixel(flow, flow, _all_valid(), n=1.0) == pytest.approx(0.0)

    def test_all_above_threshold(self):
        """EPE=5 everywhere, threshold=1 -> 100% above."""
        pred = _constant_flow(dx=5.0, dy=0.0)
        gt = _constant_flow(dx=0.0, dy=0.0)
        assert n_pixel(pred, gt, _all_valid(), n=1.0) == pytest.approx(100.0)

    def test_all_below_threshold(self):
        """EPE=0.5 everywhere, threshold=1 -> 0% above."""
        pred = _constant_flow(dx=0.5, dy=0.0)
        gt = _constant_flow(dx=0.0, dy=0.0)
        assert n_pixel(pred, gt, _all_valid(), n=1.0) == pytest.approx(0.0)

    def test_exactly_at_threshold(self):
        """EPE exactly equals threshold -> NOT counted (strictly greater)."""
        pred = _constant_flow(dx=1.0, dy=0.0)
        gt = _constant_flow(dx=0.0, dy=0.0)
        assert n_pixel(pred, gt, _all_valid(), n=1.0) == pytest.approx(0.0)

    def test_mixed_with_different_thresholds(self):
        """4 pixels with EPE = 0.5, 1.5, 2.5, 4.0."""
        H, W = 2, 2
        pred = np.zeros((H, W, 2), dtype=np.float32)
        gt = np.zeros((H, W, 2), dtype=np.float32)
        pred[0, 0] = [0.5, 0.0]  # EPE = 0.5
        pred[0, 1] = [1.5, 0.0]  # EPE = 1.5
        pred[1, 0] = [2.5, 0.0]  # EPE = 2.5
        pred[1, 1] = [4.0, 0.0]  # EPE = 4.0
        valid = _all_valid(H, W)

        # 1px: 3 out of 4 above (1.5, 2.5, 4.0)
        assert n_pixel(pred, gt, valid, n=1.0) == pytest.approx(75.0)
        # 3px: 1 out of 4 above (4.0)
        assert n_pixel(pred, gt, valid, n=3.0) == pytest.approx(25.0)
        # 5px: 0 out of 4
        assert n_pixel(pred, gt, valid, n=5.0) == pytest.approx(0.0)

    def test_mask_respected(self):
        """High-error pixel masked out should not count."""
        H, W = 2, 1
        pred = np.zeros((H, W, 2), dtype=np.float32)
        gt = np.zeros((H, W, 2), dtype=np.float32)
        pred[0, 0] = [0.5, 0.0]  # EPE = 0.5
        pred[1, 0] = [100.0, 0.0]  # EPE = 100, but invalid

        valid = np.array([[1.0], [0.0]], dtype=np.float32)
        assert n_pixel(pred, gt, valid, n=1.0) == pytest.approx(0.0)

    def test_all_invalid(self):
        assert (
            n_pixel(_constant_flow(dx=100.0), _constant_flow(), _all_invalid(), n=1.0)
            == 0.0
        )
        assert (
            n_pixel(_constant_flow(dx=100.0), _constant_flow(), _all_invalid(), n=3.0)
            == 0.0
        )
        assert (
            n_pixel(_constant_flow(dx=100.0), _constant_flow(), _all_invalid(), n=5.0)
            == 0.0
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases & numerical robustness
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_single_pixel(self):
        pred = np.array([[[3.0, 4.0]]], dtype=np.float32)
        gt = np.array([[[0.0, 0.0]]], dtype=np.float32)
        valid = np.array([[1.0]], dtype=np.float32)
        assert epe(pred, gt, valid) == pytest.approx(5.0)

    def test_large_flow_values(self):
        pred = _constant_flow(dx=1e4, dy=1e4)
        gt = _constant_flow(dx=0.0, dy=0.0)
        valid = _all_valid()
        expected = np.sqrt(1e4**2 + 1e4**2)
        assert epe(pred, gt, valid) == pytest.approx(expected, rel=1e-5)

    def test_non_square_image(self):
        pred = np.zeros((3, 7, 2), dtype=np.float32)
        gt = np.ones((3, 7, 2), dtype=np.float32)
        valid = np.ones((3, 7), dtype=np.float32)
        assert epe(pred, gt, valid) == pytest.approx(np.sqrt(2.0))

    def test_checkerboard_mask(self):
        """Alternating valid/invalid pixels."""
        H, W = 4, 4
        pred = np.zeros((H, W, 2), dtype=np.float32)
        gt = np.zeros((H, W, 2), dtype=np.float32)

        # All pixels have EPE = 2
        gt[..., 0] = 2.0

        # Checkerboard mask: only half the pixels
        valid = np.zeros((H, W), dtype=np.float32)
        valid[0::2, 0::2] = 1.0
        valid[1::2, 1::2] = 1.0

        # EPE should still be 2 regardless of mask pattern
        assert epe(pred, gt, valid) == pytest.approx(2.0)
