"""
Tests for adapter utilities, preset adapters, and the registry.

All tests use synthetic images and mock ONNX outputs - no real model needed.

Usage:
    python -m pytest tests/test_adapter.py -v
"""

import sys
import os
import pytest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.base_adapter import ModelAdapter
from core import adapter_utils as utils
from core.adapters import *
from core.registry import (
    get_adapter,
    register_adapter,
    ADAPTER_REGISTRY,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════════════


class TestNormalizeUnit:
    def test_scales_to_0_1(self):
        img = np.full((4, 4, 3), 255, dtype=np.uint8)
        out = utils.normalize_unit(img)
        np.testing.assert_allclose(out, 1.0)
        assert out.dtype == np.float32

    def test_zero_stays_zero(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        out = utils.normalize_unit(img)
        np.testing.assert_allclose(out, 0.0)


class TestNormalizeMeanStd:
    def test_known_values(self):
        img = np.full((4, 4, 3), 255, dtype=np.uint8)
        out = utils.normalize_meanstd(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # (1.0 - 0.5) / 0.5 = 1.0
        np.testing.assert_allclose(out, 1.0)

    def test_default_imagenet_stats(self):
        img = np.full((4, 4, 3), 255, dtype=np.uint8)
        out = utils.normalize_meanstd(img)
        # Should NOT be in [0, 1]
        assert out.min() < 0 or out.max() > 1


class TestRgbToBgr:
    def test_flips_channels(self):
        img = np.zeros((2, 2, 3), dtype=np.float32)
        img[..., 0] = 1.0  # R=1
        out = utils.rgb_to_bgr(img)
        assert out[0, 0, 2] == 1.0  # now B=1
        assert out[0, 0, 0] == 0.0


class TestLayoutConversion:
    def test_hwc_to_chw(self):
        img = np.zeros((4, 5, 3))
        out = utils.hwc_to_chw(img)
        assert out.shape == (3, 4, 5)

    def test_chw_to_hwc(self):
        t = np.zeros((2, 4, 5))
        out = utils.chw_to_hwc(t)
        assert out.shape == (4, 5, 2)

    def test_roundtrip(self):
        img = np.random.randn(10, 12, 3).astype(np.float32)
        roundtrip = utils.chw_to_hwc(utils.hwc_to_chw(img))
        np.testing.assert_array_equal(roundtrip, img)


class TestBatchDim:
    def test_add(self):
        img = np.zeros((3, 4, 5))
        out = utils.add_batch_dim(img)
        assert out.shape == (1, 3, 4, 5)

    def test_remove_single(self):
        x = np.zeros((1, 2, 4, 5))
        out = utils.remove_batch_dim(x)
        assert out.shape == (2, 4, 5)

    def test_remove_nested(self):
        x = np.zeros((1, 1, 2, 4, 5))
        out = utils.remove_batch_dim(x)
        assert out.shape == (2, 4, 5)

    def test_remove_3d_unchanged(self):
        x = np.zeros((2, 4, 5))
        out = utils.remove_batch_dim(x)
        assert out.shape == (2, 4, 5)


class TestInterpolateToDivisible:
    def test_resizes_to_divisible(self):
        img = np.ones((3, 100, 100))
        out = utils.interpolate_to_divisible(img, 32)
        assert out.shape[1] % 32 == 0 and out.shape[2] % 32 == 0
        assert out.shape == (3, 128, 128)

    def test_already_divisible_no_change(self):
        img = np.ones((3, 64, 64))
        out = utils.interpolate_to_divisible(img, 8)
        assert out.shape == (3, 64, 64)

    def test_factor_1_no_resize(self):
        img = np.ones((3, 37, 53))
        out = utils.interpolate_to_divisible(img, 1)
        assert out.shape == (3, 37, 53)


class TestSelectOutput:
    def test_by_index(self):
        outputs = {"a": np.array([1]), "b": np.array([2])}
        np.testing.assert_array_equal(utils.select_output(outputs, 0), [1])
        np.testing.assert_array_equal(utils.select_output(outputs, 1), [2])

    def test_by_name(self):
        outputs = {"flow": np.array([1]), "extra": np.array([2])}
        np.testing.assert_array_equal(utils.select_output(outputs, "flow"), [1])

    def test_index_oob_raises(self):
        with pytest.raises(IndexError, match="out of range"):
            utils.select_output({"a": np.array([1])}, 5)

    def test_name_missing_raises(self):
        with pytest.raises(KeyError, match="not found"):
            utils.select_output({"a": np.array([1])}, "nonexistent")


class TestResizeFlow:
    def test_noop_same_size(self):
        flow = np.ones((10, 10, 2), dtype=np.float32)
        out = utils.resize_flow(flow, 10, 10)
        np.testing.assert_array_equal(out, flow)

    def test_scales_values(self):
        flow = np.ones((10, 10, 2), dtype=np.float32)
        out = utils.resize_flow(flow, 20, 20, scale_flow=True)
        assert out.shape == (20, 20, 2)
        np.testing.assert_allclose(out[5, 5, 0], 2.0, atol=0.1)
        np.testing.assert_allclose(out[5, 5, 1], 2.0, atol=0.1)

    def test_no_scale(self):
        flow = np.ones((10, 10, 2), dtype=np.float32)
        out = utils.resize_flow(flow, 20, 20, scale_flow=False)
        assert out.shape == (20, 20, 2)
        np.testing.assert_allclose(out[5, 5, 0], 1.0, atol=0.1)


# ═══════════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegistry:
    def test_get_known_adapter(self):
        adapter = get_adapter("flownets")
        assert isinstance(adapter, FlowNetSAdapter)
        adapter = get_adapter("raft")
        assert isinstance(adapter, RaftAdapter)
        adapter = get_adapter("ofnet")
        assert isinstance(adapter, OFNetAdapter)

    def test_get_unknown_adapter(self):
        with pytest.raises(KeyError, match="Unknown adapter"):
            get_adapter("nonexistent_model")

    def test_case_insensitive(self):
        flownets_large = get_adapter("FlowNetS")
        flownets_small = get_adapter("flownets")
        assert type(flownets_large) is type(flownets_small)
        raft_large = get_adapter("RAFT")
        raft_small = get_adapter("raft")
        assert type(raft_large) is type(raft_small)
        ofnet_large = get_adapter("OFNet")
        ofnet_small = get_adapter("ofnet")
        assert type(ofnet_large) is type(ofnet_small)

    def test_register_custom_class(self):
        class MyAdapter(ModelAdapter):
            def preprocess(self, img1, img2):
                return {"x": img1}

            def postprocess(self, outputs):
                return outputs["x"]

        register_adapter("custom_test", MyAdapter)
        adapter = get_adapter("custom_test")
        assert isinstance(adapter, MyAdapter)
        # Cleanup
        del ADAPTER_REGISTRY["custom_test"]
