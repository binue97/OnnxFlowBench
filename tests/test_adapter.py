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
from core.adapters import FlowNetSAdapter, PWCNetAdapter, RAFTAdapter
from core.registry import (
    get_adapter,
    list_adapters,
    register_adapter,
    ADAPTER_REGISTRY,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _random_image(h=100, w=120):
    """Random (H, W, 3) uint8 image."""
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _mock_flow_output_chw(h, w):
    """Simulated ONNX output: (1, 2, H, W) flow in CHW layout."""
    return np.random.randn(1, 2, h, w).astype(np.float32)


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
# FlowNetS Adapter
# ═══════════════════════════════════════════════════════════════════════════════


class TestFlowNetSAdapter:
    def test_preprocess_concat_shape(self):
        adapter = FlowNetSAdapter()
        img = _random_image(h=100, w=120)
        feed = adapter.preprocess(img, img)
        assert len(feed) == 1
        assert "input" in feed
        t = feed["input"]
        assert t.shape[0] == 1 and t.shape[1] == 6  # concat
        assert t.shape[2] % 64 == 0 and t.shape[3] % 64 == 0

    def test_preprocess_meanstd_normalization(self):
        adapter = FlowNetSAdapter()
        img = np.full((64, 64, 3), 255, dtype=np.uint8)
        feed = adapter.preprocess(img, img)
        t = feed["input"]
        # FlowNetS uses (img/255 - mean) / std with mean=[0.411, 0.432, 0.45], std=[1,1,1]
        # After BGR reorder: channel 0 is B = (1.0 - 0.45) = 0.55
        # Values differ from simple unit normalization (which would give 1.0)
        assert t.dtype == np.float32
        np.testing.assert_allclose(t[0, 0, 0, 0], 0.55, atol=0.01)   # B channel
        np.testing.assert_allclose(t[0, 1, 0, 0], 0.568, atol=0.01)  # G channel
        np.testing.assert_allclose(t[0, 2, 0, 0], 0.589, atol=0.01)  # R channel

    def test_postprocess_restores_shape(self):
        adapter = FlowNetSAdapter()
        img = _random_image(h=100, w=120)
        feed = adapter.preprocess(img, img)
        _, _, ph, pw = feed["input"].shape
        mock = {"output": _mock_flow_output_chw(ph, pw)}
        flow = adapter.postprocess(mock)
        assert flow.shape == (100, 120, 2)
        assert flow.dtype == np.float32

    def test_postprocess_scales_flow_on_resize(self):
        """Flow values should be scaled by the resize ratio."""
        adapter = FlowNetSAdapter()
        img = _random_image(h=50, w=100)  # → interpolated to 64x128
        feed = adapter.preprocess(img, img)
        _, _, ph, pw = feed["input"].shape
        assert ph == 64 and pw == 128

        raw = np.ones((1, 2, ph, pw), dtype=np.float32)
        flow = adapter.postprocess({"output": raw})
        assert flow.shape == (50, 100, 2)
        # Horizontal: 100/128 ≈ 0.78
        np.testing.assert_allclose(flow[25, 50, 0], 100.0 / 128.0, atol=0.05)
        # Vertical: 50/64 ≈ 0.78
        np.testing.assert_allclose(flow[25, 50, 1], 50.0 / 64.0, atol=0.05)


# ═══════════════════════════════════════════════════════════════════════════════
# PWC-Net Adapter
# ═══════════════════════════════════════════════════════════════════════════════


class TestPWCNetAdapter:
    def test_preprocess_concat_shape(self):
        adapter = PWCNetAdapter()
        img = _random_image(h=384, w=512)
        feed = adapter.preprocess(img, img)
        assert len(feed) == 1
        assert "input" in feed
        t = feed["input"]
        assert t.shape[1] == 6  # concat
        assert t.shape[2] % 64 == 0 and t.shape[3] % 64 == 0

    def test_preprocess_unit_normalization(self):
        adapter = PWCNetAdapter()
        img = _random_image(h=64, w=64)
        feed = adapter.preprocess(img, img)
        t = feed["input"]
        assert t.min() >= 0.0
        assert t.max() <= 1.0

    def test_postprocess_quarter_res_upsampled(self):
        adapter = PWCNetAdapter()
        img = _random_image(h=384, w=512)
        feed = adapter.preprocess(img, img)
        _, _, ph, pw = feed["input"].shape
        # Model outputs at 1/4 resolution
        mock = {"flow": _mock_flow_output_chw(ph // 4, pw // 4)}
        flow = adapter.postprocess(mock)
        assert flow.shape == (384, 512, 2)

    def test_postprocess_applies_output_scale(self):
        adapter = PWCNetAdapter()
        img = _random_image(h=64, w=64)
        feed = adapter.preprocess(img, img)
        _, _, ph, pw = feed["input"].shape
        # Constant flow of 1.0 at quarter res
        raw = np.ones((1, 2, ph // 4, pw // 4), dtype=np.float32)
        flow = adapter.postprocess({"flow": raw})
        # After scale (×20) and 4× upsample without flow scaling, center should be 20.0
        np.testing.assert_allclose(flow[32, 32, :], 20.0, atol=1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# RAFT Adapter
# ═══════════════════════════════════════════════════════════════════════════════


class TestRAFTAdapter:
    def test_preprocess_separate_inputs(self):
        adapter = RAFTAdapter()
        img = _random_image(h=436, w=1024)
        feed = adapter.preprocess(img, img)
        assert "image1" in feed and "image2" in feed
        _, _, h, w = feed["image1"].shape
        assert h % 8 == 0 and w % 8 == 0

    def test_preprocess_no_normalization(self):
        adapter = RAFTAdapter()
        img = _random_image(h=64, w=64)
        feed = adapter.preprocess(img, img)
        t = feed["image1"]
        assert t.min() >= 0.0
        assert t.max() <= 255.0

    def test_preprocess_already_divisible(self):
        adapter = RAFTAdapter()
        img = _random_image(h=64, w=64)
        feed = adapter.preprocess(img, img)
        assert feed["image1"].shape == (1, 3, 64, 64)

    def test_postprocess_crops_to_original(self):
        adapter = RAFTAdapter()
        img = _random_image(h=436, w=1024)
        feed = adapter.preprocess(img, img)
        _, _, ph, pw = feed["image1"].shape
        mock = {"flow": _mock_flow_output_chw(ph, pw)}
        flow = adapter.postprocess(mock)
        assert flow.shape == (436, 1024, 2)

    def test_dtype_is_float32(self):
        adapter = RAFTAdapter()
        img = _random_image(h=64, w=64)
        feed = adapter.preprocess(img, img)
        assert feed["image1"].dtype == np.float32


# ═══════════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegistry:
    def test_list_adapters(self):
        names = list_adapters()
        assert "raft" in names
        assert "pwcnet" in names
        assert "flownets" in names

    def test_get_known_adapter(self):
        adapter = get_adapter("raft")
        assert isinstance(adapter, RAFTAdapter)

    def test_get_flownets(self):
        adapter = get_adapter("flownets")
        assert isinstance(adapter, FlowNetSAdapter)

    def test_get_pwcnet(self):
        adapter = get_adapter("pwcnet")
        assert isinstance(adapter, PWCNetAdapter)

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown adapter"):
            get_adapter("nonexistent_model")

    def test_case_insensitive(self):
        a1 = get_adapter("RAFT")
        a2 = get_adapter("raft")
        assert type(a1) is type(a2)

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
