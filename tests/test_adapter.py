"""
Tests for ModelAdapter, DefaultAdapter, AdapterConfig, and the registry.

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
from core.adapter_config import AdapterConfig
from core.default_adapter import DefaultAdapter
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


def _mock_flow_output_hwc(h, w):
    """Simulated ONNX output: (1, H, W, 2) flow in HWC layout."""
    return np.random.randn(1, h, w, 2).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# AdapterConfig defaults
# ═══════════════════════════════════════════════════════════════════════════════


class TestAdapterConfig:
    def test_default_values(self):
        cfg = AdapterConfig()
        assert cfg.input_names == ["image1", "image2"]
        assert cfg.input_format == "separate"
        assert cfg.normalization == "none"
        assert cfg.padding_factor == 8
        assert cfg.output_layout == "CHW"
        assert cfg.output_scale == 1.0
        assert cfg.output_resolution == "full"

    def test_custom_values(self):
        cfg = AdapterConfig(
            input_names=["input"],
            input_format="concat",
            normalization="unit",
            padding_factor=64,
        )
        assert cfg.input_format == "concat"
        assert cfg.normalization == "unit"
        assert cfg.padding_factor == 64


# ═══════════════════════════════════════════════════════════════════════════════
# DefaultAdapter - Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════


class TestPreprocessNormalization:
    def test_none_keeps_0_255(self):
        adapter = DefaultAdapter(AdapterConfig(normalization="none"))
        img1, img2 = _random_image(), _random_image()
        feed = adapter.preprocess(img1, img2)
        # Values should be in [0, 255]
        tensor = feed["image1"]
        assert tensor.min() >= 0.0
        assert tensor.max() <= 255.0

    def test_unit_scales_to_0_1(self):
        adapter = DefaultAdapter(AdapterConfig(normalization="unit"))
        img1, img2 = _random_image(), _random_image()
        feed = adapter.preprocess(img1, img2)
        tensor = feed["image1"]
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_imagenet_mean_centered(self):
        adapter = DefaultAdapter(AdapterConfig(normalization="imagenet"))
        # Use a constant white image so we can predict the output
        img = np.full((8, 8, 3), 255, dtype=np.uint8)
        feed = adapter.preprocess(img, img)
        tensor = feed["image1"]  # (1, 3, H, W)
        # After (255/255 - mean) / std, values should NOT be in [0, 1]
        assert tensor.min() < 0 or tensor.max() > 1

    def test_unknown_normalization_raises(self):
        adapter = DefaultAdapter(AdapterConfig(normalization="weird"))
        with pytest.raises(ValueError, match="Unknown normalization"):
            adapter.preprocess(_random_image(), _random_image())


class TestPreprocessFormat:
    def test_separate_produces_two_tensors(self):
        cfg = AdapterConfig(input_names=["img1", "img2"], input_format="separate")
        adapter = DefaultAdapter(cfg)
        feed = adapter.preprocess(_random_image(), _random_image())
        assert "img1" in feed and "img2" in feed
        assert feed["img1"].shape[1] == 3  # (1, 3, H, W)

    def test_concat_produces_one_6ch_tensor(self):
        cfg = AdapterConfig(input_names=["input"], input_format="concat")
        adapter = DefaultAdapter(cfg)
        feed = adapter.preprocess(_random_image(), _random_image())
        assert len(feed) == 1
        assert feed["input"].shape[1] == 6  # (1, 6, H, W)


class TestPreprocessPadding:
    def test_pads_to_divisible(self):
        adapter = DefaultAdapter(AdapterConfig(padding_factor=32))
        img = _random_image(h=100, w=100)  # not divisible by 32
        feed = adapter.preprocess(img, img)
        _, _, h, w = feed["image1"].shape
        assert h % 32 == 0 and w % 32 == 0

    def test_already_divisible_no_change(self):
        adapter = DefaultAdapter(AdapterConfig(padding_factor=8))
        img = _random_image(h=64, w=64)  # already divisible by 8
        feed = adapter.preprocess(img, img)
        assert feed["image1"].shape == (1, 3, 64, 64)

    def test_padding_factor_1_no_pad(self):
        adapter = DefaultAdapter(AdapterConfig(padding_factor=1))
        img = _random_image(h=37, w=53)
        feed = adapter.preprocess(img, img)
        assert feed["image1"].shape == (1, 3, 37, 53)

    def test_zero_padding_mode(self):
        cfg = AdapterConfig(padding_factor=32, padding_mode="zero")
        adapter = DefaultAdapter(cfg)
        img = np.full((30, 30, 3), 128, dtype=np.uint8)
        feed = adapter.preprocess(img, img)
        tensor = feed["image1"][0]  # (3, H, W)
        # Padded region should be 0
        assert tensor[0, 31, 0] == 0.0

    def test_replicate_padding_mode(self):
        cfg = AdapterConfig(padding_factor=32, padding_mode="replicate")
        adapter = DefaultAdapter(cfg)
        img = np.full((30, 30, 3), 128, dtype=np.uint8)
        feed = adapter.preprocess(img, img)
        tensor = feed["image1"][0]  # (3, H, W)
        # Padded region should replicate edge value (128.0)
        assert tensor[0, 31, 0] == 128.0


class TestPreprocessShape:
    def test_output_is_nchw(self):
        adapter = DefaultAdapter(AdapterConfig())
        img = _random_image(h=64, w=64)
        feed = adapter.preprocess(img, img)
        assert feed["image1"].ndim == 4
        assert feed["image1"].shape[0] == 1  # batch
        assert feed["image1"].shape[1] == 3  # channels

    def test_dtype_is_float32(self):
        adapter = DefaultAdapter(AdapterConfig())
        feed = adapter.preprocess(_random_image(), _random_image())
        assert feed["image1"].dtype == np.float32


# ═══════════════════════════════════════════════════════════════════════════════
# DefaultAdapter - Postprocessing
# ═══════════════════════════════════════════════════════════════════════════════


class TestPostprocessBasic:
    def test_chw_output(self):
        """Standard CHW model output -> (H, W, 2)."""
        adapter = DefaultAdapter(AdapterConfig(output_layout="CHW"))
        img = _random_image(h=64, w=64)
        adapter.preprocess(img, img)  # sets internal state

        mock_out = {"flow": _mock_flow_output_chw(64, 64)}
        flow = adapter.postprocess(mock_out)
        assert flow.shape == (64, 64, 2)
        assert flow.dtype == np.float32

    def test_hwc_output(self):
        """HWC model output -> (H, W, 2)."""
        adapter = DefaultAdapter(AdapterConfig(output_layout="HWC"))
        img = _random_image(h=64, w=64)
        adapter.preprocess(img, img)

        mock_out = {"flow": _mock_flow_output_hwc(64, 64)}
        flow = adapter.postprocess(mock_out)
        assert flow.shape == (64, 64, 2)

    def test_output_by_name(self):
        adapter = DefaultAdapter(AdapterConfig(output_name="flow_pred"))
        img = _random_image(h=64, w=64)
        adapter.preprocess(img, img)

        mock_out = {"flow_pred": _mock_flow_output_chw(64, 64), "extra": np.zeros(1)}
        flow = adapter.postprocess(mock_out)
        assert flow.shape == (64, 64, 2)

    def test_output_by_index(self):
        adapter = DefaultAdapter(AdapterConfig(output_name=1))
        img = _random_image(h=64, w=64)
        adapter.preprocess(img, img)

        mock_out = {
            "aux": np.zeros((1, 1, 64, 64), dtype=np.float32),
            "flow": _mock_flow_output_chw(64, 64),
        }
        flow = adapter.postprocess(mock_out)
        assert flow.shape == (64, 64, 2)

    def test_output_name_missing_raises(self):
        adapter = DefaultAdapter(AdapterConfig(output_name="nonexistent"))
        img = _random_image(h=64, w=64)
        adapter.preprocess(img, img)
        with pytest.raises(KeyError, match="not found"):
            adapter.postprocess({"flow": _mock_flow_output_chw(64, 64)})

    def test_output_index_oob_raises(self):
        adapter = DefaultAdapter(AdapterConfig(output_name=5))
        img = _random_image(h=64, w=64)
        adapter.preprocess(img, img)
        with pytest.raises(IndexError, match="out of range"):
            adapter.postprocess({"flow": _mock_flow_output_chw(64, 64)})


class TestPostprocessScale:
    def test_scale_multiplies_flow(self):
        adapter = DefaultAdapter(AdapterConfig(output_scale=2.0))
        img = _random_image(h=64, w=64)
        adapter.preprocess(img, img)

        raw = np.ones((1, 2, 64, 64), dtype=np.float32)
        flow = adapter.postprocess({"flow": raw})
        np.testing.assert_allclose(flow, 2.0)


class TestPostprocessUnpad:
    def test_unpad_crops_to_original(self):
        """Padded output should be cropped back to original size."""
        adapter = DefaultAdapter(AdapterConfig(padding_factor=32))
        img = _random_image(h=100, w=100)
        feed = adapter.preprocess(img, img)
        _, _, padded_h, padded_w = feed["image1"].shape

        mock_out = {"flow": _mock_flow_output_chw(padded_h, padded_w)}
        flow = adapter.postprocess(mock_out)
        assert flow.shape == (100, 100, 2)


class TestPostprocessUpsample:
    def test_quarter_resolution_upsampled(self):
        adapter = DefaultAdapter(
            AdapterConfig(output_resolution="quarter", padding_factor=1)
        )
        img = _random_image(h=64, w=64)
        adapter.preprocess(img, img)

        # Model outputs at 1/4 res
        mock_out = {"flow": _mock_flow_output_chw(16, 16)}
        flow = adapter.postprocess(mock_out)
        assert flow.shape == (64, 64, 2)

    def test_full_resolution_no_upsample(self):
        adapter = DefaultAdapter(
            AdapterConfig(output_resolution="full", padding_factor=1)
        )
        img = _random_image(h=64, w=64)
        adapter.preprocess(img, img)

        mock_out = {"flow": _mock_flow_output_chw(64, 64)}
        flow = adapter.postprocess(mock_out)
        assert flow.shape == (64, 64, 2)

    def test_upsample_scales_flow_values(self):
        """Upsampled flow values should be multiplied by the scale factor."""
        adapter = DefaultAdapter(
            AdapterConfig(output_resolution="quarter", padding_factor=1)
        )
        img = _random_image(h=64, w=64)
        adapter.preprocess(img, img)

        raw = np.ones((1, 2, 16, 16), dtype=np.float32)  # constant flow=1 at 1/4
        flow = adapter.postprocess({"flow": raw})
        # After 4x upsample, values should be 4.0
        np.testing.assert_allclose(flow, 4.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Full round-trip: preprocess -> (mock inference) -> postprocess
# ═══════════════════════════════════════════════════════════════════════════════


class TestRoundTrip:
    def test_raft_like(self):
        """RAFT-style: separate inputs, no norm, pad 8, CHW output."""
        adapter = get_adapter("raft")
        img = _random_image(h=436, w=1024)
        feed = adapter.preprocess(img, img)

        assert "image1" in feed and "image2" in feed
        _, _, ph, pw = feed["image1"].shape
        assert ph % 8 == 0 and pw % 8 == 0

        mock_out = {"flow": _mock_flow_output_chw(ph, pw)}
        flow = adapter.postprocess(mock_out)
        assert flow.shape == (436, 1024, 2)

    def test_pwcnet_like(self):
        """PWC-Net style: concat input, unit norm, quarter-res output."""
        adapter = get_adapter("pwcnet")
        img = _random_image(h=384, w=512)
        feed = adapter.preprocess(img, img)

        assert len(feed) == 1
        tensor = list(feed.values())[0]
        assert tensor.shape[1] == 6  # concat
        assert tensor.max() <= 1.0  # unit norm

        _, _, ph, pw = tensor.shape
        # Model outputs at 1/4 res
        mock_out = {"flow": _mock_flow_output_chw(ph // 4, pw // 4)}
        flow = adapter.postprocess(mock_out)
        assert flow.shape == (384, 512, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegistry:
    def test_list_adapters(self):
        names = list_adapters()
        assert "raft" in names
        assert "pwcnet" in names

    def test_get_known_adapter(self):
        adapter = get_adapter("raft")
        assert isinstance(adapter, DefaultAdapter)

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown adapter"):
            get_adapter("nonexistent_model")

    def test_case_insensitive(self):
        a1 = get_adapter("RAFT")
        a2 = get_adapter("raft")
        assert type(a1) is type(a2)

    def test_overrides(self):
        adapter = get_adapter("raft", padding_factor=64)
        assert adapter.config.padding_factor == 64
        # Original registry entry should be unchanged
        original = get_adapter("raft")
        assert original.config.padding_factor == 8

    def test_register_config(self):
        register_adapter(
            "test_model",
            AdapterConfig(input_names=["a", "b"], normalization="unit"),
        )
        adapter = get_adapter("test_model")
        assert isinstance(adapter, DefaultAdapter)
        assert adapter.config.normalization == "unit"
        # Cleanup
        del ADAPTER_REGISTRY["test_model"]

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
