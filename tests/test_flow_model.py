"""
Tests for FlowModel - the user-facing entry point.

Uses a dummy ONNX model to form an end-to-end test:
    preprocess -> ONNX inference -> postprocess -> (H, W, 2) flow

Usage:
    python -m pytest tests/test_flow_model.py -v
"""

import sys
import os
import pytest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
onnx = pytest.importorskip("onnx", reason="onnx not installed")

from onnx import helper, TensorProto

from core.flow_model import FlowModel
from core.base_adapter import ModelAdapter
from core.adapter_config import AdapterConfig
from core.default_adapter import DefaultAdapter


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures: dummy ONNX models
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def raft_like_model_path(tmp_path):
    """
    Simulates a RAFT-like model:
        Inputs:  image1 (1, 3, H, W), image2 (1, 3, H, W)  float32
        Output:  flow   (1, 2, H, W)  float32
    Uses Sub (image1 - image2) sliced to 2 channels as a dummy "flow".

    Since we can't easily slice in ONNX with helpers, we'll use an Identity
    on image1 and then rely on the adapter to handle the shape. Instead,
    let's make a model that takes two 2-channel inputs and returns their diff.
    """
    # Simpler: model takes (1,2,H,W) x2 and returns Sub
    X1 = helper.make_tensor_value_info("image1", TensorProto.FLOAT, [1, 2, "H", "W"])
    X2 = helper.make_tensor_value_info("image2", TensorProto.FLOAT, [1, 2, "H", "W"])
    Y = helper.make_tensor_value_info("flow", TensorProto.FLOAT, [1, 2, "H", "W"])
    node = helper.make_node("Sub", inputs=["image1", "image2"], outputs=["flow"])
    graph = helper.make_graph([node], "raft_like", [X1, X2], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    path = str(tmp_path / "raft_like.onnx")
    onnx.save(model, path)
    return path


@pytest.fixture
def identity_2ch_model_path(tmp_path):
    """
    Identity model: input (1, 2, H, W) -> output (1, 2, H, W).
    Useful for testing that preprocess/postprocess are inverses.
    """
    X = helper.make_tensor_value_info("image1", TensorProto.FLOAT, [1, 2, "H", "W"])
    Y = helper.make_tensor_value_info("flow", TensorProto.FLOAT, [1, 2, "H", "W"])
    node = helper.make_node("Identity", inputs=["image1"], outputs=["flow"])
    graph = helper.make_graph([node], "identity_2ch", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    path = str(tmp_path / "identity_2ch.onnx")
    onnx.save(model, path)
    return path


@pytest.fixture
def add_model_path(tmp_path):
    """
    Add model: image1 + image2 -> flow.
    Both inputs and output are (1, 3, H, W).
    """
    X1 = helper.make_tensor_value_info("image1", TensorProto.FLOAT, [1, 3, "H", "W"])
    X2 = helper.make_tensor_value_info("image2", TensorProto.FLOAT, [1, 3, "H", "W"])
    Y = helper.make_tensor_value_info("flow", TensorProto.FLOAT, [1, 3, "H", "W"])
    node = helper.make_node("Add", inputs=["image1", "image2"], outputs=["flow"])
    graph = helper.make_graph([node], "add_model", [X1, X2], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    path = str(tmp_path / "add_model.onnx")
    onnx.save(model, path)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Custom test adapter (bypasses real preprocessing for controlled tests)
# ═══════════════════════════════════════════════════════════════════════════════


class PassthroughAdapter(ModelAdapter):
    """
    Minimal adapter for testing: passes images through with minimal transforms.
    Expects a model with input "image1" of shape (1, 2, H, W).
    """

    def __init__(self):
        self._h = 0
        self._w = 0

    def preprocess(self, img1, img2):
        self._h, self._w = img1.shape[:2]
        # Take first 2 channels, HWC->CHW, add batch
        t = img1[:, :, :2].astype(np.float32)
        t = np.transpose(t, (2, 0, 1))[np.newaxis]
        return {"image1": t}

    def postprocess(self, outputs):
        flow = outputs["flow"]
        if flow.ndim == 4:
            flow = flow[0]  # remove batch
        # CHW -> HWC
        return np.transpose(flow, (1, 2, 0))


# ═══════════════════════════════════════════════════════════════════════════════
# Construction
# ═══════════════════════════════════════════════════════════════════════════════


class TestConstruction:
    def test_with_adapter_name(self, add_model_path):
        model = FlowModel(add_model_path, adapter="raft", device="cpu")
        assert isinstance(model.adapter, DefaultAdapter)

    def test_with_adapter_config(self, add_model_path):
        cfg = AdapterConfig(normalization="unit", padding_factor=16)
        model = FlowModel(add_model_path, adapter=cfg, device="cpu")
        assert isinstance(model.adapter, DefaultAdapter)
        assert model.adapter.config.normalization == "unit"

    def test_with_adapter_instance(self, identity_2ch_model_path):
        adapter = PassthroughAdapter()
        model = FlowModel(identity_2ch_model_path, adapter=adapter, device="cpu")
        assert model.adapter is adapter

    def test_invalid_adapter_type_raises(self, add_model_path):
        with pytest.raises(TypeError, match="adapter must be"):
            FlowModel(add_model_path, adapter=12345, device="cpu")

    def test_repr(self, add_model_path):
        model = FlowModel(add_model_path, adapter="raft", device="cpu")
        r = repr(model)
        assert "FlowModel" in r
        assert "DefaultAdapter" in r


# ═══════════════════════════════════════════════════════════════════════════════
# Predict - end-to-end with custom adapter
# ═══════════════════════════════════════════════════════════════════════════════


class TestPredictCustomAdapter:
    def test_identity_returns_input(self, identity_2ch_model_path):
        """Identity ONNX model + PassthroughAdapter -> output equals input[:,:,:2]."""
        adapter = PassthroughAdapter()
        model = FlowModel(identity_2ch_model_path, adapter=adapter, device="cpu")

        img = np.full((64, 64, 3), 42, dtype=np.uint8)
        flow = model.predict(img, img)

        assert flow.shape == (64, 64, 2)
        assert flow.dtype == np.float32
        np.testing.assert_allclose(flow, 42.0)

    def test_output_shape_matches_input(self, identity_2ch_model_path):
        """Flow spatial dims should match input image."""
        adapter = PassthroughAdapter()
        model = FlowModel(identity_2ch_model_path, adapter=adapter, device="cpu")

        for h, w in [(32, 48), (100, 200), (17, 31)]:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            flow = model.predict(img, img)
            assert flow.shape == (h, w, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# Predict - end-to-end with DefaultAdapter
# ═══════════════════════════════════════════════════════════════════════════════


class TestPredictDefaultAdapter:
    def test_add_model_with_default_adapter(self, add_model_path):
        """
        Add model (image1 + image2) with DefaultAdapter.
        Output has 3 channels but we configure output_layout to handle it.
        """
        # This model outputs (1, 3, H, W) - not real flow, but tests the pipeline.
        # We just verify no crashes and correct spatial dims.
        cfg = AdapterConfig(
            input_names=["image1", "image2"],
            normalization="none",
            padding_factor=8,
            output_name="flow",
            # The model outputs 3 channels, not 2.
            # We'll test the pipeline doesn't crash - the CHW->HWC will produce (H,W,3).
        )
        model = FlowModel(add_model_path, adapter=cfg, device="cpu")
        img = np.full((64, 64, 3), 10, dtype=np.uint8)
        flow = model.predict(img, img)
        # Output is (H, W, 3) because the model has 3 output channels
        assert flow.shape[0] == 64
        assert flow.shape[1] == 64

    def test_padding_and_unpadding_roundtrip(self, add_model_path):
        """Non-divisible input size -> padded -> inferred -> unpadded back."""
        cfg = AdapterConfig(
            input_names=["image1", "image2"],
            normalization="none",
            padding_factor=32,
            output_name="flow",
        )
        model = FlowModel(add_model_path, adapter=cfg, device="cpu")
        img = np.zeros((100, 100, 3), dtype=np.uint8)  # not divisible by 32
        flow = model.predict(img, img)
        assert flow.shape[0] == 100
        assert flow.shape[1] == 100

    def test_unit_normalization_e2e(self, add_model_path):
        """Unit normalization: 255 -> 1.0, so Add gives 2.0."""
        cfg = AdapterConfig(
            input_names=["image1", "image2"],
            normalization="unit",
            padding_factor=1,
            output_name="flow",
        )
        model = FlowModel(add_model_path, adapter=cfg, device="cpu")
        img = np.full((8, 8, 3), 255, dtype=np.uint8)
        flow = model.predict(img, img)
        # 255/255 = 1.0, Add -> 2.0
        np.testing.assert_allclose(flow, 2.0, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# Predict - error handling
# ═══════════════════════════════════════════════════════════════════════════════


class TestPredictErrors:
    def test_invalid_onnx_path(self, tmp_path):
        with pytest.raises(Exception):
            FlowModel(str(tmp_path / "nonexistent.onnx"), adapter="raft", device="cpu")
