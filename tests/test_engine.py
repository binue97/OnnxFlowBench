"""
Tests for the OnnxEngine wrapper.

Uses a tiny dummy ONNX model (identity-like) generated at test time,
so no real model file is needed.

Usage:
    python -m pytest tests/test_engine.py -v
"""

import sys
import os
import pytest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── Skip entire module if onnxruntime is not installed ──────────────────────
ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
onnx = pytest.importorskip("onnx", reason="onnx not installed (needed to build test models)")

from onnx import helper, TensorProto
from engine.onnx_engine import OnnxEngine, TensorSpec


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures: build tiny ONNX models for testing
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def identity_model_path(tmp_path):
    """
    Creates a minimal ONNX model: output = input (identity).
    Input:  "input"  float32 (1, 3, 64, 64)
    Output: "output" float32 (1, 3, 64, 64)
    """
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 64, 64])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 64, 64])
    node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
    graph = helper.make_graph([node], "identity_graph", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    path = str(tmp_path / "identity.onnx")
    onnx.save(model, path)
    return path


@pytest.fixture
def two_input_model_path(tmp_path):
    """
    Creates an ONNX model that adds two inputs: output = image1 + image2.
    Input:  "image1" float32 (1, 3, H, W)  — dynamic H, W
    Input:  "image2" float32 (1, 3, H, W)
    Output: "flow"   float32 (1, 3, H, W)
    """
    X1 = helper.make_tensor_value_info("image1", TensorProto.FLOAT, [1, 3, "H", "W"])
    X2 = helper.make_tensor_value_info("image2", TensorProto.FLOAT, [1, 3, "H", "W"])
    Y = helper.make_tensor_value_info("flow", TensorProto.FLOAT, [1, 3, "H", "W"])
    node = helper.make_node("Add", inputs=["image1", "image2"], outputs=["flow"])
    graph = helper.make_graph([node], "add_graph", [X1, X2], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    path = str(tmp_path / "two_input.onnx")
    onnx.save(model, path)
    return path


@pytest.fixture
def multi_output_model_path(tmp_path):
    """
    Creates an ONNX model with two outputs: out1 = input, out2 = input.
    """
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 64, 64])
    Y1 = helper.make_tensor_value_info("out1", TensorProto.FLOAT, [1, 2, 64, 64])
    Y2 = helper.make_tensor_value_info("out2", TensorProto.FLOAT, [1, 2, 64, 64])
    node1 = helper.make_node("Identity", inputs=["input"], outputs=["out1"])
    node2 = helper.make_node("Identity", inputs=["input"], outputs=["out2"])
    graph = helper.make_graph([node1, node2], "multi_out_graph", [X], [Y1, Y2])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    path = str(tmp_path / "multi_output.onnx")
    onnx.save(model, path)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Construction & introspection
# ═══════════════════════════════════════════════════════════════════════════════


class TestConstruction:
    def test_loads_model(self, identity_model_path):
        engine = OnnxEngine(identity_model_path, device="cpu")
        assert engine.session is not None

    def test_invalid_path_raises(self, tmp_path):
        with pytest.raises(Exception):
            OnnxEngine(str(tmp_path / "nonexistent.onnx"), device="cpu")

    def test_invalid_device_raises(self, identity_model_path):
        with pytest.raises(ValueError, match="Unknown device"):
            OnnxEngine(identity_model_path, device="tpu")

    def test_repr(self, identity_model_path):
        engine = OnnxEngine(identity_model_path, device="cpu")
        r = repr(engine)
        assert "OnnxEngine" in r
        assert "input" in r
        assert "output" in r


class TestIntrospection:
    def test_input_specs(self, identity_model_path):
        engine = OnnxEngine(identity_model_path, device="cpu")
        assert len(engine.input_specs) == 1
        spec = engine.input_specs[0]
        assert isinstance(spec, TensorSpec)
        assert spec.name == "input"
        assert spec.dtype == "float32"
        assert spec.shape == [1, 3, 64, 64]

    def test_output_specs(self, identity_model_path):
        engine = OnnxEngine(identity_model_path, device="cpu")
        assert len(engine.output_specs) == 1
        spec = engine.output_specs[0]
        assert spec.name == "output"
        assert spec.shape == [1, 3, 64, 64]

    def test_input_names(self, identity_model_path):
        engine = OnnxEngine(identity_model_path, device="cpu")
        assert engine.input_names == ["input"]

    def test_output_names(self, identity_model_path):
        engine = OnnxEngine(identity_model_path, device="cpu")
        assert engine.output_names == ["output"]

    def test_two_input_specs(self, two_input_model_path):
        engine = OnnxEngine(two_input_model_path, device="cpu")
        assert len(engine.input_specs) == 2
        assert engine.input_names == ["image1", "image2"]

    def test_dynamic_dims_as_strings(self, two_input_model_path):
        engine = OnnxEngine(two_input_model_path, device="cpu")
        spec = engine.input_specs[0]
        # Dynamic dims should be strings
        assert spec.shape[0] == 1
        assert spec.shape[1] == 3
        assert isinstance(spec.shape[2], str)  # "H"
        assert isinstance(spec.shape[3], str)  # "W"

    def test_multi_output_specs(self, multi_output_model_path):
        engine = OnnxEngine(multi_output_model_path, device="cpu")
        assert len(engine.output_specs) == 2
        assert engine.output_names == ["out1", "out2"]


# ═══════════════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════════════


class TestInference:
    def test_identity_model(self, identity_model_path):
        engine = OnnxEngine(identity_model_path, device="cpu")
        x = np.random.randn(1, 3, 64, 64).astype(np.float32)
        outputs = engine({"input": x})
        assert "output" in outputs
        np.testing.assert_array_equal(outputs["output"], x)

    def test_two_input_add(self, two_input_model_path):
        engine = OnnxEngine(two_input_model_path, device="cpu")
        a = np.ones((1, 3, 32, 32), dtype=np.float32) * 3.0
        b = np.ones((1, 3, 32, 32), dtype=np.float32) * 7.0
        outputs = engine({"image1": a, "image2": b})
        assert "flow" in outputs
        np.testing.assert_allclose(outputs["flow"], 10.0)

    def test_multi_output(self, multi_output_model_path):
        engine = OnnxEngine(multi_output_model_path, device="cpu")
        x = np.random.randn(1, 2, 64, 64).astype(np.float32)
        outputs = engine({"input": x})
        assert len(outputs) == 2
        np.testing.assert_array_equal(outputs["out1"], x)
        np.testing.assert_array_equal(outputs["out2"], x)

    def test_dynamic_shapes(self, two_input_model_path):
        """Model with dynamic H, W should accept different spatial sizes."""
        engine = OnnxEngine(two_input_model_path, device="cpu")
        for h, w in [(32, 32), (64, 128), (100, 50)]:
            a = np.zeros((1, 3, h, w), dtype=np.float32)
            b = np.ones((1, 3, h, w), dtype=np.float32)
            outputs = engine({"image1": a, "image2": b})
            assert outputs["flow"].shape == (1, 3, h, w)

    def test_output_dtype(self, identity_model_path):
        engine = OnnxEngine(identity_model_path, device="cpu")
        x = np.zeros((1, 3, 64, 64), dtype=np.float32)
        outputs = engine({"input": x})
        assert outputs["output"].dtype == np.float32

    def test_missing_input_raises(self, two_input_model_path):
        engine = OnnxEngine(two_input_model_path, device="cpu")
        a = np.zeros((1, 3, 32, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="Missing inputs"):
            engine({"image1": a})  # missing "image2"

    def test_returns_dict(self, identity_model_path):
        engine = OnnxEngine(identity_model_path, device="cpu")
        x = np.zeros((1, 3, 64, 64), dtype=np.float32)
        outputs = engine({"input": x})
        assert isinstance(outputs, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Extra inputs (ignored gracefully by ONNX Runtime)
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtraInputs:
    def test_extra_keys_are_filtered(self, identity_model_path):
        """Extra keys in the input dict should be silently filtered out."""
        engine = OnnxEngine(identity_model_path, device="cpu")
        x = np.zeros((1, 3, 64, 64), dtype=np.float32)
        extra = np.zeros((1,), dtype=np.float32)
        # Should not raise — engine filters to only expected input names
        outputs = engine({"input": x, "unused_tensor": extra})
        assert "output" in outputs
        np.testing.assert_array_equal(outputs["output"], x)
