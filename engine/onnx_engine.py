"""
Thin ONNX Runtime wrapper — session lifecycle, device placement, and raw inference.

This class only manages the ONNX session and runs inference.
"""

from dataclasses import dataclass
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError(
        "onnxruntime is required. Install with: "
        "pip install onnxruntime  (CPU) or  pip install onnxruntime-gpu  (CUDA)"
    )


@dataclass
class TensorSpec:
    name: str
    shape: list  # may contain strings for dynamic dims, e.g. ["batch", 3, "height", "width"]
    dtype: str  # numpy dtype string, e.g. "float32"


class OnnxEngine:
    """
    Thin ONNX Runtime wrapper.

    Responsibilities:
        - Load an ONNX model and create an inference session
        - Select execution provider (CPU / CUDA)
        - Expose input/output specs via introspection
        - Run inference: dict[str, ndarray] → dict[str, ndarray]
    """

    # Supported providers in priority order
    _PROVIDER_MAP = {
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "cpu": ["CPUExecutionProvider"],
    }

    def __init__(self, onnx_path: str, device: str = "cuda"):
        """
        Args:
            onnx_path: Path to the .onnx model file.
            device: "cuda" or "cpu".
        """
        self.onnx_path = onnx_path
        self.device = device.lower()

        if self.device not in self._PROVIDER_MAP:
            raise ValueError(
                f"Unknown device '{device}'. Supported: {list(self._PROVIDER_MAP.keys())}"
            )

        providers = self._PROVIDER_MAP[self.device]
        available = ort.get_available_providers()
        providers = [p for p in providers if p in available]
        if not providers:
            raise RuntimeError(
                f"No suitable ONNX Runtime provider for device='{device}'. "
                f"Available providers: {available}"
            )

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.session = ort.InferenceSession(
            onnx_path, sess_options=sess_options, providers=providers
        )

        # Cache specs at init so introspection is free
        self._input_specs = self._build_input_specs()
        self._output_specs = self._build_output_specs()

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def input_specs(self) -> list[TensorSpec]:
        """Input tensor specifications (name, shape, dtype)."""
        return self._input_specs

    @property
    def output_specs(self) -> list[TensorSpec]:
        """Output tensor specifications (name, shape, dtype)."""
        return self._output_specs

    @property
    def input_names(self) -> list[str]:
        """List of input tensor names."""
        return [s.name for s in self._input_specs]

    @property
    def output_names(self) -> list[str]:
        """List of output tensor names."""
        return [s.name for s in self._output_specs]

    def _build_input_specs(self) -> list[TensorSpec]:
        specs = []
        for inp in self.session.get_inputs():
            dtype = self._onnx_dtype_to_numpy(inp.type)
            shape = [d if isinstance(d, int) else str(d) for d in inp.shape]
            specs.append(TensorSpec(name=inp.name, shape=shape, dtype=dtype))
        return specs

    def _build_output_specs(self) -> list[TensorSpec]:
        specs = []
        for out in self.session.get_outputs():
            dtype = self._onnx_dtype_to_numpy(out.type)
            shape = [d if isinstance(d, int) else str(d) for d in out.shape]
            specs.append(TensorSpec(name=out.name, shape=shape, dtype=dtype))
        return specs

    @staticmethod
    def _onnx_dtype_to_numpy(onnx_type: str) -> str:
        """Convert ONNX type string like 'tensor(float)' to numpy dtype string."""
        mapping = {
            "tensor(float)": "float32",
            "tensor(float16)": "float16",
            "tensor(double)": "float64",
            "tensor(int32)": "int32",
            "tensor(int64)": "int64",
            "tensor(int8)": "int8",
            "tensor(uint8)": "uint8",
            "tensor(bool)": "bool",
        }
        return mapping.get(onnx_type, "float32")

    # ── Inference ─────────────────────────────────────────────────────────────

    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Run inference.

        Args:
            inputs: Dict mapping input tensor names → numpy arrays.

        Returns:
            Dict mapping output tensor names → numpy arrays.

        Raises:
            ValueError: If required input names are missing.
        """
        # Validate input names
        missing = set(self.input_names) - set(inputs.keys())
        if missing:
            raise ValueError(
                f"Missing inputs: {missing}. Model expects: {self.input_names}"
            )

        # Only pass expected inputs — ONNX Runtime rejects unknown keys
        feed = {name: inputs[name] for name in self.input_names}
        raw_outputs = self.session.run(self.output_names, feed)
        return dict(zip(self.output_names, raw_outputs))

    # ── Representation ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        inputs_str = ", ".join(
            f"{s.name}: {s.dtype}{s.shape}" for s in self._input_specs
        )
        outputs_str = ", ".join(
            f"{s.name}: {s.dtype}{s.shape}" for s in self._output_specs
        )
        return (
            f"OnnxEngine(\n"
            f"  path={self.onnx_path!r},\n"
            f"  device={self.device!r},\n"
            f"  inputs=[{inputs_str}],\n"
            f"  outputs=[{outputs_str}]\n"
            f")"
        )
