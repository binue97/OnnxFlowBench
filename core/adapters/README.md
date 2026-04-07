# Adding a New Adapter

## 1. Create the adapter file

Add `core/adapters/mymodel_adapter.py`:

```python
import numpy as np

from core.base_adapter import ModelAdapter
from core import adapter_utils as utils


class MyModelAdapter(ModelAdapter):
    def preprocess(self, img1: np.ndarray, img2: np.ndarray) -> dict[str, np.ndarray]:
        # Transform (H,W,3) uint8 images into ONNX input feed
        ...
        return {"input_name": tensor}

    def postprocess(self, outputs: dict[str, np.ndarray]) -> np.ndarray:
        # Transform ONNX outputs into (H,W,2) float32 flow
        ...
        return flow
```

See `raft_adapter.py` for a full example. Helper functions are in `core/adapter_utils`.

## 2. Export it from `__init__.py`

In `core/adapters/__init__.py`, add:

```python
from core.adapters.mymodel_adapter import MyModelAdapter
```

## 3. Register it in the registry

In `core/registry.py`, add your class to `ADAPTER_REGISTRY`:

```python
ADAPTER_REGISTRY: dict[str, type[ModelAdapter]] = {
    ...
    "mymodel": MyModelAdapter,
}
```

and add it to the import at the top of that file.

## Done

Now you can use it:

```python
model = FlowModel("mymodel.onnx", adapter="mymodel")
flow = model.predict(img1, img2)
```
