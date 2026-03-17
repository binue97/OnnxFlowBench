# OnnxFlowBench ⚡

Benchmark and run optical flow models through a single, unified ONNX pipeline.
Drop in a `.onnx` model, pick an adapter, and go — inference or full-dataset evaluation in one command.

---

## Setup

**Requires Python ≥ 3.10**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datasets

Dataloaders expect datasets under the `datasets/` directory with specific folder names.
Symlink your local copies — no need to move or duplicate data:

```bash
ln -s /path/to/your/MPI-Sintel      datasets/Sintel
ln -s /path/to/your/KITTI_2015       datasets/KITTI_2015
ln -s /path/to/your/FlyingChairs     datasets/FlyingChairs
ln -s /path/to/your/FlyingThings     datasets/FlyingThings
ln -s /path/to/your/Spring           datasets/Spring
ln -s /path/to/your/HD1K             datasets/HD1K
ln -s /path/to/your/TartanAir        datasets/TartanAir
```

Only link the datasets you plan to evaluate — you don't need all of them.

## Quick Start

### Single-pair inference

```bash
python infer.py --model raft.onnx --adapter raft \
    --img1 frame1.png --img2 frame2.png \
    --output results/ --png
```

Output formats: `--png` (color visualization), `--flo`, `--npy`.

### Dataset evaluation

```bash
python evaluate.py --model raft.onnx --adapter raft \
    --dataset sintel --dstype clean
```

Supported datasets: `sintel`, `kitti`, `chairs`, `things`, `spring`, `hd1k`, `tartanair`.

Metrics reported: EPE, Fl-all, 1px, 3px, 5px.

## How It Works

```
  img1, img2
      │
      ▼
┌─────────────┐
│ ModelAdapter │  ← pre/post processing (per-model)
│  preprocess  │
└──────┬──────┘
       ▼
┌─────────────┐
│  OnnxEngine  │  ← ONNX Runtime inference
└──────┬──────┘
       ▼
┌─────────────┐
│ ModelAdapter │
│ postprocess  │
└──────┬──────┘
       ▼
  (H, W, 2) flow
```

`FlowModel` ties it all together:

```python
from core.flow_model import FlowModel

model = FlowModel("raft.onnx", adapter="raft", device="cuda")
flow = model.predict(img1, img2)  # (H, W, 2) float32
```

## Adapters

Built-in adapters: **flownets**, **raft**.

Add your own:

```python
from core.registry import register_adapter
from core.base_adapter import ModelAdapter

class MyAdapter(ModelAdapter):
    def preprocess(self, img1, img2):
        ...
    def postprocess(self, outputs):
        ...

register_adapter("mymodel", MyAdapter)
```

## Tests

```bash
pytest tests/
```

## Acknowledgements

- **Dataloaders** adapted from [RAFT](https://github.com/princeton-vl/RAFT) and [WAFT](https://github.com/princeton-vl/WAFT) (Princeton Vision Lab).
- **Model adapters** built by referencing [RAFT](https://github.com/princeton-vl/RAFT) and [FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch).
- **Overall concept** inspired by [ptlflow](https://github.com/hmorimitsu/ptlflow).

## License

See [LICENSE](LICENSE).
