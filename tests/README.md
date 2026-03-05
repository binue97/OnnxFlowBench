# OnnxFlowBench Dataloader Tests

## Prerequisites
- `pytest` installed (`pip install pytest`)
- Real datasets placed under `datasets/` at the project root:

```
datasets/
├── FlyingChairs/
├── FlyingThings/
├── HD1K/
├── KITTI_2015/
├── Sintel/
├── Spring/
└── TartanAir/
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/test_dataloader.py -v

# Run tests for a specific dataset
python -m pytest tests/test_dataloader.py -v -k "Sintel"
python -m pytest tests/test_dataloader.py -v -k "KITTI"
python -m pytest tests/test_dataloader.py -v -k "Spring"

# Run only a single test
python -m pytest tests/test_dataloader.py -v -k "test_getitem_returns_tensors"

# Stop on first failure
python -m pytest tests/test_dataloader.py -v -x
```

## Test Structure

| Test Class | Dataset | What it checks |
|---|---|---|
| `TestFlowDatasetBase` | - | Base `FlowDataset`: empty length, `__rmul__` repetition |
| `TestMpiSintel` | Sintel | Discovery, clean/final dstypes, `__getitem__`, `extra_info` |
| `TestFlyingChairs` | FlyingChairs | Training/validation split, `__getitem__` |
| `TestKITTI` | KITTI 2015 | Discovery, `extra_info` frame IDs, `__getitem__` |
| `TestFlyingThings3D` | FlyingThings | File discovery, `__getitem__` |
| `TestSpring` | Spring | Discovery, invalid/missing split errors, `__getitem__` |
| `TestHD1K` | HD1K | Discovery, `__getitem__` |
| `TestTartanAir` | TartanAir | Discovery (images, flows, masks), `__getitem__` |
| `TestDataLoaderIntegration` | Sintel, KITTI | Batching, dataset concat (`+`), repetition (`*`) |
| `TestOutputShapesAndTypes` | Sintel | Image range [0,255], binary valid mask, no NaN/Inf in flow |

## What Each Test Category Verifies

### Discovery tests
- Dataset finds all image pairs and flow files on disk
- `len(image_list) == len(flow_list)`
- Correct flags (e.g. `is_test`, `extra_info`)

### `__getitem__` tests
- Returns `(img1, img2, flow, valid)` as `torch.Tensor`
- `img1`, `img2`: shape `(3, H, W)`, dtype `float32`
- `flow`: shape `(2, H, W)`, dtype `float32`
- `valid`: shape `(H, W)`, dtype `float32`, values in `{0, 1}`

### DataLoader integration tests
- Batching produces correct batch dimension
- `dataset1 + dataset2` concatenation works
- `N * dataset` repetition multiplies length correctly

### Output quality tests
- Pixel values in `[0, 255]`
- Valid mask is strictly binary
- Flow contains no `NaN` or `Inf`

## Adding a New Dataset Test

1. Add a new test class following the existing pattern:

```python
class TestMyDataset:
    def test_discovery(self):
        ds = MyDataset()
        assert len(ds) > 0
        assert len(ds.image_list) == len(ds.flow_list)

    def test_getitem(self):
        ds = MyDataset()
        img1, img2, flow, valid = ds[0]
        assert img1.shape[0] == 3
        assert flow.shape[0] == 2
```

2. Place the dataset under `datasets/` and import the class at the top of the test file.