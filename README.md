
valid (mask) : 0 for invalid pixel, 1 for valid pixel

supported evaluation metrics: epe, fl_all, 1px, 3px, 5px

## Single-Pair Inference

Run `infer.py` to predict optical flow from two images using an ONNX model.

### Arguments

| Argument      | Required | Default   | Description                              |
|---------------|----------|-----------|------------------------------------------|
| `--model`     | Yes      |           | Path to `.onnx` model                    |
| `--img1`      | Yes      |           | Path to first image                      |
| `--img2`      | Yes      |           | Path to second image                     |
| `--adapter`   | No       | `raft`    | Adapter name (raft, gma, flowformer, …)  |
| `--device`    | No       | `cuda`    | `cuda` or `cpu`                          |
| `--output`    | No       | `results` | Output directory                         |
| `--save-png`  | No       |           | Save color visualization (`.png`)        |
| `--save-flo`  | No       |           | Save Middlebury `.flo` format            |
| `--save-npy`  | No       |           | Save raw NumPy `.npy`                    |

### Examples

```bash
# Basic inference with PNG output
python infer.py --model raft.onnx --img1 frame1.png --img2 frame2.png --save-png

# Save all formats
python infer.py --model raft.onnx --img1 frame1.png --img2 frame2.png \
    --output results/ --save-png --save-flo --save-npy

# Use a different adapter on CPU
python infer.py --model flowformer.onnx --adapter flowformer --device cpu \
    --img1 frame1.png --img2 frame2.png --save-flo
```

