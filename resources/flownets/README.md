Model: FlowNetS
Github Repo: https://github.com/ClementPinard/FlowNetPytorch
License: MIT

[Test environments]
- Date: 2026.03.11
- PyTorch version: 2.1.2
- ONNX runtime version: 1.24.2

[ONNX export options]
- Pretrained model: flownets_EPE1.951.pth
- Constant folding: True
- Opset version: 17
- Dynamic axes
  - input: {0: 'batch', 2: 'height', 3: 'width'}
  - output: {0: 'batch', 2: 'height', 3: 'width'}
