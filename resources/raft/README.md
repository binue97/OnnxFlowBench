Model: RAFT
Github Repo: https://github.com/princeton-vl/RAFT
License: BSD-3-Clause

[Test environments]
- Date: 2026.03.13
- PyTorch version: 2.1.2
- ONNX runtime version: 1.24.2

[ONNX export options]
- Pretrained model: raft-sintel.pth
- Constant folding: True
- Opset version: 17
- Dynamic axes
  - image1: {0: 'batch', 2: 'height', 3: 'width'}
  - image2: {0: 'batch', 2: 'height', 3: 'width'}
  - flow_low: {0: 'batch', 2: 'height', 3: 'width'}
  - flow_up: {0: 'batch', 2: 'height', 3: 'width'}
