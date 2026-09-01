# HW1 Metrics

## Student-Specific Configuration

- SID4: 1980
- SEED: 1980
- SLICE: 980
- HP_ID: 0
- CLS_A: 0
- CLS_B: 1

## Q2 — Neural Network Results

### PyTorch

| Model | Mean Test Accuracy | Std Dev |
|---|---:|---:|
| Baseline [64, 32] | 0.6491 | 0.0000 |
| Modified [32] | 0.6667 | 0.0248 |

Training seeds used:

- 1980
- 1981
- 1982

### TensorFlow

| Model | Mean Test Accuracy | Std Dev |
|---|---:|---:|
| Baseline [64, 32] | 0.7807 | 0.0072 |
| Modified [32] | 0.7982 | 0.0000 |

Training seeds used:

- 1980
- 1981
- 1982

### Modified Model Configuration

HP_ID = 0

- Hidden layers: [32]
- Learning rate: 0.001
- Epochs: 30

## Q3 — CUDA Matrix Multiplication Results

| Matrix Size | CPU Time (ms) | GPU Kernel Time (ms) | H2D + D2H (ms) | GPU End-to-End (ms) | Speedup |
|---|---:|---:|---:|---:|---:|
| 256 x 256 | 53.5137 | 0.133152 | 0.247936 | 0.381088 | 140.424x |
| 1024 x 1024 | 3491.75 | 9.21133 | 2.90189 | 12.1132 | 288.260x |
| 4096 x 4096 | 248490.0 | 417.659 | 45.8121 | 463.471 | 536.150x |

Profiler used:

- NVIDIA Nsight Compute
- NVIDIA nvprof

The smallest tested matrix size where GPU execution was beneficial end-to-end was 256 x 256.