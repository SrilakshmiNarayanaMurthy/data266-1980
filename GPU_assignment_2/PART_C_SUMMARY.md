# HW4.5 Part C — Distributed Data Parallel

## Hardware and Environment

- GPU: NVIDIA GeForce RTX 4090
- Available GPU count: 1
- PyTorch distributed: Available
- NCCL backend: Available
- Two-GPU allocation: Not available

## Single-GPU DDP Functional Test

| Metric | Result |
|---|---|
| Backend | NCCL |
| World size | 1 |
| Training examples | 128 |
| Optimizer steps | 2 |
| Training time | 0.528 seconds |
| Throughput | 242.62 examples/sec |
| Peak allocated memory | 11.11 GiB |
| DDP training | Successful |

The model was wrapped using PyTorch
`DistributedDataParallel` and trained successfully
with one process on GPU 0.

The NCCL process group was initialized and destroyed
successfully.

## Multi-GPU Scaling

Two-GPU scaling was not measured because only one GPU
was available.

The following measurements are therefore unavailable:

- Two-GPU training throughput
- Two-GPU speedup
- Scaling efficiency
- Inter-GPU communication overhead

These values are intentionally not estimated or
fabricated.

## Interpretation

The experiment validates single-process DDP setup,
model wrapping, forward/backward execution, and
optimizer updates.

It does not demonstrate distributed acceleration
or multi-GPU gradient synchronization.

## Saved Evidence

- `metrics/part_c_ddp_readiness.json`
- `metrics/part_c_single_gpu_ddp.json`
- `metrics/part_c_summary.json`
