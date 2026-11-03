# HW4.5 Part A — RTX 4090 Performance Summary

## Experiment
- Model: BERT-base-uncased sentiment classifier
- Training examples: 3,500
- Sequence length: 128
- Precision: FP32
- Gradient accumulation: 1

## Full-Epoch Comparison

| Batch | Epoch (s) | Examples/s | Peak GiB | Speedup vs 16 |
|---:|---:|---:|---:|---:|
| 16 | 10.14 | 345.26 | 5.67 | 1.00x |
| 32 | 8.69 | 402.64 | 7.23 | 1.17x |
| 64 | 7.52 | 465.31 | 10.23 | 1.35x |
| 128 | 7.92 | 441.85 | 16.34 | 1.28x |

## Findings
- Best full-epoch batch size: 64
- Best throughput: 465.31 examples/sec
- Best epoch time: 7.52 seconds
- Throughput speedup vs batch 16: 1.35x

## Memory Limitation
- Physical VRAM: 23.99 GiB
- Above-VRAM reported allocations: [256, 512]
- No CUDA OOM was observed in the completed microbenchmark.
- The mechanism behind above-VRAM allocations remains unverified.

## Scope
- Microbenchmark: 10 measured training steps per batch.
- Full-epoch comparison: one epoch per batch size.
- These results do not constitute three-epoch comparisons.
