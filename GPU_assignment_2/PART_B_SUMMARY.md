# HW4.5 Part B — GPU Optimization Summary

## Experimental Setup

- GPU: NVIDIA GeForce RTX 4090 (23.99 GiB)
- Model: BERT sentiment classifier
- Training examples: 3,500
- Physical batch size: 64
- Training duration: one epoch per experiment
- Seed: 1980
- Starting checkpoint: `models/bert_best`

## Optimization Results

| Configuration | Epoch (s) | Throughput (ex/s) | Peak VRAM (GiB) | Speedup | Memory reduction (%) |
| --- | --- | --- | --- | --- | --- |
| FP32 Baseline | 7.522 | 465.31 | 10.235 | 1.000x | 0.00% |
| BF16 AMP | 4.555 | 768.43 | 8.431 | 1.651x | 17.63% |
| Checkpointing | 10.166 | 344.27 | 5.176 | 0.740x | 49.43% |
| Accumulation x4 | 7.328 | 477.62 | 10.661 | 1.026x | -4.16% |
| 8-bit AdamW | 7.767 | 450.64 | 10.037 | 0.968x | 1.93% |
| Combined | 5.009 | 698.76 | 5.254 | 1.502x | 48.66% |

## Key Findings

- **Fastest:** BF16 AMP at 768.43 examples/sec.
- **Lowest peak allocated memory:** Checkpointing at 5.18 GiB.
- **Combined optimization:** 1.50x speedup with 48.66% less peak allocated memory than FP32.
- Gradient accumulation and the combined configuration use effective batch size 256 and 14 optimizer updates; the other experiments use effective batch size 64.
- Results are single-run measurements. Small performance differences require repeated measurements before drawing firm conclusions.
- Reported GPU memory is peak PyTorch CUDA allocated memory, not total device memory.

## Figures

- `figures/part_b_throughput_comparison.png`
- `figures/part_b_memory_comparison.png`
- `figures/part_b_epoch_time_comparison.png`

## Software

- PyTorch: 2.1.2
- bitsandbytes: 0.43.1
- AMP precision: BF16
- Optimizers: AdamW and AdamW8bit
