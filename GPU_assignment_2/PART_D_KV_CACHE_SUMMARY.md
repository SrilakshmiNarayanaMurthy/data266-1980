# HW4.5 Part D — KV-Cache Analysis

## Setup

- Model: facebook/opt-125m
- GPU: NVIDIA RTX 4090
- Precision: FP16
- Backend: Hugging Face Transformers
- Context lengths: 128, 512, 1024 tokens
- Generated lengths: 32 and 64 tokens
- Three measured runs per configuration

## Comparison

| Context | New tokens | Cache speedup | Memory difference (GiB) |
|---|---:|---:|---:|
| 128 | 32 | 0.976x | +0.001 |
| 128 | 64 | 0.988x | -0.002 |
| 512 | 32 | 0.703x | +0.015 |
| 512 | 64 | 1.225x | +0.011 |
| 1024 | 32 | 0.782x | +0.032 |
| 1024 | 64 | 1.362x | +0.029 |

## Findings

- KV caching was faster in 2 of 6 comparisons.
- Results were mixed, especially for shorter generation lengths.
- KV caching retains attention keys and values and may increase GPU memory usage.
- Small model size and short runs can make cache overhead and timing noise significant.
- These are Hugging Face generation results, not vLLM throughput measurements.

## Figures

- `figures/part_d_kv_throughput_32.png`
- `figures/part_d_kv_throughput_64.png`
- `figures/part_d_kv_cache_speedup.png`
