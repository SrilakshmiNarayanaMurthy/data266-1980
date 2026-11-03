# HW4.5 GPU Assignment II — METRICS

## Experimental environment

- GPU: NVIDIA GeForce RTX 4090
- GPU count: 1
- Seed: 1980
- BERT model: bert-base-uncased
- Causal LM for Part D: facebook/opt-125m

## HW4.a baseline

- Training duration: 36.662 seconds
- Training precision: FP32
- Training batch size: 16
- Epochs: 3
- Peak allocated memory: 2.796 GiB

## Part A — Batch scaling

| Batch | Epoch seconds | Examples/s | Peak allocated GiB |
|---:|---:|---:|---:|
| 16 | 10.137 | 345.26 | 5.672 |
| 32 | 8.693 | 402.64 | 7.232 |
| 64 | 7.522 | 465.31 | 10.235 |
| 128 | 7.921 | 441.85 | 16.341 |

Fastest full-epoch batch size: **64**.

The ten-step batch-scaling experiment is separately documented in PART_A_SUMMARY.md.

## Part B — Training optimizations

| Configuration | Seconds | Examples/s | Peak GiB | Speedup |
|---|---:|---:|---:|---:|
| FP32 Baseline | 7.522 | 465.31 | 10.235 | 1.00x |
| BF16 AMP | 4.555 | 768.43 | 8.431 | 1.65x |
| Checkpointing | 10.166 | 344.27 | 5.176 | 0.74x |
| Accumulation x4 | 7.328 | 477.62 | 10.661 | 1.03x |
| 8-bit AdamW | 7.767 | 450.64 | 10.037 | 0.97x |
| Combined | 5.009 | 698.76 | 5.254 | 1.50x |

- Fastest: **BF16 AMP** (768.43 examples/s).
- Lowest peak allocated memory: **Checkpointing** (5.176 GiB).
- The Part B FP32 baseline reuses the Part A batch-64 full-epoch measurement.

## Part C — Distributed training

- Backend: nccl
- World size: 1
- Measured DDP duration: 0.527574 seconds
- Functional DDP: True
- Multi-GPU scaling measured: False
- Communication overhead measured: False

Only one RTX 4090 was available; no multi-GPU speedup or communication-overhead measurement is claimed.

## Part D — KV-cache and inference serving

- Model: facebook/opt-125m
- Precision: FP16
- KV-cache on/off: 12 configurations
- Context lengths: 128, 512, 1024 tokens
- Generation lengths: 32, 64 tokens
- KV-cache results are documented in PART_D_KV_CACHE_SUMMARY.md.

### Hugging Face vs vLLM

- Prompt length: 128 tokens/request
- Generation length: 32 tokens/request
- KV cache enabled in both backends
- One warm-up and five measured trials per batch

| Batch | HF tokens/s | vLLM tokens/s | vLLM speedup |
|---:|---:|---:|---:|
| 1 | 172.64 | 141.74 | 0.82x |
| 8 | 498.37 | 1497.43 | 3.00x |
| 32 | 1640.83 | 6050.74 | 3.69x |

At batch size 1, Hugging Face had higher throughput. At batches 8 and 32, vLLM achieved higher aggregate generated-token throughput.

The tested implementations differ: vLLM used FlashAttention, while Hugging Face used eager attention. The results do not isolate framework overhead.

## GPU-hours

- Documented measured execution: **563.198 seconds**
- Documented measured GPU-hours: **0.156444 GPU-hours**
- Duplicate Part B FP32 baseline: excluded.
- Part C DDP timing: exact source value used.

**This is not total GPU reservation usage.**

Warm-ups, model loading, engine startup, unrecorded debugging, and idle reservation periods are excluded. Full GPU-hours require allocation or reservation logs.

## Limitations

- Single RTX 4090; no two-GPU scaling measurements.
- Part A large-batch slowdown should not be labeled an OOM unless an actual OOM was observed.
- Part D uses a separate causal language model because BERT is encoder-only.
- HF and vLLM memory figures use different definitions.
- Benchmark timings exhibit trial-to-trial variability.

## Supporting reports

- PART_A_SUMMARY.md
- PART_B_SUMMARY.md
- PART_C_SUMMARY.md
- PART_D_KV_CACHE_SUMMARY.md
- PART_D_BACKEND_SUMMARY.md
- PART_E_GPU_HOURS_CORRECTED.md


## GPU reservation accounting (student-reported)

- Reported total GPU reservation: 6.000 hours
- GPU count: 1
- Reported total GPU-hours: 6.000
- Source: Student-reported; not verified against lab records
- Documented benchmark execution: 0.156444 GPU-hours
