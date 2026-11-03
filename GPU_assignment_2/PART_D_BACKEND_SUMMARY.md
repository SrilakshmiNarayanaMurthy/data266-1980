# Part D — Hugging Face vs vLLM

## Experimental setup

- GPU: NVIDIA GeForce RTX 4090
- Model: facebook/opt-125m
- Precision: FP16
- Prompt length: 128 tokens/request
- Generation: 32 tokens/request
- Batch sizes: 1, 8, 32
- Trials: 1 warm-up + 5 measured per batch
- KV cache: enabled in both backends
- Engine/model initialization excluded from inference timing

## Results

| Batch | HF tok/s | vLLM tok/s | vLLM speedup |
|---:|---:|---:|---:|
| 1 | 172.64 | 141.74 | 0.82x |
| 8 | 498.37 | 1497.43 | 3.00x |
| 32 | 1640.83 | 6050.74 | 3.69x |

## Findings

- HF achieved higher throughput at batch size 1.
- vLLM achieved higher aggregate throughput at batches 8 and 32.
- The observed benefit increased with batch size.
- Speedup values describe aggregate generated-token throughput.

## Limitations

- vLLM used FlashAttention, whereas HF used eager attention.
- This is a comparison of tested configurations, not an isolated framework-only comparison.
- vLLM was configured with eager execution and prefix caching disabled.
- WSL disabled pinned memory for vLLM.
- Latency is batch completion time, not individual request latency.
- Memory measurements have different definitions and must not be directly compared.
- HF batch-8 trial 5 was an unusually fast observation; all five trials were retained.
- Results are specific to OPT-125M, RTX 4090, and the tested workload.

## Generated figures

- figures/part_d_backend_throughput.png
- figures/part_d_backend_speedup.png
- figures/part_d_backend_latency.png
