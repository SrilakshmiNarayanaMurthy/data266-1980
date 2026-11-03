# Part E — Results Audit

## Deliverables

- PASS: Part A summary
- PASS: Part B summary
- PASS: Part C summary
- PASS: Part D KV summary
- PASS: Part D backend summary
- PASS: Part D vLLM results
- PASS: Part D HF results
- PASS: Part D comparison
- PASS: Part D KV benchmark
- PASS: Part C DDP result

## Training optimization findings

- Fastest: BF16 AMP
- Throughput: 768.428 examples/s
- Lowest memory: Gradient checkpointing
- Peak allocated: 5.176 GiB

## GPU-hours accounting

- Unadjusted sum of selected recorded durations: 0.031613 GPU-hours
- This is NOT the final GPU-hours total.
- Some experiments may overlap in the ledger.
- Additional timed GPU activities must be included.

## Accounting gaps

- Part A ten-step batch-scaling benchmark durations
- Part D KV-cache benchmark full execution duration
- Part D vLLM inference trial durations
- Part D Hugging Face inference trial durations
- Model loading and GPU initialization durations
- Warm-up execution durations
- Possible overlap between Part A and Part B runs
- Unrecorded debugging and GPU-active setup time
