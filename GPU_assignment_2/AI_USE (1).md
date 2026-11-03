# HW4.5 GPU Assignment II — AI Use Disclosure

## AI Tool

ChatGPT (OpenAI) was used as a coding, debugging,
documentation, and analysis assistant.

## Scope of AI Assistance

AI assistance was used to:

1. Structure the assignment into sequential notebook steps.
2. Suggest Python and PyTorch implementations for experiments.
3. Help implement BERT training and inference measurements.
4. Suggest batch-scaling and optimization comparisons.
5. Help configure BF16 automatic mixed precision,
   gradient checkpointing, gradient accumulation, and
   8-bit AdamW.
6. Suggest single-GPU PyTorch DistributedDataParallel
   validation using NCCL.
7. Explain why an encoder-only BERT model is unsuitable
   for autoregressive KV-cache generation experiments.
8. Assist with separate-environment setup for vLLM.
9. Help troubleshoot vLLM CUDA multiprocessing
   initialization and switch to spawn.
10. Suggest benchmarking code for Hugging Face and vLLM.
11. Help calculate throughput, latency, and GPU memory
    comparisons from recorded experiment outputs.
12. Generate structured reports, figures, run logs,
    and GPU-hours accounting scripts.

## Human Execution and Verification

All reported numerical experiment results were obtained
from code executed in the assignment environment.

The student ran the notebook cells, inspected execution
outputs, and provided the resulting logs for review.

AI-generated code was not treated as experimental evidence
until it had executed and produced recorded results.

The student reviewed intermediate outputs before proceeding
to subsequent steps.

## AI-Assisted Debugging

### Dependency isolation

A separate virtual environment was used for vLLM
dependencies to avoid disrupting the primary PyTorch
training environment.

### vLLM multiprocessing

An initial vLLM smoke test failed because of CUDA
multiprocessing initialization behavior.

The worker multiprocessing method was changed to spawn,
after which the standalone smoke test succeeded.

### Benchmark interpretation

The AI assistant helped identify that:

- A slowdown at large batch sizes is not equivalent to
  a confirmed CUDA out-of-memory failure.
- Single-GPU DDP does not establish multi-GPU scaling.
- Hugging Face and vLLM throughput comparisons do not
  isolate framework overhead when attention backends differ.
- GPU memory values from different measurement methods
  are not directly interchangeable.
- Recorded benchmark execution time is not the same
  as full GPU reservation time.

## AI Limitations and Corrections

AI-generated suggestions were checked against observed
outputs and saved metrics.

During GPU-hours accounting, the Part B FP32 baseline
was initially treated as a potentially separate duration.

Inspection of the source JSON showed that the Part B
FP32 baseline reused the Part A batch-64 experiment.

The duplicate was excluded from the corrected ledger.

The Part C DDP duration was also corrected from a rounded
value to the exact recorded elapsed_seconds.

The GPU-hours report explicitly excludes unrecorded
warm-ups, initialization, troubleshooting, and idle
reservation time.

## Experimental Integrity

AI did not independently run the GPU experiments.

Performance results were not invented to fill missing
measurements.

No two-GPU speedup or communication-overhead results
were claimed because only one GPU was available.

The assignment used a separate causal language model
for Part D rather than incorrectly claiming that BERT
supports autoregressive KV-cache generation.

## Reproducibility and Supporting Evidence

Recorded experiment evidence is available in:

- METRICS.md
- RUN_LOG.txt
- PART_A_SUMMARY.md
- PART_B_SUMMARY.md
- PART_C_SUMMARY.md
- PART_D_KV_CACHE_SUMMARY.md
- PART_D_BACKEND_SUMMARY.md
- metrics/part_e_final_metrics.json
- metrics/part_e_gpu_hours_final.csv

## Responsibility

The student remains responsible for reviewing the code,
verifying experiment outputs, interpreting results,
and ensuring the final submission complies with the
course's academic integrity and AI-use policies.
