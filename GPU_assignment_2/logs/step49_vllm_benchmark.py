
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import csv
import json
import time
import statistics
import numpy as np
import torch

from pathlib import Path


def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    project = Path("/app/HW4_5_GPU_Assignment")
    model_dir = project / "models/opt_125m"
    metrics_dir = project / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    batch_sizes = [1, 8, 32]
    generated_tokens_per_request = 32
    measured_trials = 5
    prompt_length = 128

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True
    )

    # Use deterministic, distinct token sequences across requests.
    # These same prompt token IDs can be reused for HF comparison.
    base_ids = tokenizer.encode(
        "Clinical language models must generate reliable "
        "and evidence-grounded healthcare explanations. ",
        add_special_tokens=False
    )

    assert len(base_ids) > 0

    prompts = []
    for i in range(max(batch_sizes)):
        ids = (base_ids * 20)[:prompt_length]
        ids = list(ids)
        ids[-1] = tokenizer.encode(
            f" {i}", add_special_tokens=False
        )[-1]
        prompts.append(ids)

    assert all(len(p) == prompt_length for p in prompts)

    print("=" * 65, flush=True)
    print("HW4.5 PART D — STEP 49", flush=True)
    print("vLLM BATCH THROUGHPUT BENCHMARK", flush=True)
    print("=" * 65, flush=True)

    print("Model: OPT-125M", flush=True)
    print("Batch sizes:", batch_sizes, flush=True)
    print("Prompt tokens/request:", prompt_length, flush=True)
    print(
        "Generated tokens/request:",
        generated_tokens_per_request,
        flush=True
    )
    print("Measured trials/batch:", measured_trials, flush=True)

    print("\nInitializing engine once...", flush=True)
    startup = time.perf_counter()

    llm = LLM(
        model=str(model_dir),
        dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        max_model_len=1024,
        enforce_eager=True,
        enable_prefix_caching=False,
        trust_remote_code=False
    )

    startup_seconds = time.perf_counter() - startup
    print(
        f"Engine startup: {startup_seconds:.2f}s",
        flush=True
    )

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=generated_tokens_per_request,
        ignore_eos=True
    )

    rows = []
    trial_records = []

    for batch_size in batch_sizes:
        batch_prompts = prompts[:batch_size]

        print(
            f"\nBatch {batch_size}: warm-up...",
            flush=True
        )

        llm.generate(
            [{"prompt_token_ids": p} for p in batch_prompts],
            sampling,
            use_tqdm=False
        )

        durations = []
        token_counts = []
        memory_used = []

        for trial in range(measured_trials):
            torch.cuda.synchronize()
            free_before, total_memory = torch.cuda.mem_get_info()

            start = time.perf_counter()

            outputs = llm.generate(
                [
                    {"prompt_token_ids": p}
                    for p in batch_prompts
                ],
                sampling,
                use_tqdm=False
            )

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            generated_count = sum(
                len(output.outputs[0].token_ids)
                for output in outputs
            )

            assert len(outputs) == batch_size
            assert generated_count == (
                batch_size * generated_tokens_per_request
            )

            free_after, _ = torch.cuda.mem_get_info()

            # Device memory used at the sampled endpoint.
            # This is NOT peak allocated memory.
            used_gib = (
                total_memory - free_after
            ) / (1024 ** 3)

            durations.append(elapsed)
            token_counts.append(generated_count)
            memory_used.append(used_gib)

            trial_records.append({
                "batch_size": batch_size,
                "trial": trial + 1,
                "seconds": elapsed,
                "generated_tokens": generated_count,
                "tokens_per_second": generated_count / elapsed,
                "requests_per_second": batch_size / elapsed,
                "gpu_used_gib_after": used_gib
            })

            print(
                f"  Trial {trial + 1}/{measured_trials}: "
                f"{elapsed:.3f}s, "
                f"{generated_count / elapsed:.2f} tok/s",
                flush=True
            )

        total_tokens = sum(token_counts)
        total_time = sum(durations)

        row = {
            "batch_size": batch_size,
            "prompt_tokens_per_request": prompt_length,
            "generated_tokens_per_request":
                generated_tokens_per_request,
            "trials": measured_trials,
            "throughput_tokens_per_second":
                total_tokens / total_time,
            "throughput_requests_per_second":
                (batch_size * measured_trials) / total_time,
            "batch_latency_p50_ms":
                float(np.percentile(durations, 50) * 1000),
            "batch_latency_p95_ms":
                float(np.percentile(durations, 95) * 1000),
            "gpu_used_gib_after_max": max(memory_used)
        }

        rows.append(row)

        print(
            f"  Aggregate throughput: "
            f"{row['throughput_tokens_per_second']:.2f} tok/s",
            flush=True
        )

    csv_path = metrics_dir / "part_d_vllm_benchmark.csv"
    json_path = metrics_dir / "part_d_vllm_benchmark.json"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w") as f:
        json.dump({
            "model": "facebook/opt-125m",
            "backend": "vLLM",
            "dtype": "float16",
            "gpu": torch.cuda.get_device_name(0),
            "engine_startup_seconds": startup_seconds,
            "warmup_runs_per_batch": 1,
            "measured_trials_per_batch": measured_trials,
            "prefix_caching_enabled": False,
            "prompt_token_ids": prompts,
            "summary": rows,
            "trials": trial_records,
            "latency_definition":
                "Wall-clock time to complete an entire batch",
            "gpu_memory_definition":
                "Sampled device memory used after each trial, not peak"
        }, f, indent=2)

    print("\n" + "=" * 65, flush=True)
    print("vLLM BENCHMARK SUMMARY", flush=True)
    print("=" * 65, flush=True)

    for row in rows:
        print(
            f"Batch {row['batch_size']:>2}: "
            f"{row['throughput_tokens_per_second']:>9.2f} tok/s | "
            f"p50 {row['batch_latency_p50_ms']:>8.2f} ms | "
            f"p95 {row['batch_latency_p95_ms']:>8.2f} ms",
            flush=True
        )

    print("\nSaved:", csv_path, flush=True)
    print("Saved:", json_path, flush=True)
    print("PASS: vLLM batch benchmark complete.", flush=True)


if __name__ == "__main__":
    main()
