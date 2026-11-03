
import csv
import json
import time
import numpy as np
import torch

from pathlib import Path
from transformers import AutoModelForCausalLM


def main():

    project = Path("/app/HW4_5_GPU_Assignment")
    model_dir = project / "models/opt_125m"
    metrics_dir = project / "metrics"

    vllm_path = metrics_dir / "part_d_vllm_benchmark.json"

    assert vllm_path.exists(), "Step 49 results missing."

    with open(vllm_path) as f:
        vllm_data = json.load(f)

    prompts = vllm_data["prompt_token_ids"]

    batch_sizes = [1, 8, 32]
    new_tokens = 32
    measured_trials = 5

    assert len(prompts) >= 32
    assert all(len(p) == 128 for p in prompts[:32])

    print("=" * 65, flush=True)
    print("HW4.5 PART D — STEP 50", flush=True)
    print("HUGGING FACE OPT-125M BASELINE", flush=True)
    print("=" * 65, flush=True)

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Model: OPT-125M", flush=True)
    print("Precision: FP16", flush=True)
    print("KV cache: ENABLED", flush=True)
    print("Prompt length: 128 tokens", flush=True)
    print("Generation length: 32 tokens", flush=True)
    print("Batch sizes:", batch_sizes, flush=True)

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\nLoading Hugging Face model...", flush=True)

    start_load = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True
    ).to("cuda")

    model.eval()
    torch.cuda.synchronize()

    load_seconds = time.perf_counter() - start_load

    print(
        f"Model loaded in {load_seconds:.2f}s",
        flush=True
    )

    rows = []
    trial_records = []

    # --------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------

    for batch_size in batch_sizes:

        input_ids = torch.tensor(
            prompts[:batch_size],
            dtype=torch.long,
            device="cuda"
        )

        attention_mask = torch.ones_like(input_ids)

        def generate():
            with torch.inference_mode():
                return model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=new_tokens,
                    min_new_tokens=new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=model.config.eos_token_id
                )

        print(
            f"\nBatch {batch_size}: warm-up...",
            flush=True
        )

        generate()
        torch.cuda.synchronize()

        durations = []
        peak_allocations = []

        for trial in range(measured_trials):

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            start = time.perf_counter()

            output_ids = generate()

            torch.cuda.synchronize()

            elapsed = time.perf_counter() - start

            generated_count = (
                output_ids.shape[0] *
                (output_ids.shape[1] - input_ids.shape[1])
            )

            assert output_ids.shape[0] == batch_size
            assert generated_count == batch_size * new_tokens

            peak_gib = (
                torch.cuda.max_memory_allocated()
                / (1024 ** 3)
            )

            durations.append(elapsed)
            peak_allocations.append(peak_gib)

            trial_records.append({
                "batch_size": batch_size,
                "trial": trial + 1,
                "seconds": elapsed,
                "generated_tokens": generated_count,
                "tokens_per_second":
                    generated_count / elapsed,
                "requests_per_second":
                    batch_size / elapsed,
                "peak_allocated_gib": peak_gib
            })

            print(
                f"  Trial {trial + 1}/{measured_trials}: "
                f"{elapsed:.3f}s, "
                f"{generated_count / elapsed:.2f} tok/s",
                flush=True
            )

        total_time = sum(durations)
        total_tokens = batch_size * new_tokens * measured_trials

        row = {
            "batch_size": batch_size,
            "prompt_tokens_per_request": 128,
            "generated_tokens_per_request": new_tokens,
            "trials": measured_trials,
            "throughput_tokens_per_second":
                total_tokens / total_time,
            "throughput_requests_per_second":
                batch_size * measured_trials / total_time,
            "batch_latency_p50_ms":
                float(np.percentile(durations, 50) * 1000),
            "batch_latency_p95_ms":
                float(np.percentile(durations, 95) * 1000),
            "peak_allocated_gib":
                max(peak_allocations)
        }

        rows.append(row)

        print(
            f"  Aggregate throughput: "
            f"{row['throughput_tokens_per_second']:.2f} tok/s",
            flush=True
        )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    csv_path = metrics_dir / "part_d_hf_benchmark.csv"
    json_path = metrics_dir / "part_d_hf_benchmark.json"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w") as f:
        json.dump({
            "model": "facebook/opt-125m",
            "backend": "Hugging Face Transformers",
            "dtype": "float16",
            "attention_implementation": "eager",
            "use_cache": True,
            "gpu": torch.cuda.get_device_name(0),
            "model_load_seconds": load_seconds,
            "warmup_runs_per_batch": 1,
            "measured_trials_per_batch": measured_trials,
            "prompt_source": str(vllm_path),
            "summary": rows,
            "trials": trial_records,
            "latency_definition":
                "Wall-clock time to complete an entire batch",
            "memory_definition":
                "Peak PyTorch allocated memory, not total device usage"
        }, f, indent=2)

    print("\n" + "=" * 65, flush=True)
    print("HUGGING FACE BENCHMARK SUMMARY", flush=True)
    print("=" * 65, flush=True)

    for row in rows:
        print(
            f"Batch {row['batch_size']:>2}: "
            f"{row['throughput_tokens_per_second']:>9.2f} tok/s | "
            f"p50 {row['batch_latency_p50_ms']:>8.2f} ms | "
            f"p95 {row['batch_latency_p95_ms']:>8.2f} ms | "
            f"peak {row['peak_allocated_gib']:.3f} GiB",
            flush=True
        )

    print("\nSaved:", csv_path, flush=True)
    print("Saved:", json_path, flush=True)
    print("PASS: Hugging Face baseline complete.", flush=True)


if __name__ == "__main__":
    main()
