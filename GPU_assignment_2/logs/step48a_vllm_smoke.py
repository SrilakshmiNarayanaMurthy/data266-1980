
import os

# Must be set before importing vLLM or initializing CUDA.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import json
import time
from pathlib import Path
from importlib.metadata import version


def main():

    from vllm import LLM, SamplingParams

    project_dir = Path("/app/HW4_5_GPU_Assignment")
    model_dir = project_dir / "models/opt_125m"
    metrics_dir = project_dir / "metrics"

    assert (model_dir / "config.json").exists()

    print("vLLM version:", version("vllm"), flush=True)
    print("Multiprocessing method: spawn", flush=True)

    print("\nInitializing vLLM engine...", flush=True)

    start_init = time.perf_counter()

    llm = LLM(
        model=str(model_dir),
        dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        max_model_len=1024,
        enforce_eager=True,
        trust_remote_code=False
    )

    init_seconds = time.perf_counter() - start_init

    print(
        f"Engine initialized in {init_seconds:.2f}s",
        flush=True
    )

    prompt = "Artificial intelligence in healthcare can help"

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=16,
        ignore_eos=True
    )

    print("\nGenerating 16 tokens...", flush=True)

    start = time.perf_counter()

    outputs = llm.generate(
        [prompt],
        sampling_params,
        use_tqdm=False
    )

    generation_seconds = time.perf_counter() - start

    assert len(outputs) == 1

    generated = outputs[0].outputs[0]

    assert len(generated.token_ids) == 16

    result = {
        "model": "facebook/opt-125m",
        "vllm_version": version("vllm"),
        "multiprocessing_method": "spawn",
        "engine_initialization_seconds": init_seconds,
        "generation_seconds": generation_seconds,
        "generated_tokens": len(generated.token_ids),
        "prompt": prompt,
        "generated_text": generated.text,
        "success": True
    }

    output_path = (
        metrics_dir / "part_d_vllm_smoke_test.json"
    )

    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

    print("\nGENERATION RESULT")
    print("-" * 65)
    print("Prompt:", prompt)
    print("Output:", generated.text)
    print("Generated tokens:", len(generated.token_ids))
    print(f"Generation time: {generation_seconds:.3f}s")
    print(f"Engine startup: {init_seconds:.3f}s")

    print("\nSaved:", output_path)
    print("PASS: vLLM inference verified.", flush=True)


if __name__ == "__main__":
    main()
