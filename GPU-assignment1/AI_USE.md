# AI Use Disclosure — HW2.5

## What I used AI assistance for

I used ChatGPT as a coding and debugging assistant while working through the GPU experiments. It helped me organize the notebook into reproducible steps, write benchmark code, interpret CUDA/PyTorch errors, and check whether my measured results satisfied the assignment requirements.

I personally ran all experiments on the assigned RTX 4090, collected the measurements, inspected the outputs, reran benchmarks when necessary, and verified the final values reported in the notebook and METRICS.md.

## Specific incorrect AI output and how I corrected it

During Part C, the initial AI-assisted vector-add benchmark used an expression equivalent to `C = A + B` inside the timed region. This included output-tensor allocation in the measurement and produced an effective bandwidth of only about 58.1 GB/s, which was unexpectedly low for an RTX 4090.

I detected the problem by comparing the result with the RTX 4090's specified memory bandwidth and noticing that the measured value was only a small fraction of the expected hardware capability. I then changed the benchmark so that the output tensor was allocated before timing and used `torch.add(A, B, out=C)` inside the timed region.

After this correction, the measured effective bandwidth was 893.88 GB/s, approximately 88.68% of the specified 1008 GB/s bandwidth. The corrected version better isolates the GPU memory operation by removing allocation overhead from the timed section.

## Verification

I verified benchmark outputs directly from the executed notebook, checked GPU identity using the recorded GPU UUID, used warm-up runs and repeated measurements for timing experiments, and retained observed failures such as the FP8 PyTorch matmul error rather than replacing them with assumed results. I also checked the final assignment metrics against the wording of each required section before preparing METRICS.md.
