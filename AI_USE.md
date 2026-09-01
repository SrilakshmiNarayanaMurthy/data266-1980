# AI Use Disclosure

## 1. Which parts did AI help with, and which parts did I complete myself?

I used AI as a support tool to help explain assignment requirements, understand PyTorch, TensorFlow, and CUDA concepts, structure portions of the code, and troubleshoot errors during implementation.

I personally ran all experiments in Google Colab, executed and verified the code, inspected the outputs, compared the models, collected the timing measurements, and reviewed the final results.

## 2. Give one specific example where AI produced an incorrect or unsuitable output.

During the CUDA experiment, the initial CPU matrix multiplication implementation used a simple row-column-k loop ordering. For the required 4096 x 4096 matrix, this implementation was extremely slow and the execution took more than 30 minutes, so I interrupted the run.

## 3. How did I identify the problem?

I identified the problem by observing the execution time in Google Colab. The smaller matrix sizes completed, but the 4096 x 4096 experiment was taking an impractical amount of time. This showed that the CPU implementation needed improvement before collecting the final measurements.

## 4. What did I change, and why does the corrected approach work?

I changed the CPU matrix multiplication loop ordering to a more cache-friendly row-k-column implementation while preserving the same matrix multiplication computation. This improves memory access locality and reduces unnecessary memory-access overhead. After making the change, I recompiled the CUDA program, reran all three matrix sizes, verified that the result checks passed, and used the new measurements consistently in the final comparison.