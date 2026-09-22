# HW2.5 — GPU Assignment Metrics

**GPU:** NVIDIA GeForce RTX 4090  
**GPU UUID:** GPU-cf356fb7-8770-75da-1f3a-b57d7ba2ab58  
**SID4:** 1980

## Summary Metrics

| Metric | Result |
|---|---:|
| Peak achieved TFLOPS (BF16) | 175.53 TFLOPS |
| % theoretical peak (BF16) | 106.25% |
| Effective bandwidth | 893.88 GB/s |
| Naive attention OOM boundary | Success at 147,456; OOM at 163,840 |
| Fused attention OOM boundary | > 1,048,576 (no OOM observed through largest tested length) |
| Steady-state / peak throughput | 91.48% |
| Thermal throttle onset | None observed |
| Near-power-limit onset | 25.01 s |

## Notes

- BF16 peak throughput was measured at matrix size N = 16,384.
- The effective memory bandwidth measurement was 893.88 GB/s, or approximately 88.68% of the specified 1008 GB/s RTX 4090 memory bandwidth.
- Naive attention succeeded at sequence length 147,456 and failed at the next tested length of 163,840; therefore, no exact single-token OOM boundary is claimed.
- Fused/memory-efficient attention successfully processed every tested sequence length through 1,048,576, so no fused-attention OOM failure was observed.
- During the 20-minute sustained-load experiment, the maximum temperature was 77 C, below the 84 C target temperature; therefore, no thermal throttling was observed.
- The GPU operated close to its 450 W power ceiling, with near-power-limit operation first observed at approximately 25.01 seconds.
- Peak throughput during the first 30 seconds was 173.86 TFLOPS. Median steady-state throughput during the final 5 minutes was 159.05 TFLOPS, giving a steady-state/peak ratio of 91.48%.
