# Part E — Corrected GPU-Hours Accounting

## Hardware
- NVIDIA GeForce RTX 4090
- One GPU

## Documented measured execution
- Included intervals: 63
- Measured seconds: 563.199
- Measured GPU-hours: 0.156444

## Duplicate handling
- Part B FP32 baseline reused Part A batch-64 results.
- Counted the shared epoch exactly once.
- Excluded duplicate: 7.522 seconds.

## Scope
This is a documented measured-execution subtotal,
**not the total GPU reservation duration**.

Excluded: warm-ups, model loading, initialization,
debugging, and idle reservation periods.

Full GPU-hours require the actual GPU allocation
start/end times or lab usage records.
