# HW5 Metrics: LoRA Fine-Tuning on DialogSum

## Experiment Configuration

| Setting | Value |
|---|---|
| Base model | `google/flan-t5-small` |
| Dataset | `neil-code/dialogsum-test` |
| Training split | 1,999 examples |
| Training subset | 300 examples |
| Random seed | 1980 |
| Epochs | 1 |
| Learning rate | 0.0001 |
| Batch size | 4 |
| Gradient accumulation steps | 2 |
| LoRA target modules | `q`, `v` |
| LoRA alpha | 16 |
| LoRA dropout | 0.1 |
| Mixed precision | Disabled |
| Hardware | Google Colab Tesla T4 GPU |

## LoRA Parameter Comparison

| LoRA Rank | Total Parameters | Trainable Parameters | Trainable Percentage |
|---:|---:|---:|---:|
| 4 | 77,133,184 | 172,032 | 0.2230% |
| 16 | 77,649,280 | 688,128 | 0.8862% |

The rank-16 model trained four times more adapter parameters than the
rank-4 model.

## Training Metrics

| LoRA Rank | Final Training Loss | Training Runtime (seconds) | Epochs |
|---:|---:|---:|---:|
| 4 | 4.768120815879421 | 22.1663 | 1 |
| 16 | 4.767175774825247 | 22.48 | 1 |

The rank-16 model achieved a slightly lower training loss than the rank-4
model. The difference was very small.

## Baseline and Fine-Tuned Outputs

### Dialogue 1

| Model | Generated Summary |
|---|---|
| Baseline | Get a check-up. |
| LoRA rank 4 | Get a check-up. |
| LoRA rank 16 | Get a check-up. |

Reference summary:

> Mr. Smith's getting a check-up, and Doctor Hawkins advises him to have
> one every year. Hawkins'll give some information about their classes and
> medications to help Mr. Smith quit smoking.

### Dialogue 2

| Model | Generated Summary |
|---|---|
| Baseline | Ricky and I are going to see a doctor. |
| LoRA rank 4 | Ricky and I are going to see a doctor. |
| LoRA rank 16 | Ricky and I are going to see a doctor. |

Reference summary:

> Mrs Parker takes Ricky for his vaccines. Dr. Peters checks the record
> and then gives Ricky a vaccine.

## Output Quality Comparison

The baseline, rank-4, and rank-16 models produced the same outputs for
both evaluation dialogues. Therefore, no visible improvement was
observed after LoRA fine-tuning or after increasing the LoRA rank from 4
to 16.

The lack of visible improvement may be related to the small training
subset, the one-epoch training schedule, and evaluation on only two
dialogues.

## Saved Artifacts

| Artifact | Location |
|---|---|
| Rank-4 LoRA adapter | `dialogsum_lora_r4_adapter/` |
| Rank-16 LoRA adapter | `dialogsum_lora_r16_adapter/` |
| Baseline versus rank-4 comparison | `dialogsum_baseline_vs_lora_r4.csv` |
| Rank comparison | `dialogsum_lora_rank_comparison.csv` |