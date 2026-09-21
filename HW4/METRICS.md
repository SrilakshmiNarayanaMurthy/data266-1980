# HW4 — Mini GPT Metrics

## Model and training settings

| Measurement | Value |
|---|---:|
| SID4 / seed | 1980 |
| Vocabulary size | 49 |
| Sequence length | 128 |
| Hidden dimension | 128 |
| Attention heads | 4 |
| Decoder layers | 2 |
| Total parameters | 425,521 |
| Batch size | 32 |
| Adam learning rate | 0.0003 |
| Epochs | 6 |

## Training loss

| Epoch | Average training loss |
|---:|---:|
| 1 | 3.0354 |
| 2 | 2.4151 |
| 3 | 2.2322 |
| 4 | 2.1379 |
| 5 | 2.0589 |
| 6 | 1.9780 |

## Decoding settings

Prompt: `ROMEO:`; generated characters: 200 per output.

| Method | Settings |
|---|---|
| Greedy | Highest-scoring character |
| Temperature | 0.5, 1.0, 1.5 |
| Top-k | 3, 20 |
