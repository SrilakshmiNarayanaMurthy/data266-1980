# Homework 7 - Experiment Metrics

## 1. Personal Parameters

| Parameter | Value |
|---|---|
| SID4 | 1980 |
| SEED | 1980 |
| SLICE | 980 |
| HP_ID | 0 |
| CLS_A | 0 |
| CLS_B | 1 |

## 2. Dataset

| Property | Value |
|---|---|
| Dataset | FashionMNIST |
| Training images | 60,000 |
| Test images | 10,000 |
| Image dimensions | 1 x 28 x 28 |
| Classes | 10 |
| Pixel range | [0, 1] |

## 3. Model Configuration

| Configuration | FC-AE | VAE |
|---|---|---|
| Architecture | Fully Connected | Variational |
| Latent dimension | 32 | 32 |
| Parameters | 476,720 | 480,848 |
| Epochs | 20 | 20 |
| Batch size | 128 | 128 |
| Learning rate | 0.001 | 0.001 |
| Optimizer | Adam | Adam |
| Training objective | MSE | BCE + KL |
| KL weight | N/A | 1.0 |

## 4. Reconstruction Performance

| Metric | FC-AE | VAE |
|---|---|---|
| Full test reconstruction MSE | 0.010034 | 0.014975 |
| Sample reconstruction MSE | 0.008933 | 0.014696 |
| Training time (seconds) | 85.80 | 87.08 |
| Best epoch | 20 | 20 |

FC-AE achieved 32.99% lower
full-test reconstruction MSE than VAE.

Both models were evaluated on the same 10,000 test images.

VAE reconstruction used the latent mean for deterministic
evaluation.

## 5. FC-AE Training Results

| Metric | Value |
|---|---|
| Final training MSE | 0.009818 |
| Final test MSE | 0.010034 |
| Best test MSE | 0.010034 |
| Best epoch | 20 |

## 6. VAE Training Results

| Metric | Value |
|---|---|
| Final training objective | 239.4515 |
| Final test objective | 241.3774 |
| Final test reconstruction BCE | 228.2453 |
| Final test KL divergence | 13.1321 |
| Best test objective | 241.3774 |
| Best epoch | 20 |

VAE objective values are per image.
They are not directly comparable to FC-AE MSE.

## 7. Class-Wise Reconstruction MSE

| Class       |   Test Images |   FC-AE MSE |   VAE MSE |
|:------------|--------------:|------------:|----------:|
| T-shirt/top |          1000 |    0.009094 |  0.014205 |
| Trouser     |          1000 |    0.004761 |  0.007871 |
| Pullover    |          1000 |    0.008963 |  0.014230 |
| Dress       |          1000 |    0.008971 |  0.013893 |
| Coat        |          1000 |    0.008485 |  0.013162 |
| Sandal      |          1000 |    0.015488 |  0.022542 |
| Shirt       |          1000 |    0.010072 |  0.014468 |
| Sneaker     |          1000 |    0.007544 |  0.011727 |
| Bag         |          1000 |    0.016613 |  0.022260 |
| Ankle boot  |          1000 |    0.010351 |  0.015388 |

## 8. Worst Reconstruction Examples

| Model   |   Rank |   Test Index | Class       |   Reconstruction MSE |
|:--------|-------:|-------------:|:------------|---------------------:|
| FC-AE   |      1 |         9067 | Pullover    |             0.087229 |
| FC-AE   |      2 |         3953 | T-shirt/top |             0.086388 |
| FC-AE   |      3 |         7348 | Shirt       |             0.085874 |
| FC-AE   |      4 |         1286 | Bag         |             0.080455 |
| FC-AE   |      5 |         4392 | Trouser     |             0.076498 |
| FC-AE   |      6 |         3485 | T-shirt/top |             0.075365 |
| VAE     |      1 |         1286 | Bag         |             0.098278 |
| VAE     |      2 |         6191 | T-shirt/top |             0.096253 |
| VAE     |      3 |         4193 | T-shirt/top |             0.095782 |
| VAE     |      4 |         9067 | Pullover    |             0.093127 |
| VAE     |      5 |         3953 | T-shirt/top |             0.089342 |
| VAE     |      6 |         7348 | Shirt       |             0.088180 |

## 9. Interpretation

The FC-AE achieved lower pixel-level reconstruction
error than the VAE.

The VAE additionally regularizes its latent distribution,
supporting sampling of new images.

Both models had their lowest average class-wise MSE
for trousers.

The FC-AE's highest average class-wise MSE was for bags,
while the VAE's was for sandals.

The experiment uses one seed and one architecture
configuration per model. These findings should not be
generalized to all autoencoder architectures.

## 10. Saved Figures

- figures/fc_ae_loss_curve.png
- figures/fc_ae_reconstructions.png
- figures/vae_training_curves.png
- figures/vae_reconstructions.png
- figures/fc_ae_vs_vae_mse_comparison.png
- figures/vae_generated_images.png
- figures/class_wise_reconstruction_mse.png
- figures/worst_reconstruction_examples.png

## 11. Additional Artifacts

- checkpoints/fc_ae_final.pth
- checkpoints/vae_final.pth
- metrics/fc_ae_training_history.csv
- metrics/vae_training_history.csv
- metrics/fc_ae_vs_vae_comparison.json
- FAILURE_ANALYSIS.md

Report generated: 2026-10-20T15:39:27.144482
