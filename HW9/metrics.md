
# DATA 266 — Homework 9
## Latent Diffusion and Stable Diffusion on FashionMNIST

### 1. Experiment Configuration

| Parameter | Value |
|---|---|
| Dataset | FashionMNIST |
| Personal seed | 1980 |
| Training samples | 60,000 |
| Test samples | 10,000 |
| Image resolution | 128 × 128 |
| Image channels | 3 (RGB) |
| Pretrained VAE | stabilityai/sd-vae-ft-mse |
| Latent dimensions | 4 × 16 × 16 |
| Latent scaling factor | 0.18215 |
| Diffusion timesteps | 1000 |
| Noise schedule | Linear beta |
| Beta start | 0.0001 |
| Beta end | 0.02 |
| Model | Conditional Latent Diffusion U-Net |
| Conditional U-Net parameters | 2,062,852 |
| Timestep embedding dimension | 256 |
| Attention heads | 4 |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Batch size | 32 |
| Training epochs | 5 |
| CFG dropout probability | 0.1 |
| GPU | NVIDIA Tesla T4 |
| Python | 3.13.16 |
| PyTorch | 2.11.0+cu130 |

### 2. VAE Reconstruction

The pretrained Stable Diffusion VAE was used to encode
FashionMNIST images into a lower-dimensional latent space.

- Input image dimensions: 3 × 128 × 128
- Latent dimensions: 4 × 16 × 16
- Latent scaling factor: 0.18215
- Reconstruction MSE: 0.001052

The reconstruction experiment confirmed that images could
be encoded and decoded using the pretrained VAE.

Output:
`vae_reconstruction_comparison.png`

### 3. Diffusion Model Training

The conditional U-Net was trained to predict Gaussian noise
added to VAE latents at randomly selected timesteps.

| Epoch | Average Training MSE |
|---|---:|
| 1 | 0.160785 |
| 2 | 0.113313 |
| 3 | 0.107501 |
| 4 | 0.102618 |
| 5 | 0.103192 |

#### Training Summary

| Metric | Result |
|---|---:|
| Initial epoch loss | 0.160785 |
| Final epoch loss | 0.103192 |
| Best epoch | 4 |
| Best epoch loss | 0.102618 |
| Training loss reduction | 35.82% |
| Total training time | Approximately 71.7 minutes |
| Checkpoints saved | 5 |

The training loss decreased substantially from epoch 1
to epoch 5.

Epoch 4 achieved the lowest average training loss.
Epoch 5 showed a small increase, indicating that training
loss was not strictly decreasing.

The observed loss reduction demonstrates improvement in
noise prediction on the training batches. It does not,
by itself, establish generalization or image realism.

Output:
`latent_training_loss.png`

### 4. DDIM Sampling

A deterministic DDIM sampler was implemented to generate
new latent representations from Gaussian noise.

| Parameter | Value |
|---|---|
| Sampling method | DDIM |
| Eta | 0 |
| Sampling steps | 50 |
| CFG scale | 3.0 |
| Classes | T-shirt/top, Trouser |
| Samples per class | 4 |
| Total generated images | 8 |
| Sampling time | 0.5152 seconds |

The sampled latents were decoded using the pretrained VAE.

Output:
`cfg_generated_fashionmnist.png`

### 5. Generated Image Quality

Eight generated images were visually reviewed.

A three-level rating scale was used:

- 0: Unrecognizable
- 1: Partially recognizable
- 2: Clearly recognizable

| Metric | Result |
|---|---:|
| Total images reviewed | 8 |
| Clearly recognizable | 5 |
| Partially recognizable | 3 |
| Unrecognizable | 0 |
| Clear recognition rate | 62.50% |
| Mean quality rating | 1.625 / 2 |

#### Class-Level Results

| Class | Images | Mean Rating | Clearly Recognizable |
|---|---:|---:|---:|
| T-shirt/top | 4 | 1.25 | 1 |
| Trouser | 4 | 2.00 | 4 |

Trouser images were generally more recognizable than
T-shirt/top images in this small visual evaluation.

These results are subjective visual ratings, not
classification accuracy.

### 6. Image Diversity Analysis

Pairwise pixel-level mean squared error was calculated
between generated images belonging to the same class.

| Metric | Result |
|---|---:|
| T-shirt/top pairwise MSE | 0.080547 |
| Trouser pairwise MSE | 0.023142 |
| Between-class mean-image difference | 0.237177 |

T-shirt/top samples showed greater pixel-level variation
than trouser samples.

Pixel-level MSE does not independently establish
perceptual diversity or image quality.

### 7. Classifier-Free Guidance Comparison

Three guidance scales were evaluated:

- CFG = 1.0
- CFG = 3.0
- CFG = 5.0

The same initial noise was used for matched comparisons.

| CFG Scale | Sampling Time |
|---|---:|
| 1.0 | 0.7765 seconds |
| 3.0 | 0.7240 seconds |
| 5.0 | 0.7261 seconds |

#### Pairwise Image Differences

| Comparison | Pixel MSE | Pixel MAE |
|---|---:|---:|
| CFG 1.0 vs 3.0 | 0.046194 | 0.128931 |
| CFG 1.0 vs 5.0 | 0.058263 | 0.149806 |
| CFG 3.0 vs 5.0 | 0.002903 | 0.029414 |

The results show that CFG 3.0 and CFG 5.0 generated
relatively similar images for the matched samples.

CFG 1.0 produced more visibly different outputs.

#### Selected CFG Scale: 3.0

CFG 1.0 produced more fragmented T-shirt shapes and
visible artifacts.

CFG 3.0 generated more recognizable T-shirt and
trouser silhouettes.

CFG 5.0 showed no clear visual improvement over CFG 3.0.

Therefore, CFG 3.0 was selected for the final experiment.

This selection is based on a qualitative comparison
of four matched images per scale, not a statistically
established optimum.

### 8. Limitations

1. Training was limited to five epochs.
2. Some generated images were blurry.
3. T-shirt/top samples contained shape distortions.
4. Visual evaluation covered only eight images.
5. CFG comparison used four matched samples per scale.
6. No external classifier accuracy was measured.
7. No FID score was calculated.
8. No independent held-out denoising loss was reported.

### 9. Conclusion

The experiment successfully implemented a conditional
latent diffusion pipeline using a pretrained Stable
Diffusion VAE, a custom conditional U-Net, classifier-free
guidance, and deterministic DDIM sampling.

Training noise-prediction loss decreased by 35.82%
over five epochs.

Five of eight generated samples were judged clearly
recognizable in the visual review.

CFG 3.0 provided the preferred qualitative results
among the tested guidance scales.

The experiment demonstrates the latent diffusion workflow
while also highlighting limitations in generated image
quality and evaluation sample size.
