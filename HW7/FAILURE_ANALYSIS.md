# HW7 - Autoencoder Failure Analysis

## 1. Experiment Overview

Dataset: FashionMNIST

Test images: 10,000

Models:
- Fully Connected Autoencoder (FC-AE)
- Variational Autoencoder (VAE)

Training epochs: 20

Latent dimension: 32

## 2. Overall Reconstruction Results

| Model | Test MSE |
|---|---|
| FC-AE | 0.010034 |
| VAE | 0.014975 |

FC-AE achieved lower reconstruction MSE.

The VAE additionally optimizes KL divergence to regularize
its latent distribution, introducing a trade-off between
reconstruction fidelity and generative capability.

## 3. Class-Wise Error Analysis

Highest average FC-AE error:
Bag (0.016613)

Highest average VAE error:
Sandal (0.022542)

Lowest average FC-AE error:
Trouser (0.004761)

Lowest average VAE error:
Trouser (0.007871)

## 4. Worst Reconstruction Examples

The six highest-error images for each model were identified
using per-image pixel MSE.

### FC-AE

| Model   |   Rank |   Test Index | Class       |   Reconstruction MSE |
|:--------|-------:|-------------:|:------------|---------------------:|
| FC-AE   |      1 |         9067 | Pullover    |            0.0872294 |
| FC-AE   |      2 |         3953 | T-shirt/top |            0.0863876 |
| FC-AE   |      3 |         7348 | Shirt       |            0.0858735 |
| FC-AE   |      4 |         1286 | Bag         |            0.0804554 |
| FC-AE   |      5 |         4392 | Trouser     |            0.0764981 |
| FC-AE   |      6 |         3485 | T-shirt/top |            0.0753651 |

### VAE

| Model   |   Rank |   Test Index | Class       |   Reconstruction MSE |
|:--------|-------:|-------------:|:------------|---------------------:|
| VAE     |      1 |         1286 | Bag         |            0.0982779 |
| VAE     |      2 |         6191 | T-shirt/top |            0.0962528 |
| VAE     |      3 |         4193 | T-shirt/top |            0.0957821 |
| VAE     |      4 |         9067 | Pullover    |            0.0931269 |
| VAE     |      5 |         3953 | T-shirt/top |            0.0893421 |
| VAE     |      6 |         7348 | Shirt       |            0.0881799 |

## 5. Failure Categories and Diagnoses

### Category A: Loss of Fine Details

High-frequency image details, such as stripes, patterns,
and edges, can be smoothed during reconstruction.

Diagnosis:
The compressed latent representation may not preserve
all fine-grained visual information.

### Category B: Unusual Clothing Structures

Several high-error examples contain distinctive shapes
or uncommon clothing configurations.

Diagnosis:
The models may reconstruct common clothing structures
more accurately than unusual examples.

### Category C: Shape and Texture Distortion

Some reconstructed images preserve the broad silhouette
but lose internal structure and texture.

Diagnosis:
Pixel-based reconstruction objectives do not explicitly
guarantee preservation of perceptual details.

## 6. Model Comparison

FC-AE:
- Lower full-test reconstruction MSE.
- Directly optimized for pixel reconstruction.
- Does not explicitly regularize its latent distribution.

VAE:
- Higher reconstruction MSE.
- Learns a regularized probabilistic latent representation.
- Supports generating images through latent sampling.

## 7. Limitations

1. Evaluation uses FashionMNIST only.
2. Both models use fully connected architectures.
3. Results come from one training seed.
4. MSE does not fully capture perceptual image quality.
5. VAE reconstruction uses the latent mean rather than
   stochastic latent sampling.

## 8. Conclusion

The FC-AE achieved better pixel-level reconstruction
accuracy on FashionMNIST.

The VAE traded some reconstruction accuracy for a
regularized latent representation that supports sampling.

The worst reconstruction examples suggest that unusual
shapes and fine visual details remain challenging.

These findings are specific to the tested architectures,
training settings, and dataset.
