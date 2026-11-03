
# DATA 266 — Homework 9
## Generated Image Failure Analysis

### 1. Objective

The purpose of this analysis is to identify
limitations and failure patterns in images generated
by the conditional latent diffusion model.

The experiment generated FashionMNIST clothing images
using deterministic DDIM sampling and classifier-free
guidance.

The two selected classes were:

- Class 0: T-shirt/top
- Class 1: Trouser

### 2. Evaluation Method

Eight generated images were manually reviewed.

Four images were generated for each class.

The following rating scale was used:

| Rating | Interpretation |
|---|---|
| 0 | Unrecognizable |
| 1 | Partially recognizable |
| 2 | Clearly recognizable |

The evaluation considered:

- Clothing silhouette
- Class recognizability
- Shape consistency
- Visual artifacts
- Blurriness
- Overall visual quality

### 3. Visual Evaluation Results

| Metric | Result |
|---|---:|
| Total images | 8 |
| Clearly recognizable | 5 |
| Partially recognizable | 3 |
| Unrecognizable | 0 |
| Clear recognition rate | 62.50% |
| Mean quality rating | 1.625 / 2 |

### 4. Class-Level Analysis

#### T-shirt/top

Four T-shirt/top images were evaluated.

Ratings:

- Image 1: 1
- Image 2: 1
- Image 3: 2
- Image 4: 1

Mean quality rating: 1.25 / 2

Clearly recognizable: 1 of 4

Observed limitations:

- Some T-shirt shapes were irregular.
- Sleeve boundaries were not always clear.
- Certain samples contained dark artifacts.
- Some generated images had distorted silhouettes.
- Fine clothing details were limited.

The T-shirt/top class was more difficult for the
trained model to generate consistently.

#### Trouser

Four trouser images were evaluated.

Ratings:

- Image 1: 2
- Image 2: 2
- Image 3: 2
- Image 4: 2

Mean quality rating: 2.00 / 2

Clearly recognizable: 4 of 4

Observed limitations:

- Trouser silhouettes were generally recognizable.
- Some images lacked sharp edges.
- Fine texture was limited.
- Some samples showed similar overall shapes.

The trouser class achieved higher visual
recognizability in this evaluation.

### 5. Pixel-Level Diversity

| Metric | Result |
|---|---:|
| T-shirt/top pairwise MSE | 0.080547 |
| Trouser pairwise MSE | 0.023142 |
| Between-class mean difference | 0.237177 |

The higher pairwise MSE for T-shirt/top indicates
greater pixel-level variation among those samples.

However, this does not necessarily mean that
the T-shirt/top images were more realistic or
perceptually diverse.

### 6. Classifier-Free Guidance Analysis

Three CFG scales were compared using matched
initial noise.

| CFG Scale | Visual Observation |
|---|---|
| 1.0 | More fragmented shapes and artifacts |
| 3.0 | More recognizable clothing silhouettes |
| 5.0 | Similar to CFG 3.0 without clear improvement |

Selected CFG scale: **3.0**

#### Pairwise Differences

| Comparison | Pixel MSE |
|---|---:|
| CFG 1.0 vs 3.0 | 0.046194 |
| CFG 1.0 vs 5.0 | 0.058263 |
| CFG 3.0 vs 5.0 | 0.002903 |

CFG 3.0 and CFG 5.0 produced similar images
for the matched samples.

CFG 1.0 showed more visible distortions.

CFG 3.0 was selected based on qualitative
recognizability and image appearance.

### 7. Main Failure Patterns

#### 7.1 Shape Distortion

Some generated clothing items had irregular
or incomplete silhouettes.

This was particularly noticeable for T-shirt/top.

#### 7.2 Blurriness

Several generated images lacked sharp boundaries
and fine details.

This may reflect limitations in model training,
latent representations, or decoding.

The experiment did not isolate the cause.

#### 7.3 Visual Artifacts

Some images contained dark or inconsistent
regions that did not resemble natural clothing
structures.

#### 7.4 Limited Fine Details

Generated images captured approximate clothing
shapes more successfully than fine texture.

#### 7.5 Uneven Class Quality

Trouser samples were more recognizable than
T-shirt/top samples in the eight-image review.

This observation is limited to the evaluated
samples and does not establish a general
class-level performance difference.

### 8. Potential Improvements

1. Train for additional epochs and compare checkpoints.
2. Evaluate more generated images across random seeds.
3. Test additional CFG values and sampling steps.
4. Use an external classifier to evaluate class consistency.
5. Evaluate perceptual image quality with suitable metrics.
6. Investigate the effect of VAE reconstruction quality.
7. Compare model predictions using held-out examples.
8. Review architecture capacity and training stability.

### 9. Evaluation Limitations

- Only eight generated images were visually evaluated.
- Only four matched samples were compared per CFG scale.
- Ratings were subjective.
- No FID score was calculated.
- No external classifier was used.
- Training was limited to five epochs.
- No statistical confidence interval was calculated
  for visual recognition.

### 10. Conclusion

The conditional latent diffusion model generated
recognizable FashionMNIST clothing silhouettes.

Five of eight images were rated clearly recognizable.

Trouser samples were more consistently recognizable
than T-shirt/top samples in this small evaluation.

The main observed limitations were shape distortion,
blurriness, visual artifacts, and limited fine detail.

CFG 3.0 provided the preferred qualitative results
among the three tested guidance scales.

Additional training and broader evaluation would be
needed to establish stronger image-quality conclusions.
