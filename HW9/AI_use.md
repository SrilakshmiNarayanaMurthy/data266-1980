
# DATA 266 — Homework 9
## AI Use Disclosure

### 1. AI Tool Used

**Tool:** ChatGPT (OpenAI)

ChatGPT was used as a coding, debugging, explanation,
and documentation assistant during the implementation
of Homework 9.

### 2. Purpose of AI Assistance

AI assistance was used to understand and implement
the latent diffusion workflow, including:

- Pretrained Stable Diffusion VAE encoding and decoding.
- Latent scaling and forward diffusion.
- Diffusion noise schedules.
- Sinusoidal timestep embeddings.
- Residual blocks and self-attention.
- Conditional U-Net architecture.
- Classifier-free guidance.
- DDIM sampling.
- Training metrics and visualizations.
- Image-quality evaluation.
- CFG comparison and interpretation.
- Experiment reporting and documentation.

### 3. How AI Was Used

ChatGPT provided step-by-step implementation guidance
and suggested Python/PyTorch code.

The code was executed in Google Colab using an
NVIDIA Tesla T4 GPU.

Execution logs, tensor dimensions, model parameter
counts, training losses, and generated images were
reviewed during implementation.

The experiment was conducted using FashionMNIST
with personalized seed 1980.

### 4. AI-Assisted Debugging and Corrections

#### 4.1 Self-Attention Parameter Count

An initial AI explanation incorrectly predicted
the parameter count of the attention block.

The executed implementation reported 66,304
parameters.

The verified count was used in the final results.

#### 4.2 Interactive Visual Evaluation

An initial visual evaluation implementation used
repeated input() calls.

This caused the Colab cell to wait for manual input.

The approach was revised to use explicit ratings
and noninteractive report generation.

#### 4.3 Image Quality Interpretation

Pixel-level MSE statistics alone were insufficient
to determine whether the generated clothing images
were recognizable.

Visual review was added to identify blurry regions,
shape distortions, artifacts, and recognizability.

#### 4.4 CFG Scale Selection

Three classifier-free guidance scales were tested:

- CFG 1.0
- CFG 3.0
- CFG 5.0

AI assistance was used to compare and interpret
the generated images.

The final selection of CFG 3.0 was based on
qualitative visual review rather than pixel MSE alone.

### 5. Experimental Work and Verification

The experiment involved:

- Loading and preprocessing FashionMNIST.
- Encoding images with a pretrained VAE.
- Implementing the conditional latent diffusion model.
- Training for five epochs.
- Saving model checkpoints.
- Generating images using DDIM.
- Evaluating generated image quality.
- Comparing classifier-free guidance scales.
- Documenting observed failure patterns.

The reported numerical results were obtained from
the executed Colab experiment rather than invented
or estimated by AI.

### 6. AI Limitations

AI-generated code and explanations may contain
errors and require verification.

The following limitations apply to the experiment:

- Training was limited to five epochs.
- Visual evaluation used eight generated images.
- Image-quality ratings were subjective.
- No external classification accuracy was measured.
- No FID score was calculated.
- Some generated samples remained blurry or distorted.

### 7. Responsibility and Disclosure

AI was used to assist with code development,
debugging, explanations, and documentation.

The experiment was executed in Google Colab, and
the reported outputs were checked against the
observed results.

The final submission is intended to reflect the
actual implementation, measurements, and limitations
of the completed experiment.

The student remains responsible for reviewing the
code, results, interpretations, and submitted work.
