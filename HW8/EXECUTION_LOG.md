# HOMEWORK 8 — EXECUTION LOG

## Assignment
Mini Diffusion Model (DDPM to DDIM)

## Environment
- Framework: PyTorch
- GPU: NVIDIA Tesla T4
- Dataset: MNIST
- Training samples: 60,000
- Random seed: 42

## Implementation Steps
1. Environment setup and GPU verification — PASS
2. MNIST dataset loading — PASS
3. Forward diffusion implementation — PASS
4. Forward diffusion visualization — PASS
5. CNN architecture verification — PASS
6. Training configuration verification — PASS
7. DDPM training (50 epochs) — PASS
8. Training loss visualization — PASS
9. DDPM reverse sampling — PASS
10. DDPM image visualization — PASS
11. DDIM sampling (50 and 20 steps) — PASS
12. DDPM versus DDIM comparison — PASS
13. Experimental discussion — Added to notebook
14. Final deliverables verification — PASS
15. AI usage disclosure — Added to notebook

## Training Configuration
- Diffusion steps: 100
- Beta schedule: 0.0001 to 0.02
- CNN convolutional layers: 6
- Batch size: 128
- Epochs: 50
- Learning rate: 0.001
- Optimizer: Adam
- Loss: Mean Squared Error

## Training Results
- Initial epoch loss: 0.155908
- Final epoch loss: 0.058062
- Best epoch: 50
- Training time: 1002.49 seconds
- GPU training hours: 0.2785

## Sampling Results
- DDPM (100 steps): 0.2340 seconds
- DDIM (50 steps): 0.1437 seconds
- DDIM (20 steps): 0.0475 seconds
- Images generated per method: 16
- Image dimensions: 28 x 28

## Observations
The CNN training loss decreased over 50 epochs.
DDIM required less sampling time than DDPM.
Generated images contained distorted and overlapping strokes,
limiting handwritten digit recognizability.

## Verification
All six checked output files were present in HW8_Results.
The notebook contains the experiment code, visualizations,
discussion, and AI usage documentation.
