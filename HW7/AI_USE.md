# Homework 7 - AI Use Disclosure

## 1. How I Used AI

I used ChatGPT to assist with:
- Structuring the PyTorch implementation.
- Implementing FC-AE and VAE architectures.
- Defining reconstruction and KL divergence losses.
- Preparing training and evaluation code.
- Creating visualizations and metric reports.
- Debugging Google Drive path errors.
- Organizing experiment documentation.

My own work included:
- Executing the notebook in Google Colab.
- Training both models on FashionMNIST.
- Checking training outputs and model behavior.
- Comparing reconstruction metrics.
- Reviewing generated figures and failure examples.

I reviewed and tested the AI-assisted code rather than
assuming that every suggested implementation was correct.

## 2. Specific AI-Assisted Error

An earlier visualization cell attempted to use HW7_DIR
before the variable was available.

Actual error:

NameError: name 'HW7_DIR' is not defined

A subsequent folder verification also failed:

AssertionError: HW7 folder not found.
Mount Google Drive first.

## 3. How I Identified the Error

The notebook stopped when attempting to create the
figures directory.

The error traceback identified the missing HW7_DIR
variable. The directory assertion then showed that
the expected Google Drive path was not accessible.

## 4. What I Changed and Why

I verified the Google Drive connection and corrected
the working directory to:

/content/drive/MyDrive/GEN AI HWS/HW7

I then defined HW7_DIR before creating output folders.

The corrected code successfully saved figures and
experiment artifacts to the intended directory.

## 5. Verification

I verified the corrected workflow by checking:
- Successful FC-AE and VAE training.
- Finite reconstruction losses.
- Correct model input and output shapes.
- Saved checkpoints and metrics files.
- Full-test reconstruction comparison.
- Saved reconstruction and error-analysis figures.

## 6. Limitations

The experiments use one training seed and fixed
architectures. Reconstruction MSE measures pixel-level
accuracy but does not fully describe perceptual quality.

The VAE also optimizes KL divergence, so its total
training objective should not be directly compared
against the FC-AE MSE objective.

## Disclosure

This document describes AI-assisted work and observed
errors. The student should review and revise this
description to accurately reflect their own contributions.
