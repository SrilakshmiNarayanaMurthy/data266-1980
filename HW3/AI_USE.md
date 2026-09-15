# AI Use Disclosure – Homework 3

## AI Assistance Used

AI assistance was used for:


- refining prompt-engineering examples for Question 1.
- Assisting with the PyTorch implementation of self-attention and causal masking for Question 2.
- Debugging code errors during implementation.
- Explaining and interpreting the attention matrices, training results, and heatmaps.
- Organizing the experimental findings for the final report.

## Specific Incorrect AI Assistance

During the causal masked self-attention implementation, the initial code attempted to train `model_masked` before an instance of the causal attention model had been created. This resulted in the following error:

`NameError: name 'model_masked' is not defined`

## How the Error Was Identified

The issue was identified from the Python traceback produced when the causal model training cell was executed. The causal model class had been defined successfully, but no model object named `model_masked` had yet been instantiated.

## Correction Made

The issue was corrected by explicitly creating an instance of the causal self-attention model before starting training:

`model_masked = CausalSelfAttentionModel(vocab_size=vocab_size, embed_dim=embed_dim).to(device)`

After this correction, the model trained successfully.

## Verification

All experiments were executed and their outputs were inspected. The causal masking implementation was additionally verified numerically by measuring attention assigned to future token positions. The maximum future-token attention was `0.0`, confirming that the causal mask correctly prevented attention to future positions.
