# AI_USE.md — Homework 6

## 1. Parts for which I used AI assistance

I used ChatGPT to help organize the notebook into the required assignment
sections, clarify the implementation steps, review code structure, and verify
that the outputs matched the assignment requirements.

I used assistance while implementing:
- the supervised ResNet-18 baseline,
- rotation-based self-supervised pretraining,
- frozen-encoder linear evaluation,
- SimCLR-style contrastive pretraining,
- NT-Xent loss,
- cosine-similarity nearest-neighbor retrieval,
- result summaries and experiment documentation.

I ran all experiments in Google Colab and verified the actual outputs produced
by the notebook.

## 2. Specific AI-generated suggestion that I verified

For the SimCLR section, ChatGPT suggested implementing NT-Xent loss by
L2-normalizing the projection vectors, computing pairwise similarities using
a matrix multiplication, masking self-similarities, and using the matching
augmented view as the positive target.

I verified this implementation by running it on an actual batch. The two
projection tensors had shape [128, 128], cosine similarity was used with
temperature tau = 0.2, and the loss was finite. The full SimCLR training loss
then decreased from 4.3528 in the first epoch to 2.2109 in the final epoch.

## 3. How I checked the AI-assisted work

I did not assume that the generated code was correct. I checked the tensor
shapes, dataset sizes, trainable parameter counts, frozen-encoder status,
training losses, accuracies, and nearest-neighbor outputs after running each
section.

For example, before linear evaluation for both self-supervised methods, I
verified that the number of trainable encoder parameters was 0. I also
verified that all three methods used the same 500-image labeled subset for
their downstream evaluation.

## 4. What I learned or changed after verification

The experiments showed that the model with the best classification accuracy
did not necessarily have the highest nearest-neighbor class agreement for the
three selected query images.

SimCLR achieved the best final classification accuracy at 52.89%, while
Rotation SSL achieved the highest Top-5 same-class neighbor agreement for the
three selected queries at 73.33%. Therefore, I kept these two measurements
separate in my analysis instead of assuming that the model with the highest
classification accuracy must also have the best nearest-neighbor result.

I also learned the importance of removing the self-supervised task head and
freezing the encoder before linear evaluation so that the downstream result
measures the quality of the learned representation.
