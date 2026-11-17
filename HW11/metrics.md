# CLIP Evaluation Metrics

## Experimental Setup

| Setting | Value |
|---|---|
| Model | OpenAI CLIP ViT-B/32 |
| Dataset | CIFAR-10 test subset |
| Number of classes | 10 |
| Images per class | 100 |
| Total images | 1,000 |
| Sampling seed | 42 |
| Sampling method | Random, without replacement within each class |
| Image preprocessing | CLIP-provided preprocessing |
| Embedding dimension | 512 |
| Normalization | L2 normalization |
| Similarity metric | Cosine similarity |
| Training / fine-tuning | None |

## Caption–Image Similarity

Caption: **“a photo of a dog.”**

Cosine similarity was calculated as the dot product of normalized
image and caption embeddings. Scores are not probabilities.

| Actual class | Images | Mean | Standard deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|
| dog | 100 | 0.2599 | 0.0108 | 0.2238 | 0.2821 |
| cat | 100 | 0.2371 | 0.0119 | 0.2093 | 0.2751 |
| deer | 100 | 0.2315 | 0.0144 | 0.1859 | 0.2640 |
| frog | 100 | 0.2265 | 0.0121 | 0.1937 | 0.2527 |
| bird | 100 | 0.2222 | 0.0133 | 0.1986 | 0.2662 |
| horse | 100 | 0.2168 | 0.0148 | 0.1699 | 0.2598 |
| airplane | 100 | 0.1998 | 0.0141 | 0.1544 | 0.2355 |
| automobile | 100 | 0.1971 | 0.0106 | 0.1699 | 0.2206 |
| ship | 100 | 0.1914 | 0.0141 | 0.1599 | 0.2280 |
| truck | 100 | 0.1904 | 0.0114 | 0.1621 | 0.2286 |

Nine of the ten highest-scoring images were dogs; one was a cat.
Dog images had the highest mean similarity to the caption.

Visualizations in the notebook include the top and bottom ten matches
with scores and a class-level similarity chart. Error bars show
±1 standard deviation within each class.

## Zero-Shot Classification

Each image was compared with these class prompts:

- a photo of an airplane.
- a photo of an automobile.
- a photo of a bird.
- a photo of a cat.
- a photo of a deer.
- a photo of a dog.
- a photo of a frog.
- a photo of a horse.
- a photo of a ship.
- a photo of a truck.

The predicted class was the prompt with the highest cosine similarity.

| Metric | Result |
|---|---:|
| Evaluated images | 1,000 |
| Correct predictions | 885 |
| Incorrect predictions | 115 |
| Overall accuracy | 88.50% |
| Macro-average per-class accuracy | 88.50% |

The macro-average equals overall accuracy because each class contains
the same number of evaluated images.

### Per-Class Accuracy

Per-class accuracy is the number of correct predictions for an actual
class divided by the number of evaluated images in that class
(equivalent to class recall).

| Class | Total | Correct | Incorrect | Accuracy |
|---|---:|---:|---:|---:|
| airplane | 100 | 92 | 8 | 92% |
| automobile | 100 | 92 | 8 | 92% |
| bird | 100 | 85 | 15 | 85% |
| cat | 100 | 91 | 9 | 91% |
| deer | 100 | 72 | 28 | 72% |
| dog | 100 | 91 | 9 | 91% |
| frog | 100 | 76 | 24 | 76% |
| horse | 100 | 98 | 2 | 98% |
| ship | 100 | 97 | 3 | 97% |
| truck | 100 | 91 | 9 | 91% |

### Confusion Matrix

The full 10 × 10 confusion matrix is displayed in Step 8 of the
notebook. Rows represent actual classes; columns represent predicted
classes. Each row contains 100 images.

The largest reported directional confusions were:

| Actual class | Predicted class | Errors |
|---|---|---:|
| deer | horse | 14 |
| frog | bird | 11 |
| automobile | truck | 8 |
| dog | cat | 6 |
| truck | automobile | 6 |
| deer | bird | 6 |
| cat | dog | 6 |
| bird | cat | 5 |
| deer | dog | 5 |
| bird | dog | 4 |

## Findings and Limitations

- Horse achieved the highest accuracy (98%), followed by ship (97%).
- Deer (72%) and frog (76%) had the lowest accuracies.
- The caption similarity experiment ranked dogs highest on average.
- Shared visual features may contribute to classification errors;
  confusion counts alone do not establish their cause.
- Results apply to the selected 1,000-image subset, not the full test set.
- CIFAR-10's native 32 × 32 resolution limits visual detail.
- Results may vary with the image sample, model, and prompt wording.
