# ============================================================
# HW6 - Create METRICS.md
# ============================================================

metrics_text = """# Homework 6 - METRICS

## Student Parameters
- SID4: 1980
- SEED: 1980
- SLICE: 980
- HP_ID: 0
- CLS_A: 0
- CLS_B: 1

## Dataset
- Dataset: STL-10
- Backbone: ResNet-18
- Pretrained weights: None
- Labeled training subset: 500 images
- Test set: 8,000 images

## Part A - Supervised Learning
- Training images: 500
- Epochs: 15
- Final training accuracy: 85.40%
- Final test accuracy: 39.91%
- Best test accuracy: 40.75%
- Best epoch: 14

## Part B - Rotation Self-Supervised Learning
- Unlabeled pretraining images: 100,000
- Rotation classes: 0, 90, 180, 270 degrees
- Pretraining epochs: 15
- Final rotation accuracy: 89.25%
- Linear evaluation images: 500
- Linear evaluation epochs: 20
- Encoder frozen: Yes
- Final test accuracy: 42.67%
- Best test accuracy: 42.67%
- Best epoch: 20

## Part C - SimCLR
- Unlabeled pretraining images: 20,000
- Pretraining epochs: 20
- Batch size: 128
- Temperature (tau): 0.2
- Projection dimension: 128
- Similarity: Cosine similarity
- Initial contrastive loss: 4.3528
- Final contrastive loss: 2.2109
- Linear evaluation images: 500
- Linear evaluation epochs: 20
- Encoder frozen: Yes
- Projection head used during linear evaluation: No
- Final test accuracy: 52.89%
- Best test accuracy: 53.40%
- Best epoch: 12

## Part D - Nearest-Neighbor Analysis
Three fixed test queries were used:
- Airplane - test index 3
- Bird - test index 5
- Car - test index 24

Top-5 same-class neighbor agreement for these three queries:
- Supervised: 46.67% (7/15)
- Rotation SSL: 73.33% (11/15)
- SimCLR: 33.33% (5/15)

The same query images were used for all three encoders.
Cosine similarity was used for nearest-neighbor retrieval.
The query image itself was excluded from its neighbor list.

## Final Comparison
- Supervised final test accuracy: 39.91%
- Rotation SSL final test accuracy: 42.67%
- SimCLR final test accuracy: 52.89%

SimCLR achieved the highest downstream classification accuracy.
Rotation SSL achieved the highest Top-5 same-class neighbor agreement
for the three selected query images.
"""

with open("METRICS.md", "w") as f:
    f.write(metrics_text)

print("=" * 70)
print("METRICS.md CREATED")
print("=" * 70)

with open("METRICS.md", "r") as f:
    print(f.read())

print("=" * 70)
print("PASS: METRICS.md successfully created.")
