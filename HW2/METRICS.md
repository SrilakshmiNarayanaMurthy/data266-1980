# Homework 2 - Metrics

## Personal Parameters

- SID4: 1980
- SEED: 1980
- SLICE: 980
- HP_ID: 0
- CLS_A: 0
- CLS_B: 1

---

## Q1 - Embedding-Based Transfer Learning

Pretrained Model: word2vec-google-news-300  
Fine-tuning Dataset: IMDB Movie Reviews

### Original vs. Fine-Tuned Vector Similarity

| Word | Cosine Similarity |
|---|---:|
| cast | 0.6397 |
| score | 0.4856 |
| plot | 0.6455 |
| screen | 0.6661 |
| review | 0.5153 |

Most shifted word: **score** (0.4856)

Least shifted word: **screen** (0.6661)

The lower cosine similarity for "score" indicates that its embedding changed the most after fine-tuning on IMDB movie reviews.

---

## Q2 - Retrieval-Augmented Generation

Documents: 10 Wikipedia movie pages

Embedding Model: all-MiniLM-L6-v2  
Vector Store: FAISS  
LLM: Qwen/Qwen2.5-0.5B-Instruct  
Retriever: Top-3 chunks

### Baseline Chunking

Chunk size: 500  
Chunk overlap: 50  
Total chunks: 2044

### Retrieval Evaluation

| Question | Correct Information in Top-3 | First Relevant Rank | Final Answer Correct |
|---|---|---:|---|
| Shawshank Redemption director | Yes | 2 | Yes |
| Vito Corleone actor | Yes | 2 | No |
| Inception director | Yes | 1 | Yes |
| Interstellar music composer | Yes | 1 | Yes |
| Jurassic Park director | Yes | 2 | No |

Retrieval Success Rate: **5/5 = 100%**

Two generation failures were observed even though the correct information was successfully retrieved:

1. The Godfather question retrieved Marlon Brando, but the LLM produced an incorrect answer involving Michael Corleone/Al Pacino.
2. The Jurassic Park question retrieved Steven Spielberg, but the LLM initially stated that the answer could not be found.

### Alternative Chunking Experiment

Alternative chunk size: 1000  
Alternative overlap: 100  
Total chunks: 1111

For the two failed questions, increasing the chunk size from 500/50 to 1000/100 resulted in correct final answers.

The Jurassic Park response contained the correct final answer, Steven Spielberg, but its explanation also contained an inaccurate statement involving Michael Crichton.

---

## Q3 - Training Optimizations

Hardware: NVIDIA Tesla T4

Controlled settings:

- Input size: 128
- Hidden size: 256
- Output size: 10
- Batch size: 64
- Training steps: 100
- Number of samples: 6400
- Seed: 1980

### Tensor Creation

| Method | Mean Time |
|---|---:|
| CPU creation + GPU transfer | 0.0786 s |
| Direct GPU creation | 0.0106 s |

Direct GPU tensor creation speedup: **7.39x**

### Training Optimization Results

| Method | Mean Time (s) | Peak GPU Memory (MB) | Final Loss |
|---|---:|---:|---:|
| Baseline | 0.1212 | 19.93 | 2.2893 |
| Weight Init - Default | 0.1221 | 18.06 | 2.2893 |
| Weight Init - Xavier | 0.1215 | 18.06 | 2.4267 |
| Activation Checkpointing | 0.2367 | 19.08 | 2.2893 |
| Gradient Accumulation | 0.4048 | 19.06 | 2.2893 |
| Mixed Precision | 0.2974 | 19.35 | 2.2894 |

Direct GPU tensor creation produced the clearest performance improvement. Activation checkpointing, gradient accumulation, and mixed precision did not improve runtime or peak GPU memory for this small model, although their final losses remained close to the baseline.
