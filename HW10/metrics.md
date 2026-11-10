RAG Retrieval Evaluation Metrics
Experiment setup
Setting	Value
Dataset	ccdv/arxiv-summarization, document, first 100 training papers
Chunk size / overlap	240 / 40 tokens
Fixed-size chunks	5,131
Sentence-aware chunks	5,811
Embeddings	sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
Vector store	ChromaDB, cosine distance
Cross-encoder	cross-encoder/ms-marco-MiniLM-L-6-v2
Baseline retrieval	Top 5
Reranking	Retrieve 10 candidates, rerank to top 5
Evaluation	2 initial questions, 1 conversational follow-up; 38 pooled chunk relevance judgments


Per-query retrieval results
Query	Chunking	Method	Precision@5	Pooled Recall@5	Reciprocal Rank@5
Q1	Fixed	Baseline	1.00	0.8333	1.00
Q1	Fixed	Reranked	0.80	0.6667	1.00
Q1	Sentence-aware	Baseline	1.00	0.8333	1.00
Q1	Sentence-aware	Reranked	0.80	0.6667	1.00
Q2	Fixed	Baseline	0.80	0.8000	1.00
Q2	Fixed	Reranked	0.60	0.6000	1.00
Q2	Sentence-aware	Baseline	0.60	0.6000	1.00
Q2	Sentence-aware	Reranked	0.60	0.6000	1.00
Q3	Sentence-aware	Without memory	0.20	0.3333	0.50
Q3	Sentence-aware	With memory	0.60	1.0000	1.00


Q1: What are the advantages of additive models in statistical learning?

Q2: How do regularized kernel methods improve learning rates?

Q3 follow-up: How do they handle high-dimensional data?
Average results for Q1 and Q2
Chunking	Method	Precision@5	Pooled Recall@5	MRR@5
Fixed	Baseline	0.90	0.8167	1.00
Fixed	Reranked	0.70	0.6334	1.00
Sentence-aware	Baseline	0.80	0.7166	1.00
Sentence-aware	Reranked	0.70	0.6334	1.00


MRR@5 is the mean of reciprocal ranks over Q1 and Q2. The displayed pooled-recall averages are rounded averages of the per-query values.
Conversational memory
Without memory, the follow-up query was How do they handle high-dimensional data?. With memory, it was rewritten to How do additive models in statistical learning handle high-dimensional data?.

Precision@5 increased from 0.20 to 0.60, pooled Recall@5 from 0.3333 to 1.0000, and reciprocal rank from 0.50 to 1.00.
Interpretation and limitations
- Fixed-size baseline retrieval had the highest mean Precision@5 in this small evaluation.
- Cross-encoder reranking changed the rankings but did not improve mean Precision@5 here.
- Conversational memory helped resolve an ambiguous follow-up reference.
- Pooled Recall@5 is calculated against the manually judged retrieved candidate pool, not all relevant chunks in the corpus.
- The evaluation uses only two initial queries and one follow-up. It cannot establish general performance differences.
- Some source text contains @xmath and @xcite placeholders that can reduce evidence readability.
- Relevance labels were manually assigned and subsequently reviewed with AI assistance; the final metrics use the revised labels.
- These are retrieval metrics, not factual-answer generation metrics.
