# Sprint 2 — Advanced Retrieval

## Goal

Improve the baseline RAG retrieval pipeline and evaluate different retrieval strategies using measurable information retrieval metrics.

---

## 1. Semantic Chunking

Implemented a custom semantic chunker based on sentence embeddings and cosine similarity.

Adjacent sentences are compared:

- high similarity → keep in the same chunk
- low similarity → create a semantic boundary

The final threshold used during experimentation was:

```text
minimum_similarity = 0.30
```

This value is empirical and depends on the embedding model and dataset.

### Recursive Fallback

Semantic chunks can still become too large.

Therefore the pipeline uses:

```text
Document
↓
Semantic Chunking
↓
Semantic Chunk
↓
If chunk exceeds max size
↓
RecursiveCharacterTextSplitter
```

Semantic chunking determines topic boundaries.

Recursive chunking acts as a size safety mechanism.

---

## 2. Recursive vs Semantic Chunking

Recursive chunking produced fewer but less topic-focused chunks.

Semantic chunking separated:

- destination/hotel information
- cancellation information
- insurance information

more clearly.

However, visual coherence alone does not prove better retrieval quality.

Retrieval quality must be evaluated using metrics.

---

## 3. Retrieval Evaluation

A golden evaluation dataset was created.

Metrics implemented:

### Hit@K / Recall@K

Checks whether the expected source appears in the first K results.

### MRR

Measures the rank of the first relevant result.

```text
rank 1 → 1.0
rank 2 → 0.5
rank 3 → 0.33
```

### Precision@K

Measures how many retrieved results are relevant.

### nDCG@K

Measures ranking quality while rewarding relevant results appearing higher in the ranking.

---

## 4. Metadata Filtering

Metadata filtering was implemented for retrieval experiments.

Example metadata:

```text
tenant_id
language
document_type
source
```

Important production distinction:

Local experiments may filter candidates in Python.

Production systems should use database-side filtering, for example Qdrant payload filters.

Metadata filtering can also be part of the security boundary, especially for tenant isolation.

---

## 5. BM25 Sparse Retrieval

Implemented BM25 lexical retrieval using `rank-bm25`.

BM25 is useful when exact terms matter.

Examples:

```text
non-refundable
Belek
loyalty program
```

Dense retrieval and BM25 solve different problems:

```text
Dense → semantic similarity
BM25  → lexical similarity
```

---

## 6. Hybrid Retrieval

Combined:

```text
Dense Retrieval
+
BM25 Retrieval
```

using Reciprocal Rank Fusion.

Architecture:

```text
Query
├── Dense Retrieval
└── BM25 Retrieval
        ↓
       RRF
        ↓
Final ranking
```

---

## 7. Reciprocal Rank Fusion

RRF combines ranking positions rather than raw retrieval scores.

Example:

```text
score = 1 / (k + rank)
```

Raw BM25 and cosine similarity scores should not be directly summed because they operate on different scales.

If the same chunk appears in multiple rankings, its RRF contributions are added together.

---

## 8. Cross-Encoder Reranking

Implemented:

```text
Hybrid Retrieval
↓
Candidate Pool
↓
Cross-Encoder
↓
Final Top-K
```

Retriever responsibility:

```text
find good candidates
maximize recall
```

Reranker responsibility:

```text
accurately reorder candidates
```

The CrossEncoder evaluates query-document pairs jointly and is more expensive than dense retrieval, so it is only applied to a small candidate set.

---

## 9. Query Rewriting

Implemented LLM-based query rewriting.

Example:

```text
Can I get my money back?
```

may become a more explicit retrieval query.

Risk:

Query rewriting can introduce intent drift.

Therefore rewriting should not automatically be enabled for every query.

---

## 10. Multi-Query Retrieval

Implemented generation of multiple query variants.

Architecture:

```text
Original Query
↓
Generate Alternative Queries
↓
Retrieve for Each Query
↓
Fuse Rankings
↓
Rerank
```

The original query is retained as a safety mechanism.

Recommended usage:

```text
simple query
→ normal hybrid retrieval

ambiguous / difficult query
→ multi-query retrieval
```

Query rewriting and multi-query retrieval are alternatives rather than mandatory sequential stages.

---

## 11. Contextual Retrieval

Implemented ingestion-time contextualization.

Correct architecture:

```text
INGESTION

Document
↓
Chunk
↓
Generate short document-aware context
↓
Context + Chunk
↓
Embed
↓
Store
```

Retrieval remains:

```text
Query
↓
Embed
↓
Search indexed contextualized representations
```

Therefore:

> Contextual Retrieval is an ingestion-time enrichment technique whose effect is measured at retrieval time.

Context generation uses both:

```text
full document
+
current chunk
```

It does not replace good chunking.

Contextual Retrieval showed no measurable improvement on the initial small evaluation dataset.

---

## 12. LangSmith Observability

LangSmith tracing was integrated.

Custom retrieval stages can be observed using `@traceable`.

Examples:

```text
hybrid_retrieval
reranking
```

LangSmith provides:

- traces
- inputs / outputs
- latency
- experiment tracking
- evaluator results

Tracing answers:

> What happened during execution?

Evaluation answers:

> How well did it perform?

Experiments answer:

> Is architecture A better than architecture B?

---

## 13. LangSmith Retrieval Experiments

Three retrieval architectures were compared on the same evaluation dataset.

### Dense Baseline

```text
Hit@3       1.00
MRR         0.96
nDCG@3      0.97
Precision@3 0.64
```

### Hybrid + RRF

```text
Hit@3       1.00
MRR         0.94
nDCG@3      0.94
Precision@3 0.58
```

### Hybrid + Cross-Encoder Reranker

```text
Hit@3       1.00
MRR         1.00
nDCG@3      1.00
Precision@3 0.61
```

---

## 14. Experiment Conclusions

All retrieval architectures successfully placed the expected source inside the Top-3.

Dense retrieval alone performed strongly.

Hybrid retrieval using RRF did not improve ranking quality on this dataset.

The Cross-Encoder reranker produced the best ranking results:

```text
MRR = 1.00
nDCG@3 = 1.00
```

Therefore the strongest tested architecture was:

```text
Dense
+
BM25
↓
RRF
↓
Candidate Pool
↓
Cross-Encoder Reranker
↓
Final Top-K
```

However, the experiment does not prove that every component always improves retrieval.

In particular, BM25 + RRF alone performed slightly worse than the dense baseline.

This demonstrates why retrieval architecture decisions should be evaluation-driven rather than based only on theoretical expectations.

---

## 15. Current Retrieval Architecture

```text
OFFLINE / INGESTION

Documents
↓
Loader + Metadata
↓
Chunking
↓
Optional Contextualization
↓
Dense Embeddings
↓
Vector Index

BM25 lexical index
```

```text
ONLINE / RETRIEVAL

User Query
↓
Optional Multi-Query
↓
Dense Retrieval + BM25
↓
RRF
↓
Candidate Pool
↓
Cross-Encoder Reranker
↓
Final Top-K
↓
LLM
↓
Answer + Deterministic Sources
```

---

## Key Engineering Lessons

1. More retrieval components do not automatically improve quality.
2. Retrieval changes must be measured.
3. Dense and sparse retrieval provide complementary signals.
4. RRF combines rankings without requiring comparable raw scores.
5. Rerankers improve precision at the final candidate stage.
6. Metadata filters can be both retrieval constraints and security controls.
7. Contextual Retrieval belongs to ingestion, not the online retrieval step.
8. Small evaluation datasets can hide meaningful differences.
9. RAG evaluation should separate retrieval quality from generation quality.
10. Production architecture should be justified by measured trade-offs.
