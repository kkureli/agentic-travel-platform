# Agentic Travel Platform

A production-oriented AI engineering project for building an agentic travel intelligence platform step by step.

The project is designed to explore and implement modern LLM engineering concepts including RAG, advanced retrieval, embedding fine-tuning, LangGraph orchestration, tool calling, GraphRAG, multi-agent systems, evaluation, observability, and production AI practices.

## Current Status

### Sprint 1 — Baseline RAG ✅

The first sprint implements a complete baseline Retrieval-Augmented Generation pipeline.

## Architecture

### Ingestion

```text
Markdown Documents
        ↓
Document Loader
        ↓
LangChain Documents
        ↓
Recursive Chunking
        ↓
Sentence Transformer
        ↓
Embeddings
        ↓
Qdrant Vector Database
```

### Retrieval

```text
User Question
        ↓
Query Embedding
        ↓
Qdrant Semantic Search
        ↓
Top-K Chunks
        ↓
LLM Context
        ↓
Generated Answer
        ↓
Sources
```

## Implemented Features

- FastAPI application structure
- Environment configuration with Pydantic Settings
- Docker Compose infrastructure
- PostgreSQL
- Qdrant
- Redis
- Neo4j
- Markdown knowledge base
- LangChain document loading
- Recursive text chunking
- Local Sentence Transformer embeddings
- Qdrant vector indexing
- Cosine similarity search
- Semantic retrieval
- Baseline RAG generation
- Source metadata tracking
- Structured RAG responses with Pydantic

## Embedding Model

The current baseline uses:

`sentence-transformers/all-MiniLM-L6-v2`

Embedding dimension:

`384`

The same embedding model is used for both document chunks and user queries so that they exist in the same vector space.

## Tech Stack

### Backend

- Python
- FastAPI
- Pydantic

### LLM / RAG

- LangChain
- Sentence Transformers

### Data

- Qdrant
- PostgreSQL
- Redis
- Neo4j

### Infrastructure

- Docker
- Docker Compose

## Project Structure

```text
app/
├── api/
├── agents/
├── core/
├── knowledge_graph/
├── rag/
│   ├── chunking/
│   ├── embeddings/
│   ├── loaders/
│   └── vector_store/
├── schemas/
├── services/
├── tools/
└── workflows/

data/
├── raw/
└── processed/

tests/
```

## Running the Infrastructure

```bash
docker compose up -d
```

Check running services:

```bash
docker compose ps
```

## Install Dependencies

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project:

```bash
pip install -e .
```

## Run the API

```bash
uvicorn app.main:app --reload
```

API documentation:

`http://localhost:8000/docs`

## Ingest the Knowledge Base

```bash
python -m app.rag.ingestion
```

The ingestion pipeline:

1. Loads Markdown documents.
2. Splits documents into chunks.
3. Generates embeddings for each chunk.
4. Stores text, metadata, and vectors in Qdrant.

## Roadmap

### Sprint 2 — Advanced Retrieval & Evaluation

- Semantic chunking
- Metadata filtering
- BM25 / sparse retrieval
- Hybrid search
- Reciprocal Rank Fusion
- Cross-encoder reranking
- Query rewriting
- Multi-query retrieval
- Contextual Retrieval
- Retrieval evaluation
- LangSmith tracing

### Sprint 3 — Embedding Fine-Tuning

- Positive and negative samples
- Hard negatives
- Contrastive learning
- Sentence Transformer fine-tuning
- Base vs fine-tuned retrieval evaluation

### Sprint 4 — LangGraph & Agentic RAG

- State
- Nodes and edges
- Conditional routing
- Retrieval grading
- Query rewrite loops
- Planning
- Agentic retrieval

### Sprint 5 — Tools & External Systems

- SQL Agent
- Text-to-SQL
- Web Search Agent
- Booking APIs
- CRM APIs
- Multi-source reasoning

### Sprint 6 — GraphRAG & Multi-Agent

- Neo4j knowledge graph
- Graph retrieval
- Vector + graph retrieval
- Agent orchestration
- Multi-agent workflows

### Sprint 7 — Production AI

- Security
- Prompt injection protection
- Human-in-the-loop
- Retry and fallback strategies
- Caching
- LangSmith evaluation
- Agent evaluation
- CI/CD
- Deployment

## Goal

The goal of this project is not only to build an AI application, but to understand the engineering decisions behind production RAG and agentic AI systems:

- Why a retrieval strategy is selected
- How retrieval quality is measured
- When agents are useful
- When deterministic workflows are preferable
- How different data sources should be queried
- How AI systems are evaluated, observed, secured, and deployed
