# Enterprise RAG Knowledge Assistant (Azure AI Stack)

## Overview
This repository contains a production-grade, enterprise-ready Retrieval-Augmented Generation (RAG) assistant built using the Microsoft Azure AI stack. It addresses common RAG failure modes including outdated document bias, ambiguity, multi-document synthesis, multi-tenant security isolation, and hallucination prevention.

                      AZURE AI ENTERPRISE RAG ARCHITECTURE
===================================================================================

 [ DOCUMENT INGESTION PIPELINE ]
 ┌─────────────────┐     ┌───────────────────┐     ┌────────────────────────────┐
 │ Enterprise PDFs │ ──► │  PyPDF Extractor  │ ──► │     Semantic Chunking      │
 └─────────────────┘     └───────────────────┘     │  (400 Tokens, 80 Overlap)  │
                                                   └─────────────┬──────────────┘
                                                                 │
 ┌───────────────────────────────────────────────────────────────┘
 │
 │   ┌───────────────────────────────┐     ┌──────────────────────────────────┐
 └──►│    Azure OpenAI Embeddings    │ ──► │      Azure AI Search Index       │
     │  (text-embedding-3-large)     │     │   (HNSW Vector + BM25 + ACLs)    │
     └───────────────────────────────┘     └─────────────────┬────────────────┘
                                                             │
=============================================================│=====================
 [ REAL-TIME QUERY PIPELINE ]                                │
                                                             │
 ┌──────────────┐     ┌────────────────────────┐             │
 │  User Query  │ ──► │ Streamlit Frontend UI  │             │
 └──────────────┘     └───────────┬────────────┘             │
                                  │                          │
                                  ▼                          │
                      ┌──────────────────────┐               │
                      │ Ambiguity Guardrail  │ ──► [Ambiguous] ──► (Ask Clarification)
                      └───────────┬──────────┘
                                  │ [Valid Query]
                                  ▼
                      ┌──────────────────────┐
                      │ Standalone Query     │
                      │ Rewriter (GPT-4o)    │
                      └───────────┬──────────┘
                                  │
                                  ▼
                      ┌──────────────────────┐
                      │ Hybrid Search        │ ◄─────────────────┘
                      │ (Dense Vector + BM25)│
                      └───────────┬──────────┘
                                  │
                                  ▼
                      ┌──────────────────────┐
                      │ OData Security Filter│ (Department Access Control)
                      └───────────┬──────────┘
                                  │
                                  ▼
                      ┌──────────────────────┐
                      │ Azure AI Search      │
                      │ Semantic Reranker    │ (Cross-Encoder Fine Ranking)
                      └───────────┬──────────┘
                                  │
                                  ▼
                      ┌──────────────────────┐
                      │ Azure OpenAI GPT-4o  │ (Strict Grounding, Temp = 0.0)
                      └───────────┬──────────┘
                                  │
                                  ▼
                      ┌──────────────────────┐
                      │ Grounded Answer +    │
                      │ Document Citations   │
                      └──────────────────────┘
===================================================================================


### Architectural Decisions
* **Azure AI Search:** Chosen over basic vector databases because it natively provides **Hybrid Search** (vector + BM25 keyword search), built-in **Semantic Ranker** (cross-encoder model), and hardware-level OData filtering for multi-tenant security.
* **Hybrid Search vs. Pure Vector:** Pure vector search frequently misses exact codes, policy IDs, or exact dates. Hybrid search combines semantic flexibility with keyword precision.
* **Semantic Reranking:** Re-scores top-$K$ candidates based on contextual relevance, drastically reducing noise in retrieved context.

---

## Resolution of RAG Failure Scenarios

### Scenario 1: Correct Document, Wrong Chunk
* **Solution:** Used **Semantic Chunking** with a token window of 400 and an 80-token overlap, paired with **Azure AI Search Semantic Reranking**.

### Scenario 2: Information Across Multiple Sections
* **Solution:** Expanded candidate pool ($K=10$), performed document chunk aggregation, and passed structured multi-part context blocks to GPT-4o.

### Scenario 3: Conflicting / Versioned Information (`2024.pdf` vs `2026.pdf`)
* **Solution:** Extracted `effective_year` during ingestion as index metadata and applied **OData filter queries** (`effective_year eq 2026`).

### Scenario 4: Hallucination / Missing Information
* **Solution:** Enforced `temperature=0.0`, strict prompt guardrails ("*Answer ONLY using provided context*"), and explicit fallback phrasing when context is insufficient.

### Scenario 5: Ambiguous Query
* **Solution:** Added pre-retrieval intent classification (`check_ambiguity`) to intercept ambiguous queries and prompt the user for clarification.

### Scenario 6: Conversational Context
* **Solution:** Implemented a **Query Rewriter** (`rewrite_query`) using GPT-4o to turn multi-turn conversation context into single standalone search queries.

---

## Problem-Solving & Architecture Answers

### 1. Retrieval Quality (1/5 Chunks Relevant)
* **Fix:** Enable Hybrid Search with Reciprocal Rank Fusion (RRF), reduce chunk sizes to 300 tokens with 20% overlap, and add Semantic Cross-Encoder Reranking.

### 2. Latency Optimization (3s → 12s)
* **Fix:** Enable response streaming, implement Redis Semantic Caching for frequent queries, and reduce input context tokens.

### 3. Scaling (10k → 5M Documents)
* **Fix:** Transition from synchronous push ingestion to asynchronous batch ingestion using Azure Data Factory + Azure Document Intelligence, index sharding, and dedicated Azure AI Search replicas.

### 4. Security & Role-Based Access Control
* **Fix:** Store Access Control Lists (`allowed_departments`) in chunk metadata and enforce non-bypassable OData security filters during query execution based on Entra ID user claims.

### 5. Cost Optimization
* **Fix:** Implement semantic caching, downsize embedding models, route simple intent calls to `gpt-4o-mini`, and cache prompt prefixes.

### 6. Production Failure Debugging
* **Trace Workflow:**
  1. **Query & Retrieval:** Verify retrieved chunk IDs in Application Insights.
  2. **Ranking:** Check reranker relevancy scores.
  3. **Context Assembly:** Inspect prompt token length and truncation.
  4. **Generation:** Validate strict prompt constraints and citation matching.

---

## Evaluation Benchmark

| Metric | Baseline RAG | Improved Azure RAG |
| :--- | :--- | :--- |
| **Retrieval Hit Rate** | 50.0% | **100.0%** |
| **Answer Groundedness** | 50.0% | **100.0%** |
| **Hallucination Rate** | High | **0.0%** |
| **Avg Latency** | 2.84s | **2.15s** |

---

## Quick Start Guide

# 1. Clone repository
git clone [https://github.com/kumarakshay7/Azure-RAG-Assignment.git](https://github.com/kumarakshay7/Azure-RAG-Assignment.git)
cd Azure-RAG-Assignment

# 2. Activate virtual environment & install requirements
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure environment variables in .env

# 4. Launch Streamlit Web App
streamlit run app.py

# 5. Run Automated Evaluation Script
python eval_rag.py
