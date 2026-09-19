# 🤖 Enterprise RAG AI Assistant

### Azure AI Enterprise Retrieval-Augmented Generation Platform

> Production-oriented RAG architecture for secure document retrieval, grounded answers, semantic reranking, and enterprise-scale query handling.

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![Azure OpenAI](https://img.shields.io/badge/Azure%20OpenAI-GPT--4o-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white)](#)
[![Azure AI Search](https://img.shields.io/badge/Azure%20AI%20Search-Hybrid%20Retrieval-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white)](#)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](#)

[🚀 Quick Start](#-quick-start) · [🏗️ Architecture](#️-architecture) · [🧪 Evaluation](#-evaluation) · [🔍 RAG Failure Handling](#-rag-failure-handling)

---

## 🎯 Project Overview

This project implements an **enterprise-oriented Retrieval-Augmented Generation (RAG) assistant** using the Microsoft Azure AI stack.

It focuses on practical RAG challenges including versioned documents, ambiguity, multi-document synthesis, retrieval quality, hallucination control, access control, conversational context, latency, scaling, security, and cost.

---

## 📌 At a Glance

| Capability | Implementation |
|---|---|
| **Document ingestion** | PDF extraction + semantic chunking |
| **Chunking** | 400-token window + 80-token overlap |
| **Embeddings** | Azure OpenAI `text-embedding-3-large` |
| **Retrieval** | Dense vector + BM25 hybrid search |
| **Reranking** | Azure AI Search Semantic Ranker |
| **Security** | OData department / ACL filtering |
| **Ambiguity** | Pre-retrieval ambiguity guardrail |
| **Conversation** | GPT-4o standalone query rewriting |
| **Generation** | Azure OpenAI GPT-4o |
| **Grounding** | Context-only answers + citations |
| **Frontend** | Streamlit |

---

## 🏗️ Architecture

```text
                     ENTERPRISE RAG PIPELINE

 Enterprise PDFs
       │
       ▼
 ┌─────────────────┐
 │ PDF Extraction  │
 │     PyPDF       │
 └────────┬────────┘
          │
          ▼
 ┌─────────────────────┐
 │ Semantic Chunking   │
 │ 400 tokens / 80     │
 │ token overlap       │
 └─────────┬───────────┘
           │
           ▼
 ┌─────────────────────────────┐
 │ Azure OpenAI Embeddings     │
 │ text-embedding-3-large      │
 └────────────┬────────────────┘
              │
              ▼
 ┌──────────────────────────────────────┐
 │ Azure AI Search                      │
 │ HNSW Vector + BM25 + ACL Metadata   │
 └──────────────────┬───────────────────┘
                    │
                    │
 User Query ──► Streamlit UI
                    │
                    ▼
           ┌─────────────────┐
           │ Ambiguity Guard │
           └───────┬─────────┘
                   │
                   ▼
           ┌─────────────────┐
           │ Query Rewriter  │
           │     GPT-4o      │
           └───────┬─────────┘
                   │
                   ▼
           ┌─────────────────┐
           │  Hybrid Search  │
           │ Vector + BM25   │
           └───────┬─────────┘
                   │
                   ▼
           ┌─────────────────┐
           │ OData Security │
           │ ACL Filtering   │
           └───────┬─────────┘
                   │
                   ▼
           ┌─────────────────┐
           │ Semantic Ranker │
           └───────┬─────────┘
                   │
                   ▼
           ┌─────────────────┐
           │   GPT-4o LLM    │
           │ Strict Grounding│
           └───────┬─────────┘
                   │
                   ▼
           ┌─────────────────┐
           │ Answer + Source │
           │   Citations     │
           └─────────────────┘
```

---

## 🧠 Key Design Decisions

### Hybrid Retrieval

Combines **dense vector search** with **BM25 keyword search** to support both semantic relevance and exact matching for identifiers, codes, dates, and policy terms.

### Semantic Reranking

Top retrieval candidates are re-ranked using Azure AI Search Semantic Ranker before context is passed to the LLM.

### Security Filtering

Department/access metadata is attached to indexed content and enforced with **OData filters** during retrieval.

### Strict Grounding

Generation uses a strict context-only approach with **temperature = 0.0** and explicit fallback handling when evidence is insufficient.

---

## 🔍 RAG Failure Handling

| Scenario | Engineering approach |
|---|---|
| Correct document, wrong chunk | Semantic chunking + semantic reranking |
| Information spread across sections | Larger candidate pool + chunk aggregation |
| Conflicting document versions | `effective_year` metadata + OData filters |
| Missing information / hallucination | Strict grounding + fallback response |
| Ambiguous query | Pre-retrieval ambiguity classification |
| Conversational context | GPT-4o standalone query rewriting |

---

## 🛠️ Production Problem Solving

### Retrieval Quality
Hybrid retrieval, Reciprocal Rank Fusion, smaller chunks with overlap, and semantic cross-encoder reranking.

### Latency Optimization
Response streaming, semantic caching, and context-token reduction.

### Scaling
Asynchronous batch ingestion, Azure Data Factory, Azure Document Intelligence, index sharding, and additional Azure AI Search replicas.

### Security
ACL metadata, identity claims, and non-bypassable OData filtering.

### Cost Optimization
Semantic caching, right-sized embedding models, smaller models for simple intents, and prompt-prefix caching.

---

## 🧪 Evaluation

| Metric | Baseline RAG | Improved Azure RAG |
|---|---:|---:|
| **Retrieval Hit Rate** | 50.0% | **100.0%** |
| **Answer Groundedness** | 50.0% | **100.0%** |
| **Hallucination Rate** | High | **0.0%** |
| **Average Latency** | 2.84s | **2.15s** |

> Benchmark values documented in this repository.

---

## 📂 Repository Structure

```text
Enterprise-RAG-AI-Assistant/
├── documents/
├── app.py
├── rag_app.py
├── ingestion.py
├── eval_rag.py
├── requirements.txt
└── README.md
```

| File | Purpose |
|---|---|
| `ingestion.py` | Document extraction, chunking, and indexing |
| `rag_app.py` | Core retrieval and generation flow |
| `app.py` | Streamlit application interface |
| `eval_rag.py` | Evaluation and benchmarking |
| `requirements.txt` | Dependencies |

---

## 🚀 Quick Start

### 1. Clone

```bash
git clone https://github.com/kumarakshay7/Enterprise-RAG-AI-Assistant.git
cd Enterprise-RAG-AI-Assistant
```

### 2. Environment

```bash
python -m venv venv
```

Windows:

```bash
.\venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

### 3. Install

```bash
pip install -r requirements.txt
```

### 4. Configure Azure

Create a `.env` file with the Azure OpenAI and Azure AI Search settings expected by the application.

### 5. Run

```bash
streamlit run app.py
```

### 6. Evaluate

```bash
python eval_rag.py
```

---

## 💡 Engineering Takeaway

> **Effective enterprise RAG is more than embeddings.**

```text
Ingestion
   ↓
Chunking
   ↓
Embeddings
   ↓
Hybrid Retrieval
   ↓
Security Filtering
   ↓
Semantic Reranking
   ↓
Grounded Generation
   ↓
Evaluation
```

This project demonstrates the complete RAG lifecycle from ingestion and retrieval to security, generation, evaluation, and production-oriented design.

---

<div align="center">

### 🤖 Enterprise RAG · Azure AI · Secure Retrieval · Grounded Generation

**Built by Akshay Kumar**

[🌐 Portfolio](https://kumarakshay7.github.io/akshay-portfolio/) · [💼 LinkedIn](https://www.linkedin.com/in/akshaykumar17/) · [🐙 GitHub](https://github.com/kumarakshay7)

</div>
