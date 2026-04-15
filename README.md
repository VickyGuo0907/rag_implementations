# RAG Technique Summary
### A Senior Architect's Reference Implementation

> Comprehensive guide and production-quality implementations of 14 RAG techniques  
> using **LangChain** and **LlamaIndex** with **LMStudio** (local LLM).  
> Based on: [NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques)

---

## Project Structure

```
RAG_technique_summary/
├── config/
│   └── config.yaml                    # ← All settings live here
│
├── core/                              # Shared infrastructure
│   ├── base_rag.py                    # Abstract base class for all RAG
│   ├── config_loader.py               # Singleton YAML config loader
│   ├── llm_client.py                  # LMStudio LLM factory
│   ├── embeddings.py                  # Embedding model factory
│   ├── vector_store.py                # Vector store factory
│   └── document_loader.py             # Document loading & chunking
│
├── techniques/                        # 14 RAG implementations
│   ├── 01_naive_rag/                  ✅ Full implementation
│   │   ├── langchain_impl.py
│   │   ├── llamaindex_impl.py
│   │   └── README.md                  # Flowchart, use cases, pros/cons
│   ├── 02_advanced_rag/               ✅ Full implementation
│   ├── 03_hyde_rag/                   ✅ Full implementation
│   ├── 04_query_transform_rag/        🔧 Stub — pattern established
│   ├── 05_fusion_rag/                 🔧 Stub
│   ├── 06_parent_document_rag/        🔧 Stub
│   ├── 07_reranking_rag/              🔧 Stub
│   ├── 08_self_rag/                   🔧 Stub
│   ├── 09_corrective_rag/             🔧 Stub
│   ├── 10_adaptive_rag/               🔧 Stub
│   ├── 11_graph_rag/                  🔧 Stub
│   ├── 12_raptor_rag/                 🔧 Stub
│   ├── 13_agentic_rag/                🔧 Stub
│   └── 14_multimodal_rag/             🔧 Stub
│
├── evaluation/
│   ├── ragas_evaluator.py             # RAGAS evaluation framework
│   └── __init__.py
│
├── data/
│   └── sample_docs/                   # Sample documents for testing
│
├── scripts/
│   └── run_technique.py               # CLI runner for any technique
│
├── requirements.txt
└── README.md                          # ← You are here
```

---

## Quick Start

### 1. Prerequisites

```bash
# Install LMStudio: https://lmstudio.ai
# Load a model (e.g., llama-3.1-8b-instruct) and start the server on port 1234

# Clone / unzip this project
cd RAG_technique_summary

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

Edit `config/config.yaml` — the minimum you need to change:

```yaml
lmstudio:
  model: "your-model-name"    # e.g., "llama-3.1-8b-instruct"

embeddings:
  model: "nomic-embed-text-v1.5"  # Must be loaded in LMStudio
```

### 3. Run Naive RAG

```bash
# Interactive mode (default sample docs)
python scripts/run_technique.py --technique naive_rag --framework langchain

# Single query
python scripts/run_technique.py \
  --technique naive_rag \
  --framework langchain \
  --docs ./data/sample_docs \
  --query "What is RAG and how does it work?"

# LlamaIndex version
python scripts/run_technique.py --technique naive_rag --framework llamaindex

# With RAGAS evaluation
python scripts/run_technique.py --technique naive_rag --evaluate
```

### 4. Run from Python

```python
from core.config_loader import ConfigLoader
from techniques.naive_rag.langchain_impl import NaiveRAGLangChain

cfg = ConfigLoader.get()
rag = NaiveRAGLangChain(config=cfg._config)

rag.index([
    "RAG combines retrieval systems with language models...",
    "The key components are: indexer, vector store, retriever, and LLM...",
])

result = rag.query("What are the components of a RAG system?")
result.print_summary()
```

---

## RAG Technique Overview & Decision Guide

### Technique Comparison Matrix

| # | Technique | Complexity | Latency | Accuracy | Status |
|---|-----------|:----------:|:-------:|:--------:|:------:|
| 01 | Naive RAG | ⭐ | 🟢 Low | 🟡 Moderate | ✅ Done |
| 02 | Advanced RAG | ⭐⭐⭐ | 🟡 Medium | 🟢 High | ✅ Done |
| 03 | HyDE RAG | ⭐⭐ | 🟡 Medium | 🟢 High | ✅ Done |
| 04 | Query Transform RAG | ⭐⭐ | 🟡 Medium | 🟢 High | 🔧 Stub |
| 05 | Fusion RAG | ⭐⭐⭐ | 🟡 Medium | 🟢 High | 🔧 Stub |
| 06 | Parent Document RAG | ⭐⭐ | 🟢 Low | 🟢 High | 🔧 Stub |
| 07 | Reranking RAG | ⭐⭐ | 🟡 Medium | 🟢 High | 🔧 Stub |
| 08 | Self-RAG | ⭐⭐⭐⭐ | 🔴 High | 🟢 High | 🔧 Stub |
| 09 | Corrective RAG | ⭐⭐⭐⭐ | 🔴 High | 🟢 Very High | 🔧 Stub |
| 10 | Adaptive RAG | ⭐⭐⭐ | 🟡 Varies | 🟢 High | 🔧 Stub |
| 11 | GraphRAG | ⭐⭐⭐⭐⭐ | 🔴 High | 🟢 Very High | 🔧 Stub |
| 12 | RAPTOR | ⭐⭐⭐⭐⭐ | 🔴 High | 🟢 Very High | 🔧 Stub |
| 13 | Agentic RAG | ⭐⭐⭐⭐⭐ | 🔴 High | 🟢 Very High | 🔧 Stub |
| 14 | Multi-modal RAG | ⭐⭐⭐⭐ | 🔴 High | 🟢 High | 🔧 Stub |

---

### Decision Flowchart: Which RAG Should I Use?

```
START: What is your primary requirement?
│
├─ 🚀 SPEED — Is latency the top priority?
│   └─ → Use Naive RAG (01) or Parent Document RAG (06)
│
├─ 🎯 ACCURACY — Need high accuracy in production?
│   ├─ Standard Q&A over documents → Advanced RAG (02) + Reranking (07)
│   ├─ Vocabulary mismatch (users vs. docs) → HyDE RAG (03)
│   ├─ Ambiguous / multi-faceted queries → Fusion RAG (05)
│   └─ Must verify and self-correct → Self-RAG (08) or CRAG (09)
│
├─ 🔗 RELATIONSHIPS — Do you need multi-hop reasoning?
│   ├─ Entity relationships, knowledge graphs → GraphRAG (11)
│   └─ Hierarchical doc structure (books, reports) → RAPTOR (12)
│
├─ 🤖 AUTOMATION — Complex, multi-step tasks?
│   └─ → Agentic RAG (13)
│
├─ 📸 MULTIMODAL — Documents with images/tables?
│   └─ → Multi-modal RAG (14)
│
├─ 💰 COST — Mixed query types, want to save tokens?
│   └─ → Adaptive RAG (10) — routes simple queries to no-retrieval
│
└─ 🧪 PROTOTYPING — Just getting started?
    └─ → Start with Naive RAG (01), baseline everything, then upgrade
```

---

### Use Case → Recommended RAG Technique

| Use Case | Primary | Secondary |
|----------|---------|-----------|
| Customer support FAQ | Naive RAG or Advanced RAG | Reranking RAG |
| Enterprise knowledge base | Advanced RAG | Fusion RAG |
| Legal document research | Corrective RAG | Reranking RAG |
| Medical Q&A | Self-RAG | Corrective RAG |
| Technical documentation | HyDE RAG | Advanced RAG |
| Research paper assistant | RAPTOR | GraphRAG |
| Multi-source intelligence | Agentic RAG | Fusion RAG |
| Product catalog with images | Multi-modal RAG | Advanced RAG |
| Financial reports | GraphRAG | RAPTOR |
| Real-time information | Corrective RAG (web search) | Agentic RAG |
| Code documentation | HyDE RAG | Advanced RAG |
| Teaching / EdTech | Adaptive RAG | Self-RAG |

---

## Architecture Principles (Senior Architect Notes)

### 1. The Abstraction Layer
Every technique inherits from `BaseRAG`, which enforces a consistent interface:
- `index(documents)` — always called before querying
- `query(question)` → `RAGResult` — standardized output with sources + metadata
- `get_info()` — self-describing metadata

This means you can swap techniques transparently in any application:
```python
rag: BaseRAG = NaiveRAGLangChain(config)  # or AdvancedRAGLangChain, HyDERAGLangChain...
rag.index(docs)
result = rag.query("What is the capital of France?")
```

### 2. Configuration-First Design
Zero hardcoded values. Every parameter lives in `config/config.yaml`:
- LLM settings (model, temperature, timeout)
- Embedding model
- Vector store provider
- RAG-specific parameters (chunk_size, top_k, reranker model, etc.)

Change `lmstudio.model` once → all techniques use the new model.

### 3. Dual-Framework Support
Every technique has LangChain and LlamaIndex implementations:
- **LangChain**: Better for complex chains, LCEL, custom agents, production pipelines
- **LlamaIndex**: Better for out-of-the-box RAG pipelines, data connectors, structured data

### 4. The Evaluation Loop
Always evaluate before and after adding complexity:
```
Naive RAG baseline → Measure RAGAS scores → Add Advanced RAG → Measure again → ...
```
The `RAGASEvaluator` makes this easy and consistent across all techniques.

### 5. Implementing a New Technique
1. Create `techniques/XX_name/` directory
2. Create `langchain_impl.py` inheriting from `BaseRAG`
3. Set `TECHNIQUE_NAME` and `FRAMEWORK` class attributes
4. Implement `_build_pipeline()`, `index()`, and `_query()`
5. Create `README.md` with flowchart, use cases, and pros/cons
6. Register in `techniques/__init__.py`

---

## Configuration Reference

See `config/config.yaml` for the full configuration with inline comments.

Key sections:
- `lmstudio` — LMStudio server connection and model settings
- `embeddings` — embedding model selection
- `vector_store` — ChromaDB / FAISS / Qdrant selection
- `document` — chunk size, overlap, chunking strategy
- `retrieval` — top_k, search type, reranking settings
- `evaluation` — RAGAS metrics configuration
- `rag_techniques` — per-technique settings

---

## References

- [RAG Paper — Lewis et al. 2020](https://arxiv.org/abs/2005.11401)
- [NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques) — 42 technique notebooks
- [RAGAS Documentation](https://docs.ragas.io)
- [LMStudio](https://lmstudio.ai)
- [LangChain Docs](https://python.langchain.com)
- [LlamaIndex Docs](https://docs.llamaindex.ai)
- [HyDE Paper](https://arxiv.org/abs/2212.10496)
- [RAPTOR Paper](https://arxiv.org/abs/2401.18059)
- [Self-RAG Paper](https://arxiv.org/abs/2310.11511)
- [CRAG Paper](https://arxiv.org/abs/2401.15884)
