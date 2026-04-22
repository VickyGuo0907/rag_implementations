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
│   ├── 01_naive_rag/                  ✅ Production Ready
│   │   ├── __init__.py                # Clean exports (LangChain + LlamaIndex)
│   │   ├── langchain_impl.py          # 154 lines, LCEL chain with ChromaDB
│   │   ├── llamaindex_impl.py         # 117 lines, VectorStoreIndex + QueryEngine
│   │   └── README.md                  # Flowchart, comparison, use cases, pros/cons
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

## Recent Updates & Fixes

### ✅ Latest Improvements (April 2026)

1. **PDF Document Support** 
   - Fixed document loader to properly handle PDF files alongside text documents
   - Installed `pypdf` dependency for PDF parsing
   - Example: `python scripts/run_technique.py --technique naive_rag --docs ./data/sample_docs` now loads both `.txt` and `.pdf` files

2. **01_Naive_RAG Code Cleanup**
   - Removed unused imports and ~50 lines of demo code
   - Fixed `__init__.py` to properly export both `NaiveRAGLangChain` and `NaiveRAGLlamaIndex`
   - Improved code quality: no unused imports, clear separation of concerns, comprehensive logging
   - Both implementations now cleanly accessible: `from techniques.naive_rag import NaiveRAGLangChain, NaiveRAGLlamaIndex`

3. **Embedding Model Configuration Fix**
   - Resolved LMStudio embedding compatibility issues with both frameworks
   - **Current recommendation**: Use HuggingFace embeddings (local, works with all frameworks)
   - **Optional**: LMStudio embeddings work with LlamaIndex (requires standard OpenAI model names)
   - Added detailed configuration notes in `config.yaml` explaining the tradeoffs

### 📋 Supported Document Types

| Type | Support | Notes |
|------|---------|-------|
| `.txt` | ✅ Full | Plain text files |
| `.md` | ✅ Full | Markdown files |
| `.pdf` | ✅ Full | Binary PDF files (via pypdf) |
| `.csv` | ✅ Full | Comma-separated values |
| `.docx` / `.doc` | ✅ Full | Microsoft Word documents |
| `.html` | ✅ URL | Via WebBaseLoader for URLs |

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
  model: "your-model-name"    # e.g., "qwen/qwen3.5-35b-a3b"

embeddings:
  provider: "huggingface"     # Use HuggingFace (local, no server needed)
  huggingface_fallback:
    model: "sentence-transformers/all-MiniLM-L6-v2"  # Or any HF model
    device: "cpu"             # Options: cpu, cuda, mps (M-series Macs)
```

**Note on Embeddings:**
- **Recommended**: HuggingFace embeddings (works with both LangChain & LlamaIndex, local, no server)
- **Alternative**: LMStudio embeddings work with LlamaIndex only (model name must be standard OpenAI name like `text-embedding-3-small`)

### 3. Run Naive RAG

```bash
# LangChain version (interactive mode, default sample docs)
python scripts/run_technique.py --technique naive_rag --framework langchain

# LlamaIndex version
python scripts/run_technique.py --technique naive_rag --framework llamaindex

# Single query
python scripts/run_technique.py \
  --technique naive_rag \
  --framework langchain \
  --docs ./data/sample_docs \
  --query "What is RAG and how does it work?"

# With RAGAS evaluation
python scripts/run_technique.py --technique naive_rag --evaluate

# Load both text and PDF documents
python scripts/run_technique.py \
  --technique naive_rag \
  --framework langchain \
  --docs ./data/sample_docs \
  --query "Explain the key concepts"
```

### 4. Run from Python

```python
from core.config_loader import ConfigLoader
from techniques.naive_rag import NaiveRAGLangChain, NaiveRAGLlamaIndex

cfg = ConfigLoader.get()

# Use either implementation interchangeably
rag = NaiveRAGLangChain(config=cfg._config)
# or: rag = NaiveRAGLlamaIndex(config=cfg._config)

rag.index([
    "RAG combines retrieval systems with language models...",
    "The key components are: indexer, vector store, retriever, and LLM...",
])

result = rag.query("What are the components of a RAG system?")
result.print_summary()
```

### 5. Verify Installation

```bash
# Test document loading (PDF + text files)
python -c "
from core.document_loader import load_documents
docs = load_documents('data/sample_docs')
print(f'✅ Loaded {len(docs)} documents')
"

# Test both implementations
python scripts/run_technique.py --technique naive_rag --framework langchain --query "Test"
python scripts/run_technique.py --technique naive_rag --framework llamaindex --query "Test"
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

### 5. Code Quality Standards

All implementations follow these standards:
- ✅ No unused imports
- ✅ Comprehensive docstrings
- ✅ Type hints on all functions
- ✅ Clear error handling with logging
- ✅ Consistent with BaseRAG interface
- ✅ Framework-independent abstractions
- ✅ Both LangChain and LlamaIndex support

Example: `01_naive_rag` is fully production-ready with clean code, detailed documentation, and dual-framework support.

### 6. Implementing a New Technique
1. Create `techniques/XX_name/` directory
2. Create `langchain_impl.py` inheriting from `BaseRAG`
3. Create `llamaindex_impl.py` (same interface)
4. Set `TECHNIQUE_NAME` and `FRAMEWORK` class attributes
5. Implement `_build_pipeline()`, `index()`, and `_query()`
6. Create `README.md` with flowchart, use cases, comparison, and pros/cons
7. Register in `scripts/run_technique.py` TECHNIQUE_CLASSES mapping
8. Test: `python scripts/run_technique.py --technique your_technique --framework langchain`

---

## Troubleshooting

### Embedding Model Issues

**Error: "OpenAIEmbeddingModelType is not valid"**
- **Cause**: Using custom embedding model name with LlamaIndex + OpenAI-compatible API
- **Solution**: Use standard OpenAI model names like `text-embedding-3-small` (Recommended: switch to HuggingFace embeddings)

**Error: "'input' field must be a string or an array"**
- **Cause**: LangChain + LMStudio embeddings API incompatibility
- **Solution**: Use HuggingFace embeddings instead (set `embeddings.provider: "huggingface"`)

**Slow embedding generation**
- **Cause**: Using CPU embeddings without optimization
- **Solution**: 
  - Set `device: "mps"` if on M-series Mac (Apple Silicon)
  - Set `device: "cuda"` if NVIDIA GPU available
  - Use lighter embedding model: `sentence-transformers/all-MiniLM-L6-v2` (default) or `sentence-transformers/paraphrase-MiniLM-L6-v2`

### Document Loading Issues

**Error: "UnicodeDecodeError" when loading PDF**
- **Cause**: PDF file treated as text file
- **Solution**: Ensure `pypdf` is installed (`pip install pypdf`)

**Error: "No module named 'langchain_community.document_loaders'"**
- **Solution**: Install required dependencies: `pip install -r requirements.txt`

### LMStudio Connection Issues

**Error: "Connection refused at localhost:1234"**
- **Solution**: Ensure LMStudio is running with server enabled (default port 1234)

**Error: "Model not found"**
- **Solution**: Ensure the model name in `config.yaml` matches exactly what's loaded in LMStudio

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
