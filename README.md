# RAG Techniques — A Reference Implementation of 14 Patterns

> Production-quality reference implementations of **14 RAG techniques**, each built in both **LangChain** and **LlamaIndex**, under one consistent architecture with **RAGAS** evaluation and a CLI + web UI.
> Runs locally on **LMStudio** (local LLM). For learning, comparison, and as a design reference.
> Technique catalog adapted from [NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques); the architecture, dual-framework abstraction, evaluation harness, and decision guide are original to this repo.

---

## Contents

- [What this is (and isn't)](#what-this-is-and-isnt)
- [Quick start](#quick-start)
- [The 14 techniques](#the-14-techniques)
- [Which RAG should I use?](#which-rag-should-i-use)
- [Benchmark results](#benchmark-results)
- [Architecture principles](#architecture-principles)
- [CLI reference](#cli-reference)
- [Configuration & troubleshooting](#configuration--troubleshooting)
- [References](#references)

---

## What this is (and isn't)

**What it is:** a clean, abstracted, config-driven implementation of 14 RAG patterns you can run, compare, and evaluate locally. Every technique shares one interface (`BaseRAG`), ships in both LangChain and LlamaIndex, and is measurable with RAGAS. It's a reference for *how the patterns differ and when to use each*, not a deployed service.

**What production would add** — being explicit about the line is the point:

| Concern | This repo | Production |
|---|---|---|
| LLM serving | Local LMStudio | Managed endpoint (AWS Bedrock / vLLM on GPU), autoscaling, fallback model + routing |
| Embeddings | Local HF MiniLM on CPU/MPS | Batched embedding service, GPU, caching |
| Vector store | ChromaDB (local) | Managed/distributed (OpenSearch, pgvector, Qdrant, Pinecone) with replication |
| Ingestion | Folder of sample docs | Pipeline with change-data-capture, dedup, incremental re-index |
| Retrieval @ scale | In-memory, small corpus | Sharding, hybrid search, ANN tuning, p95 latency budgets |
| Evaluation | RAGAS on demand | Eval as a CI gate; regression alerts on every change |
| Observability | Console / RAGAS output | Tracing (OpenTelemetry), prompt/version logging, dashboards, alerting |
| Security | None | AuthN/Z, PII handling, data residency, prompt-injection defense on retrieved text |
| Cost | Local (free) | Token budgeting, cache hit-rate tracking, per-request cost monitoring |
| Reliability | Single process | Retries, timeouts, rate-limit handling, graceful degradation |

**Reference deployment (how one of these would ship on AWS):**
> Documents land in **S3** → an ingestion job (Lambda/ECS) parses and chunks them → embeddings via **Bedrock** (or a vLLM endpoint) → vectors in **OpenSearch** or **pgvector (RDS)**. Query path: **API Gateway → Lambda/ECS** running the RAG technique, with a **reranker** before the **Bedrock** generation call. **RAGAS** runs as a scheduled eval job and a CI gate; **CloudWatch + OpenTelemetry** carry traces, latency, and cost. Guardrails (input/output filtering, injection defense on retrieved text) wrap the LLM call.

---

## Quick start

### 1. Prerequisites
```bash
# Install LMStudio (https://lmstudio.ai), load a model
# (e.g. llama-3.1-8b-instruct), start the server on port 1234.

git clone https://github.com/VickyGuo0907/rag_implementations.git
cd rag_implementations
pip install .
```

### 2. Configure
Everything lives in `config/config.yaml`. The minimum to set:
```yaml
lmstudio:
  model: "your-model-name"
embeddings:
  provider: "huggingface"        # local, no server needed (recommended)
  huggingface_fallback:
    model: "sentence-transformers/all-MiniLM-L6-v2"
    device: "cpu"                # cpu | cuda | mps (Apple Silicon)
```

### 3. Run
```bash
# CLI — interactive
python main.py run --technique naive_rag --framework langchain

# CLI — single query, with evaluation
python main.py run --technique naive_rag --query "What is RAG?" --evaluate

# Web UI
streamlit run app.py            # opens http://localhost:8501
```

### 4. Use from Python
```python
from core.config_loader import ConfigLoader
from techniques.registry import load_class

# any technique key + "langchain" | "llamaindex"
rag = load_class("naive_rag", "langchain")(config=ConfigLoader.get()._config)
rag.index(["RAG combines retrieval with language models...",
           "Components: indexer, vector store, retriever, LLM..."])
print(rag.query("What are the components of a RAG system?").answer)
```

> Any technique is swappable through the same `BaseRAG` interface — change the technique key above.

---

## The 14 techniques

Grouped by **what kind of problem they solve**, not by difficulty — complexity overlaps across tiers.

**Tier 1 — Foundational** · the retrieve-then-generate baseline everything builds on
| # | Technique | Complexity | Latency | Accuracy |
|---|---|:-:|:-:|:-:|
| 01 | Naive RAG | ⭐ | Low | Moderate |
| 02 | Advanced RAG | ⭐⭐⭐ | Medium | High |

**Tier 2 — Query & retrieval enhancement** · linear improvements to one pipeline stage
| # | Technique | Complexity | Latency | Accuracy |
|---|---|:-:|:-:|:-:|
| 03 | HyDE RAG | ⭐⭐ | Medium | High |
| 04 | Query Transform RAG | ⭐⭐ | Medium | High |
| 05 | Fusion RAG | ⭐⭐⭐ | Medium | High |
| 06 | Parent Document RAG | ⭐⭐ | Low | High |
| 07 | Reranking RAG | ⭐⭐ | Medium | High |

**Tier 3 — Adaptive & self-reflective** · real control flow: self-critique, correction, routing
| # | Technique | Complexity | Latency | Accuracy |
|---|---|:-:|:-:|:-:|
| 08 | Self-RAG | ⭐⭐⭐⭐ | High | High |
| 09 | Corrective RAG (CRAG) | ⭐⭐⭐⭐ | High | Very High |
| 10 | Adaptive RAG | ⭐⭐⭐ | Varies | High |

**Tier 4 — Structural & specialized** · different data structures / machinery
| # | Technique | Complexity | Latency | Accuracy |
|---|---|:-:|:-:|:-:|
| 11 | GraphRAG | ⭐⭐⭐⭐⭐ | High | Very High |
| 12 | RAPTOR | ⭐⭐⭐⭐⭐ | High | Very High |
| 13 | Agentic RAG | ⭐⭐⭐⭐⭐ | High | Very High |
| 14 | Multi-modal RAG | ⭐⭐⭐⭐ | High | High |

Each lives in `techniques/NN_name/` with a `langchain_impl.py`, a `llamaindex_impl.py`, and a `README.md` (flowchart, use cases, pros/cons).

**Supported documents:** `.txt`, `.md`, `.pdf` (pypdf), `.csv`, `.docx`/`.doc`, `.html` (URL via WebBaseLoader).

---

## Which RAG should I use?

```
What is your primary requirement?
├─ SPEED, latency is top priority        → Naive (01) or Parent Document (06)
├─ ACCURACY in production
│   ├─ standard doc Q&A                   → Advanced (02) + Reranking (07)
│   ├─ vocabulary mismatch (users vs docs)→ HyDE (03)
│   ├─ ambiguous / multi-faceted queries  → Fusion (05)
│   └─ must verify / self-correct         → Self-RAG (08) or CRAG (09)
├─ RELATIONSHIPS, multi-hop reasoning
│   ├─ entities / knowledge graphs        → GraphRAG (11)
│   └─ hierarchical docs (books, reports) → RAPTOR (12)
├─ AUTOMATION, multi-step tasks           → Agentic (13)
├─ MULTIMODAL, images/tables              → Multi-modal (14)
├─ COST, mixed query types                → Adaptive (10) — routes simple queries to no-retrieval
└─ PROTOTYPING                            → start Naive (01), baseline, then upgrade
```

| Use case | Primary | Secondary |
|---|---|---|
| Customer support FAQ | Naive / Advanced | Reranking |
| Enterprise knowledge base | Advanced | Fusion |
| Legal document research | Corrective (CRAG) | Reranking |
| Medical Q&A | Self-RAG | Corrective |
| Technical documentation | HyDE | Advanced |
| Research paper assistant | RAPTOR | GraphRAG |
| Multi-source intelligence | Agentic | Fusion |
| Product catalog with images | Multi-modal | Advanced |
| Financial reports | GraphRAG | RAPTOR |
| Real-time information | Corrective (web) | Agentic |

---

## Benchmark results

> Run `python main.py run --technique <t> --evaluate` across the dataset and fill this in. Structured-but-empty is fine; placeholder numbers are not.

**Methodology** — Dataset: `<name / # Q-A pairs>` · Generation: `<model>` · Embeddings: `all-MiniLM-L6-v2` · Retrieval: top_k=`<n>`, chunk=`<n>`/overlap=`<n>` · Metrics: RAGAS (faithfulness, answer relevancy, context precision, context recall) + mean latency.

| # | Technique | Faithful. | Ans. Rel. | Ctx Prec. | Ctx Recall | Latency (s) | Notes |
|---|---|:-:|:-:|:-:|:-:|:-:|---|
| 01 | Naive (baseline) | — | — | — | — | — | baseline |
| 02 | Advanced | — | — | — | — | — | |
| 07 | Reranking | — | — | — | — | — | |
| 03 | HyDE | — | — | — | — | — | |
| 09 | Corrective (CRAG) | — | — | — | — | — | |
| 11 | GraphRAG | — | — | — | — | — | multi-hop set |

**Findings:** *(2–3 sentences once measured — e.g. "Reranking lifted context precision from X to Y at +Zs; CRAG gave the biggest faithfulness gain on the adversarial subset but ~2× cost. Best accuracy-per-dollar on this corpus: Advanced RAG.")*

---

## Architecture principles

1. **One abstraction.** Every technique inherits `BaseRAG`: `index(docs)` → `query(question)` → `RAGResult` (answer + sources + metadata). Techniques are interchangeable in any app.
2. **Configuration-first.** Zero hardcoded values — model, embeddings, vector store, chunk size, top_k, reranker all live in `config/config.yaml`. Change once, every technique follows.
3. **Dual-framework.** Each technique has a LangChain and a LlamaIndex implementation behind the same interface — LangChain for custom chains/agents, LlamaIndex for out-of-the-box pipelines and connectors.
4. **Evaluation loop.** Baseline → measure (RAGAS) → add complexity → measure again. Never add a technique without measuring what it bought.
5. **Consistent quality bar.** Type hints, docstrings, error handling + logging, framework-independent abstractions across all implementations.

**Adding a technique:** create `techniques/NN_name/`, implement `langchain_impl.py` + `llamaindex_impl.py` from `BaseRAG` (`_build_pipeline`, `index`, `_query`), add a `README.md`, and register in `techniques/registry.py` (`TECHNIQUES` + `IMPLEMENTATIONS`).

---

## CLI reference

`python main.py <command> [options]`

| Command | Purpose |
|---|---|
| `run` | Execute a technique, with optional `--query` and `--evaluate` |
| `list` | All techniques with complexity and status |
| `info <technique>` | Details for one technique |
| `eval` | Evaluate a technique with RAGAS |
| `config show` / `config validate` | Inspect or validate configuration |

**`run` options:** `--technique` (required) · `--framework langchain|llamaindex` (default langchain) · `--docs <path>` (default `./data/sample_docs`) · `--query "<q>"` (omit for interactive) · `--config <path>` · `--evaluate`.

**CLI vs Web UI:** CLI for automation, scripting, batch eval; Streamlit UI for interactive exploration and demos (technique dropdown, framework toggle, source cards, RAGAS charts).

---

## Configuration & troubleshooting

`config/config.yaml` sections: `lmstudio`, `embeddings`, `vector_store` (ChromaDB/FAISS/Qdrant), `document` (chunk size/overlap), `retrieval` (top_k, search type, reranking), `evaluation` (RAGAS), `rag_techniques` (per-technique).

Common issues:
- **Embedding model errors / incompatibility** → use HuggingFace embeddings (`embeddings.provider: "huggingface"`); LMStudio embeddings require standard OpenAI model names and work with LlamaIndex only.
- **Slow embeddings** → set `device: "mps"` (Apple Silicon) or `"cuda"` (NVIDIA); keep the MiniLM model.
- **PDF `UnicodeDecodeError`** → ensure `pypdf` is installed.
- **`Connection refused at localhost:1234`** → LMStudio server isn't running, or the config model name doesn't match the loaded model.

---

## References

- [RAG — Lewis et al. 2020](https://arxiv.org/abs/2005.11401) · [HyDE](https://arxiv.org/abs/2212.10496) · [RAPTOR](https://arxiv.org/abs/2401.18059) · [Self-RAG](https://arxiv.org/abs/2310.11511) · [CRAG](https://arxiv.org/abs/2401.15884)
- [NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques) (technique catalog this repo adapts) · [LangChain RAG from Scratch](https://github.com/langchain-ai/rag-from-scratch)
- [RAGAS](https://docs.ragas.io) · [LMStudio](https://lmstudio.ai) · [LangChain](https://python.langchain.com) · [LlamaIndex](https://docs.llamaindex.ai)
