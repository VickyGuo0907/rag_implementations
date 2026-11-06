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

Measured with RAGAS on a small, hand-written set. Treat it as a worked example of the method, not a leaderboard: 6 questions, one run per technique, and a small local judge model. The same technique moved by up to ~0.15 between runs here, so differences of that size are noise.

**Methodology** — Dataset: `data/eval/test_questions.json` (first 6 of 10 Q/A pairs, 3 from the RAG overview and 3 from the Burlington FAQ, with reference answers) · Corpus: `ai_and_rag_overview.txt` + `burlington_faq.md` · Generation and judge: `qwen/qwen3-4b-2507` (LMStudio, local) · Embeddings: `all-MiniLM-L6-v2` · Retrieval: top_k=8, chunk=512 characters / overlap=50 · Framework: LangChain · Fresh vector store per run · Metrics: RAGAS faithfulness, answer relevancy, context precision, context recall.

| # | Technique | Faithful. | Ans. Rel. | Ctx Prec. | Ctx Recall | Notes |
|---|---|:-:|:-:|:-:|:-:|---|
| 01 | Naive (baseline) | 1.00 | 0.79 | 0.57 | 1.00 | baseline |
| 02 | Advanced | 0.97 | 0.88 | 0.68 | 1.00 | |
| 07 | Reranking | 1.00 | 0.89 | 0.71 | 1.00 | |
| 03 | HyDE | 0.83 | **0.90** | 0.54 | 0.83 | lowest faithfulness and precision |
| 09 | Corrective (CRAG) | 1.00 | 0.76 | **0.83** | 1.00 | best context precision |
| 11 | GraphRAG | — | — | — | — | not run (one LLM call per chunk is slow locally) |

Latency is not reported: the evaluation time includes the judge model's calls, so it says nothing about query speed.

**Effect of `top_k`:** an earlier run at top_k=5 had Naive at 0.83 faithfulness / 0.59 answer relevancy / 0.74 precision, and CRAG at only 0.67 context recall. At top_k=8 Naive's answer relevancy rose to 0.79 and CRAG's recall to 1.00, but context precision fell for most techniques (Naive 0.74 to 0.57) because more of the retrieved chunks are irrelevant. This is the recall-versus-precision trade-off; a question whose answer was split across two chunks (the four RAG components) only worked at the higher value.

**Findings:** At top_k=8 every technique reached full context recall except HyDE (0.83), so the differences are mostly in precision and answer quality. CRAG filtered retrieved text best (precision 0.83 against 0.57 for Naive), while HyDE gave the highest answer relevancy (0.90) but the lowest faithfulness and precision, plausibly because its hypothetical documents pull in loosely related chunks. Reranking matched the best answer relevancy without hurting faithfulness. With 6 questions, one run each, and a 4B judge, repeat on a larger set and a stronger judge before drawing conclusions.

**Reproduce:** `python -m evaluation.run_benchmark --technique reranking_rag --framework langchain --docs <dir> -n 6`

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
  - Companion tutorial for the ideas here: its multi-query and RAG-Fusion notebooks ↔ `04_query_transform_rag` / `05_fusion_rag`; HyDE ↔ `03_hyde_rag`; multi-representation indexing ↔ `06_parent_document_rag`; RAPTOR ↔ `12_raptor_rag`; routing ↔ `10_adaptive_rag`.
- [RAGAS](https://docs.ragas.io) · [LMStudio](https://lmstudio.ai) · [LangChain](https://python.langchain.com) · [LlamaIndex](https://docs.llamaindex.ai)
