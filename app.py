#!/usr/bin/env python
"""RAG Techniques Explorer — Streamlit UI for interactive RAG querying and evaluation."""

import sys
from pathlib import Path
from typing import Optional

import streamlit as st

# Ensure project root is in path (matches main.py)
sys.path.insert(0, str(Path(__file__).parent))

from main import TECHNIQUE_CLASSES, TECHNIQUES_METADATA, initialize_rag, load_and_index_documents
from evaluation.ragas_evaluator import RAGASEvaluator


st.set_page_config(page_title="RAG Explorer", layout="wide", initial_sidebar_state="expanded")


def init_session_state():
    """Initialize session_state defaults on first load."""
    defaults = {
        "rag": None,
        "rag_key": None,
        "last_result": None,
        "eval_result": None,
        "init_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar() -> tuple[str, str, str, bool]:
    """Render sidebar config panel. Returns (technique, framework, doc_path, clicked_init)."""
    st.sidebar.title("⚙️ RAG Configuration")

    technique = st.sidebar.selectbox(
        "Technique",
        options=list(TECHNIQUES_METADATA.keys()),
        format_func=lambda t: f"{TECHNIQUES_METADATA[t]['name']} — {TECHNIQUES_METADATA[t]['description'][:40]}...",
    )

    meta = TECHNIQUES_METADATA[technique]
    st.sidebar.caption(f"Status: {meta['status']} | Complexity: {meta['complexity']} | Latency: {meta['latency']}")

    framework = st.sidebar.radio("Framework", options=["LangChain", "LlamaIndex"], horizontal=True)
    framework_key = framework.lower().replace("langchain", "langchain").replace("llamaindex", "llamaindex")

    doc_path = st.sidebar.text_input("Documents Path", value="./data/sample_docs")

    clicked_init = st.sidebar.button("🚀 Initialize RAG", use_container_width=True)

    return technique, framework_key, doc_path, clicked_init


def handle_initialization(technique: str, framework: str, doc_path: str, clicked: bool):
    """Handle RAG initialization with session_state caching and fingerprint validation."""
    rag_key = f"{technique}_{framework}_{doc_path}"

    # Invalidate on config change
    if rag_key != st.session_state.get("rag_key"):
        st.session_state["rag"] = None
        st.session_state["last_result"] = None
        st.session_state["eval_result"] = None

    if not clicked:
        return

    # Validate technique+framework combo exists
    if (technique, framework) not in TECHNIQUE_CLASSES:
        st.session_state["init_error"] = f"Invalid combo: {technique} + {framework}"
        st.sidebar.error(f"❌ {st.session_state['init_error']}")
        return

    # Initialize
    try:
        with st.sidebar.spinner("🔄 Initializing RAG and indexing documents..."):
            rag = initialize_rag(technique, framework, config_path=None)
            load_and_index_documents(rag, doc_path)
            st.session_state["rag"] = rag
            st.session_state["rag_key"] = rag_key
            st.session_state["init_error"] = None
        st.sidebar.success("✅ RAG initialized!")
    except Exception as e:
        st.session_state["rag"] = None
        st.session_state["init_error"] = str(e)
        st.sidebar.error(f"❌ Initialization failed: {e}")


def render_query_area():
    """Render query input and button."""
    if st.session_state["rag"] is None:
        st.warning("⚠️ Please initialize RAG first (sidebar)")
        return

    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input("Ask a question about your documents...", key="query_input")
    with col2:
        clicked_ask = st.button("🔍 Ask", use_container_width=True)

    if not clicked_ask or not query.strip():
        return

    try:
        with st.spinner("⏳ Thinking..."):
            result = st.session_state["rag"].query(query)
            st.session_state["last_result"] = result
    except Exception as e:
        st.error(f"❌ Query failed: {e}")


def render_answer():
    """Display answer with metrics."""
    if st.session_state["last_result"] is None:
        return

    result = st.session_state["last_result"]
    st.markdown("---")
    st.markdown(f"## {result.answer}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("⏱️ Latency", f"{result.latency_ms:.0f} ms")
    with col2:
        st.metric("🧠 Technique", result.technique)
    with col3:
        st.metric("🔧 Framework", result.framework)


def render_sources():
    """Display retrieved sources with score visualization."""
    if st.session_state["last_result"] is None:
        return

    result = st.session_state["last_result"]
    if not result.source_documents:
        st.info("ℹ️ No sources retrieved")
        return

    st.markdown("---")
    st.subheader("📚 Retrieved Sources")

    for i, doc in enumerate(result.source_documents, 1):
        score_str = f"Score: {doc.score:.3f}" if doc.score is not None else "No score"
        with st.expander(f"**Source {i}** — {score_str}"):
            if doc.score is not None:
                clamped_score = min(max(doc.score, 0.0), 1.0)
                st.progress(clamped_score)
            else:
                st.caption("No relevance score available")

            if doc.metadata:
                st.caption(f"📋 Metadata: {doc.metadata}")
            st.write(doc.content)


def render_evaluation():
    """Run and display RAGAS evaluation."""
    if st.session_state["rag"] is None:
        return

    with st.expander("📊 Evaluation (RAGAS)", expanded=False):
        if st.button("🔬 Run Evaluation"):
            try:
                with st.spinner("⏳ Running RAGAS evaluation... (2-5 min)"):
                    evaluator = RAGASEvaluator()
                    eval_questions = [
                        "What is RAG?",
                        "What are the main components of RAG?",
                        "How does Advanced RAG differ from Naive RAG?",
                    ]
                    eval_result = evaluator.evaluate(st.session_state["rag"], eval_questions)
                    st.session_state["eval_result"] = eval_result
            except Exception as e:
                st.error(f"❌ Evaluation failed: {e}")

        if st.session_state["eval_result"] is not None:
            eval_result = st.session_state["eval_result"]
            st.subheader(f"Results — {eval_result.num_samples} samples")

            # Metric tiles
            cols = st.columns(len(eval_result.scores))
            for col, (metric_name, score) in zip(cols, eval_result.scores.items()):
                with col:
                    st.metric(metric_name.replace("_", " ").title(), f"{score:.3f}")

            # Bar chart
            st.bar_chart(eval_result.scores)


def main():
    """Main app orchestration."""
    st.title("🚀 RAG Techniques Explorer")
    st.markdown("*Interactive interface for experimenting with different RAG techniques*")

    init_session_state()
    technique, framework, doc_path, clicked_init = render_sidebar()
    handle_initialization(technique, framework, doc_path, clicked_init)
    render_query_area()
    render_answer()
    render_sources()
    render_evaluation()


if __name__ == "__main__":
    main()
