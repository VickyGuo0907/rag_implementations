"""
Run a single RAG technique
===========================
Usage:
    python scripts/run_technique.py --technique naive_rag --framework langchain
    python scripts/run_technique.py --technique naive_rag --framework llamaindex --docs ./data/sample_docs
    python scripts/run_technique.py --technique naive_rag --query "What is RAG?"

Options:
    --technique   : RAG technique key (e.g., naive_rag, advanced_rag, hyde_rag)
    --framework   : langchain or llamaindex (default: langchain)
    --docs        : Path to documents directory or file (default: ./data/sample_docs)
    --query       : Single query to run (default: interactive mode)
    --config      : Path to config.yaml (default: ./config/config.yaml)
    --evaluate    : Run RAGAS evaluation after answering
"""

import argparse
import sys
import logging
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config_loader import ConfigLoader
from core.document_loader import load_documents


# ---------------------------------------------------------------------------
# Technique → Class mapping
# ---------------------------------------------------------------------------

TECHNIQUE_CLASSES = {
    ("naive_rag", "langchain"):   "techniques.01_naive_rag.langchain_impl.NaiveRAGLangChain",
    ("naive_rag", "llamaindex"):  "techniques.01_naive_rag.llamaindex_impl.NaiveRAGLlamaIndex",
    ("advanced_rag", "langchain"): "techniques.02_advanced_rag.langchain_impl.AdvancedRAGLangChain",
    ("advanced_rag", "llamaindex"): "techniques.02_advanced_rag.llamaindex_impl.AdvancedRAGLlamaIndex",
    ("hyde_rag", "langchain"):    "techniques.03_hyde_rag.langchain_impl.HyDERAGLangChain",
    ("hyde_rag", "llamaindex"):   "techniques.03_hyde_rag.llamaindex_impl.HyDERAGLlamaIndex",
    # Add more as techniques are implemented:
    # ("fusion_rag", "langchain"):  "techniques.05_fusion_rag.langchain_impl.FusionRAGLangChain",
}


def load_class(dotted_path: str):
    """Dynamically import and return a class from a dotted module path."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def main():
    parser = argparse.ArgumentParser(description="Run a RAG technique")
    parser.add_argument("--technique", required=True, help="RAG technique key")
    parser.add_argument("--framework", default="langchain", choices=["langchain", "llamaindex"])
    parser.add_argument("--docs", default="./data/sample_docs", help="Document path")
    parser.add_argument("--query", default=None, help="Single query (omit for interactive)")
    parser.add_argument("--config", default=None, help="Config file path")
    parser.add_argument("--evaluate", action="store_true", help="Run RAGAS evaluation")
    args = parser.parse_args()

    # Setup
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    cfg = ConfigLoader.get(args.config)

    # Load technique class
    key = (args.technique, args.framework)
    if key not in TECHNIQUE_CLASSES:
        print(f"\n❌ No implemented class for: technique='{args.technique}', framework='{args.framework}'")
        print(f"   Available: {list(TECHNIQUE_CLASSES.keys())}")
        sys.exit(1)

    RAGClass = load_class(TECHNIQUE_CLASSES[key])
    print(f"\n🚀 Initializing {RAGClass.__name__}...")
    rag = RAGClass(config=cfg._config)

    # Load and index documents
    doc_path = Path(args.docs)
    if doc_path.exists():
        print(f"📄 Loading documents from: {doc_path}")
        lc_docs = load_documents(str(doc_path))
        raw_texts = [d.page_content for d in lc_docs]
        metas = [d.metadata for d in lc_docs]
    else:
        print(f"⚠️  No documents found at {doc_path}. Using built-in sample documents.")
        raw_texts = [
            "RAG (Retrieval-Augmented Generation) combines retrieval with generation to produce grounded, accurate answers.",
            "The main components of RAG are: document indexer, vector store, retriever, and language model.",
            "Advanced RAG improves upon naive RAG with query rewriting, reranking, and contextual compression.",
        ]
        metas = [{"source": "sample"}] * len(raw_texts)

    print(f"🗂️  Indexing {len(raw_texts)} documents...")
    rag.index(raw_texts, metas)
    print("✅ Indexing complete!\n")

    # Query mode
    if args.query:
        result = rag.query(args.query)
        result.print_summary()
    else:
        # Interactive mode
        print("💬 Interactive mode (type 'quit' to exit)\n")
        while True:
            try:
                question = input("❓ Your question: ").strip()
                if question.lower() in ("quit", "exit", "q"):
                    break
                if not question:
                    continue
                result = rag.query(question)
                result.print_summary()
            except KeyboardInterrupt:
                break

    # Optional RAGAS evaluation
    if args.evaluate:
        from evaluation.ragas_evaluator import RAGASEvaluator
        eval_questions = [
            "What is RAG?",
            "What are the main components of RAG?",
            "How does Advanced RAG differ from Naive RAG?",
            "What is Pytorch Tensor?"
        ]
        evaluator = RAGASEvaluator()
        eval_result = evaluator.evaluate(rag, eval_questions)
        eval_result.print_report()


if __name__ == "__main__":
    main()
