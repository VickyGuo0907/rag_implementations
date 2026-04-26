#!/usr/bin/env python
"""
RAG Techniques CLI - Main entry point for the project
======================================================

A comprehensive reference implementation for 14+ RAG techniques with
LangChain and LlamaIndex support, powered by LMStudio.

Usage:
    python main.py run --technique naive_rag
    python main.py run --technique naive_rag --framework llamaindex --query "What is RAG?"
    python main.py list
    python main.py info naive_rag
    python main.py eval --technique naive_rag
    python main.py config show
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from core.config_loader import ConfigLoader
from core.document_loader import load_documents

# ---------------------------------------------------------------------------
# Technique → Class mapping
# ---------------------------------------------------------------------------

TECHNIQUE_CLASSES = {
    ("naive_rag", "langchain"): "techniques.01_naive_rag.langchain_impl.NaiveRAGLangChain",
    ("naive_rag", "llamaindex"): "techniques.01_naive_rag.llamaindex_impl.NaiveRAGLlamaIndex",
    ("advanced_rag", "langchain"): "techniques.02_advanced_rag.langchain_impl.AdvancedRAGLangChain",
    ("advanced_rag", "llamaindex"): "techniques.02_advanced_rag.llamaindex_impl.AdvancedRAGLlamaIndex",
    ("hyde_rag", "langchain"): "techniques.03_hyde_rag.langchain_impl.HyDERAGLangChain",
    ("hyde_rag", "llamaindex"): "techniques.03_hyde_rag.llamaindex_impl.HyDERAGLlamaIndex",
}

TECHNIQUES_METADATA = {
    "naive_rag": {
        "name": "Naive RAG",
        "complexity": "⭐",
        "latency": "🟢 Low",
        "accuracy": "🟡 Moderate",
        "status": "✅ Done",
        "description": "Basic retrieval-augmented generation with vector similarity search",
    },
    "advanced_rag": {
        "name": "Advanced RAG",
        "complexity": "⭐⭐⭐",
        "latency": "🟡 Medium",
        "accuracy": "🟢 High",
        "status": "✅ Done",
        "description": "Enhanced RAG with query rewriting, reranking, and contextual compression",
    },
    "hyde_rag": {
        "name": "HyDE RAG",
        "complexity": "⭐⭐",
        "latency": "🟡 Medium",
        "accuracy": "🟢 High",
        "status": "✅ Done",
        "description": "Hypothetical document embeddings for vocabulary mismatch resolution",
    },
}


def load_class(dotted_path: str):
    """Dynamically import and return a class from a dotted module path."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def setup_logging(level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(levelname)s | %(name)s | %(message)s"
    )


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------

def cmd_run(args):
    """Run a RAG technique with the specified configuration."""
    setup_logging()

    # Initialize RAG system
    rag = initialize_rag(args.technique, args.framework, args.config)

    # Load and index documents
    load_and_index_documents(rag, args.docs)

    # Run queries
    if args.query:
        run_single_query(rag, args.query)
    else:
        run_interactive_mode(rag)

    # Optional evaluation
    if args.evaluate:
        run_evaluation(rag)


def initialize_rag(technique: str, framework: str, config_path: Optional[str]):
    """Initialize and return RAG instance."""
    key = (technique, framework)
    if key not in TECHNIQUE_CLASSES:
        print(f"\n❌ No implemented class for: technique='{technique}', framework='{framework}'")
        print(f"   Available: {list(TECHNIQUE_CLASSES.keys())}")
        sys.exit(1)

    cfg = ConfigLoader.get(config_path)
    RAGClass = load_class(TECHNIQUE_CLASSES[key])
    print(f"\n🚀 Initializing {RAGClass.__name__}...")
    return RAGClass(config=cfg._config)


def load_and_index_documents(rag, doc_path: str) -> tuple[list[str], list[dict]]:
    """Load documents from path or use sample documents, then index them."""
    path = Path(doc_path)
    if path.exists():
        print(f"📄 Loading documents from: {path}")
        lc_docs = load_documents(str(path))
        raw_texts = [d.page_content for d in lc_docs]
        metas = [d.metadata for d in lc_docs]
    else:
        print(f"⚠️  No documents found at {path}. Using built-in sample documents.")
        raw_texts = [
            ("RAG (Retrieval-Augmented Generation) combines retrieval with generation "
             "to produce grounded, accurate answers."),
            ("The main components of RAG are: document indexer, vector store, retriever, "
             "and language model."),
            ("Advanced RAG improves upon naive RAG with query rewriting, reranking, "
             "and contextual compression."),
        ]
        metas = [{"source": "sample"}] * len(raw_texts)

    print(f"🗂️  Indexing {len(raw_texts)} documents...")
    rag.index(raw_texts, metas)
    print("✅ Indexing complete!\n")
    return raw_texts, metas


def run_single_query(rag, query: str):
    """Run a single query and print results."""
    result = rag.query(query)
    result.print_summary()


def run_interactive_mode(rag):
    """Run interactive query loop."""
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


def run_evaluation(rag):
    """Run RAGAS evaluation."""
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


# ---------------------------------------------------------------------------
# Subcommand: list
# ---------------------------------------------------------------------------

def cmd_list(args):
    """List all available RAG techniques."""
    print("\n📚 Available RAG Techniques\n")
    print("┌─ Technique ────────────┬─ Status ─┬─ Complexity ─┬─ Latency ──────┬─ Accuracy ────┐")

    for technique_key in sorted(TECHNIQUES_METADATA.keys()):
        meta = TECHNIQUES_METADATA[technique_key]
        name = meta["name"]
        status = meta["status"]
        complexity = meta["complexity"]
        latency = meta["latency"]
        accuracy = meta["accuracy"]

        print(
            f"│ {name:<23} │ {status:<9} │ {complexity:<12} │ {latency:<14} │ {accuracy:<14} │"
        )

    print("└────────────────────────┴──────────┴─────────────┴────────────────┴────────────────┘")
    print("\nUse 'python main.py run --technique <name>' to run a technique")
    print("Use 'python main.py info <name>' to see detailed information\n")


# ---------------------------------------------------------------------------
# Subcommand: info
# ---------------------------------------------------------------------------

def cmd_info(args):
    """Show detailed information about a technique."""
    technique = args.technique
    if technique not in TECHNIQUES_METADATA:
        print(f"\n❌ Unknown technique: {technique}")
        print(f"   Available: {', '.join(TECHNIQUES_METADATA.keys())}\n")
        sys.exit(1)

    meta = TECHNIQUES_METADATA[technique]
    print(f"\n📖 {meta['name']}")
    print(f"   Description: {meta['description']}")
    print(f"   Status: {meta['status']}")
    print(f"   Complexity: {meta['complexity']}")
    print(f"   Latency: {meta['latency']}")
    print(f"   Accuracy: {meta['accuracy']}")
    print(f"\n💻 Available Implementations:")
    print(f"   - LangChain:  python main.py run --technique {technique} --framework langchain")
    print(f"   - LlamaIndex: python main.py run --technique {technique} --framework llamaindex")
    print()


# ---------------------------------------------------------------------------
# Subcommand: eval
# ---------------------------------------------------------------------------

def cmd_eval(args):
    """Evaluate a RAG technique using RAGAS."""
    setup_logging()

    # Initialize RAG system
    rag = initialize_rag(args.technique, args.framework, args.config)

    # Load and index documents
    load_and_index_documents(rag, args.docs)

    # Run evaluation
    run_evaluation(rag)


# ---------------------------------------------------------------------------
# Subcommand: config
# ---------------------------------------------------------------------------

def cmd_config(args):
    """Manage project configuration."""
    import yaml

    config_path = Path("config/config.yaml")

    if args.action == "show":
        cfg = ConfigLoader.get()
        print("\n⚙️  Current Configuration:")
        print(f"   Config file: {config_path.resolve()}")
        with open(config_path) as f:
            config_content = yaml.safe_load(f)
        print(yaml.dump(config_content, default_flow_style=False, indent=2))
    elif args.action == "validate":
        try:
            cfg = ConfigLoader.get()
            print("\n✅ Configuration is valid!")
            print(f"   Config file: {config_path.resolve()}")
        except Exception as e:
            print(f"\n❌ Configuration error: {e}")
            sys.exit(1)


# ---------------------------------------------------------------------------
# Main argument parser
# ---------------------------------------------------------------------------

def create_parser():
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="RAG Techniques CLI - Comprehensive reference implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py run --technique naive_rag
  python main.py run --technique advanced_rag --framework llamaindex --query "What is RAG?"
  python main.py list
  python main.py info naive_rag
  python main.py eval --technique naive_rag --framework langchain
  python main.py config show
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a RAG technique")
    run_parser.add_argument("--technique", required=True, help="RAG technique key")
    run_parser.add_argument(
        "--framework",
        default="langchain",
        choices=["langchain", "llamaindex"],
        help="Framework to use (default: langchain)"
    )
    run_parser.add_argument(
        "--docs",
        default="./data/sample_docs",
        help="Path to documents directory or file (default: ./data/sample_docs)"
    )
    run_parser.add_argument(
        "--query",
        default=None,
        help="Single query to run (omit for interactive mode)"
    )
    run_parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml (default: ./config/config.yaml)"
    )
    run_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run RAGAS evaluation after answering"
    )
    run_parser.set_defaults(func=cmd_run)

    # List command
    list_parser = subparsers.add_parser("list", help="List all available techniques")
    list_parser.set_defaults(func=cmd_list)

    # Info command
    info_parser = subparsers.add_parser("info", help="Show detailed technique information")
    info_parser.add_argument("technique", help="Technique name")
    info_parser.set_defaults(func=cmd_info)

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a RAG technique")
    eval_parser.add_argument("--technique", required=True, help="RAG technique key")
    eval_parser.add_argument(
        "--framework",
        default="langchain",
        choices=["langchain", "llamaindex"],
        help="Framework to use (default: langchain)"
    )
    eval_parser.add_argument(
        "--docs",
        default="./data/sample_docs",
        help="Path to documents directory or file"
    )
    eval_parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml"
    )
    eval_parser.set_defaults(func=cmd_eval)

    # Config command
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_parser.add_argument(
        "action",
        choices=["show", "validate"],
        help="Configuration action"
    )
    config_parser.set_defaults(func=cmd_config)

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
