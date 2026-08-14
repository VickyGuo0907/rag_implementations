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
import logging
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from core.config_loader import ConfigLoader
from core.document_loader import load_documents
from techniques.registry import (
    TECHNIQUES,
    get_implementations,
    is_registered,
    load_class,
)


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


def initialize_rag(technique: str, framework: str, config_path: str | None):
    """Initialize and return RAG instance."""
    if not is_registered(technique, framework):
        print(f"\n❌ No implemented class for: technique='{technique}', framework='{framework}'")
        print(f"   Available: {get_implementations()}")
        sys.exit(1)

    cfg = ConfigLoader.get(config_path)
    RAGClass = load_class(technique, framework)
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

_STATUS_ICONS = {"implemented": "✅ Done", "stub": "🔧 Stub"}
_COMPLEXITY_ICONS = {1: "⭐", 2: "⭐⭐", 3: "⭐⭐⭐", 4: "⭐⭐⭐⭐", 5: "⭐⭐⭐⭐⭐"}
_LATENCY_ICONS = {"Low": "🟢 Low", "Medium": "🟡 Medium", "High": "🔴 High", "Varies": "🟡 Varies"}
_ACCURACY_ICONS = {
    "Moderate": "🟡 Moderate",
    "High": "🟢 High",
    "Very High": "🟢 Very High",
}


def cmd_list(args):
    """List all available RAG techniques."""
    print("\n📚 Available RAG Techniques\n")
    print("┌─ Technique ────────────┬─ Status ─┬─ Complexity ─┬─ Latency ──────┬─ Accuracy ────┐")

    for technique_key in sorted(TECHNIQUES.keys()):
        meta = TECHNIQUES[technique_key]
        name = meta["name"]
        status = _STATUS_ICONS.get(meta["status"], meta["status"])
        complexity = _COMPLEXITY_ICONS.get(meta["complexity"], "⭐" * meta["complexity"])
        latency = _LATENCY_ICONS.get(meta["latency"], meta["latency"])
        accuracy = _ACCURACY_ICONS.get(meta["accuracy"], meta["accuracy"])

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
    if technique not in TECHNIQUES:
        print(f"\n❌ Unknown technique: {technique}")
        print(f"   Available: {', '.join(TECHNIQUES.keys())}\n")
        sys.exit(1)

    meta = TECHNIQUES[technique]
    print(f"\n📖 {meta['name']}")
    print(f"   Description: {meta['description']}")
    print(f"   Status: {_STATUS_ICONS.get(meta['status'], meta['status'])}")
    print(f"   Complexity: {_COMPLEXITY_ICONS.get(meta['complexity'], '⭐' * meta['complexity'])}")
    print(f"   Latency: {_LATENCY_ICONS.get(meta['latency'], meta['latency'])}")
    print(f"   Accuracy: {_ACCURACY_ICONS.get(meta['accuracy'], meta['accuracy'])}")
    print("\n💻 Available Implementations:")
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
        ConfigLoader.get()
        print("\n⚙️  Current Configuration:")
        print(f"   Config file: {config_path.resolve()}")
        with open(config_path) as f:
            config_content = yaml.safe_load(f)
        print(yaml.dump(config_content, default_flow_style=False, indent=2))
    elif args.action == "validate":
        try:
            ConfigLoader.get()
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
