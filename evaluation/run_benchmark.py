"""
Benchmark one technique with RAGAS against data/eval/test_questions.json.

Unlike `main.py run --evaluate`, this scores all four RAGAS metrics (the dataset
carries reference answers) and indexes into a fresh, throwaway vector store so
leftovers from earlier runs cannot skew retrieval.

    python -m evaluation.run_benchmark --technique naive_rag --framework langchain -n 6
"""

import argparse
import json
import math
import tempfile
import time
from pathlib import Path

DEFAULT_DATASET = "data/eval/test_questions.json"


def load_dataset(path: str, limit: int | None) -> tuple[list[str], list[str]]:
    """Return (questions, ground_truths) from a JSON list of {question, ground_truth}."""
    rows = json.loads(Path(path).read_text())[:limit]
    return [r["question"] for r in rows], [r["ground_truth"] for r in rows]


def check_scores(scores: dict[str, float]) -> None:
    """Raise if any metric is NaN: RAGAS reports failed judge calls (e.g. LLM down) as NaN."""
    failed = [name for name, value in scores.items() if math.isnan(value)]
    if failed:
        raise RuntimeError(
            f"RAGAS produced no score for {failed}; the judge LLM calls failed. "
            "Check that the LMStudio server is running and the model is loaded."
        )


def run_benchmark(technique: str, framework: str, docs: str, limit: int | None,
                  dataset: str = DEFAULT_DATASET, config_path: str | None = None) -> dict:
    """Index `docs` into a fresh store, score the technique with RAGAS, return a result dict."""
    from core.config_loader import ConfigLoader
    from core.document_loader import load_documents
    from evaluation.ragas_evaluator import RAGASEvaluator
    from techniques.registry import load_class

    questions, truths = load_dataset(dataset, limit)
    cfg = ConfigLoader.get(config_path)
    with tempfile.TemporaryDirectory() as store:
        cfg._config["vector_store"]["persist_directory"] = store
        rag = load_class(technique, framework)(config=cfg._config)
        loaded = load_documents(docs)
        rag.index([d.page_content for d in loaded], [d.metadata for d in loaded])
        start = time.time()
        result = RAGASEvaluator(config_path).evaluate(rag, questions, ground_truths=truths)
    check_scores(result.scores)
    return {"technique": technique, "framework": framework, "num_questions": len(questions),
            "scores": result.scores, "eval_seconds": round(time.time() - start, 1)}


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--technique", required=True)
    p.add_argument("--framework", default="langchain", choices=["langchain", "llamaindex"])
    p.add_argument("--docs", default="data/sample_docs")
    p.add_argument("-n", "--limit", type=int, default=None, help="use only the first N questions")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--config", default=None)
    a = p.parse_args()
    print(json.dumps(run_benchmark(a.technique, a.framework, a.docs, a.limit, a.dataset,
                                   a.config), indent=2))


if __name__ == "__main__":
    main()
