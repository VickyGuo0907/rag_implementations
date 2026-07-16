"""
RAGAS Evaluation Framework
============================
Evaluates any RAG implementation using RAGAS metrics:
  - Faithfulness       : Is the answer grounded in the retrieved context?
  - Answer Relevancy   : Is the answer relevant to the question?
  - Context Recall     : Are all relevant docs retrieved?
  - Context Precision  : Are retrieved docs precise / non-noisy?

Usage:
    from evaluation.ragas_evaluator import RAGASEvaluator
    evaluator = RAGASEvaluator()
    scores = evaluator.evaluate(rag_instance, test_questions, ground_truths)
    evaluator.print_report(scores)

Reference: https://docs.ragas.io/en/latest/
"""

import asyncio
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config_loader import ConfigLoader
from core.base_rag import BaseRAG

logger = logging.getLogger(__name__)


@contextmanager
def _standard_asyncio_loop():
    """
    Temporarily replace the current thread's uvloop event loop (if any)
    with a standard-library asyncio loop.

    Streamlit installs uvloop.EventLoopPolicy globally at startup. RAGAS's
    executor module calls nest_asyncio.apply() at *import time*, and
    nest_asyncio can never patch a uvloop.Loop -- it's a Cython extension
    that doesn't subclass asyncio.BaseEventLoop, so the isinstance check in
    nest_asyncio._patch_loop always fails for it. That patch only needs to
    succeed once (nest_asyncio marks the loop *class* as patched), so this
    must wrap the first import of `ragas` anywhere in the process, not just
    calls to `ragas.evaluate()`.

    Streamlit's ScriptRunner thread has the uvloop *policy* installed but
    may not have any loop instance actually set on it yet -- in that case
    asyncio.get_event_loop() raises RuntimeError instead of returning a
    uvloop.Loop, so we can't rely on inspecting the current loop to decide
    whether to act; we must check the policy instead.
    """
    try:
        import uvloop
    except ImportError:
        yield
        return

    current_policy = asyncio.get_event_loop_policy()
    if not isinstance(current_policy, uvloop.EventLoopPolicy):
        yield
        return

    try:
        previous_loop = asyncio.get_event_loop()
    except RuntimeError:
        previous_loop = None

    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    std_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(std_loop)
    try:
        yield
    finally:
        std_loop.close()
        asyncio.set_event_loop_policy(current_policy)
        asyncio.set_event_loop(previous_loop)


@dataclass
class EvaluationResult:
    """Results from a RAGAS evaluation run."""
    technique: str
    framework: str
    num_samples: int
    scores: Dict[str, float]
    raw_results: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def print_report(self) -> None:
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  RAGAS Evaluation Report")
        print(f"  Technique : {self.technique} ({self.framework})")
        print(f"  Samples   : {self.num_samples}")
        print(sep)
        for metric, score in self.scores.items():
            bar = "█" * int(score * 20)
            print(f"  {metric:<25} {score:.4f}  {bar}")
        print(sep)
        overall = sum(self.scores.values()) / len(self.scores) if self.scores else 0
        print(f"  {'Overall Average':<25} {overall:.4f}")
        print(sep + "\n")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "technique": self.technique,
            "framework": self.framework,
            "num_samples": self.num_samples,
            "scores": self.scores,
            "metadata": self.metadata,
        }


class RAGASEvaluator:
    """
    Evaluates RAG pipelines using the RAGAS framework.

    RAGAS evaluates without requiring labeled answers (reference-free),
    using the LLM itself as an evaluator for most metrics.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.cfg = ConfigLoader.get(config_path)
        self.eval_cfg = self.cfg.evaluation
        self._check_ragas_installed()

    def _check_ragas_installed(self) -> None:
        try:
            with _standard_asyncio_loop():
                import ragas
        except ImportError:
            logger.warning(
                "RAGAS not installed. Run: pip install ragas\n"
                "Evaluation will use fallback metrics."
            )

    def evaluate(
            self,
            rag: BaseRAG,
            questions: List[str],
            ground_truths: Optional[List[str]] = None,
            contexts_override: Optional[List[List[str]]] = None,
    ) -> EvaluationResult:
        """
        Run RAGAS evaluation on a RAG pipeline.

        Args:
            rag: Any BaseRAG subclass instance (must be indexed already).
            questions: List of evaluation questions.
            ground_truths: Optional reference answers for answer_correctness metric.
            contexts_override: Optional pre-retrieved contexts (skips RAG retrieval).

        Returns:
            EvaluationResult with per-metric scores.
        """
        logger.info(f"Evaluating {rag.TECHNIQUE_NAME} on {len(questions)} questions...")

        # Step 1: Run RAG pipeline on all questions
        answers = []
        contexts = []

        for i, question in enumerate(questions):
            logger.info(f"  [{i + 1}/{len(questions)}] {question[:60]}...")
            try:
                result = rag.query(question)
                answers.append(result.answer)
                contexts.append([doc.content for doc in result.source_documents])
            except Exception as e:
                logger.error(f"  Query failed: {e}")
                answers.append("")
                contexts.append([])

        # Step 2: Run RAGAS evaluation (or fall back to heuristic metrics)
        # Note: RAGAS has compatibility issues with LMStudio and custom LLMs.
        # We default to fallback metrics unless explicitly enabled in config.
        eval_cfg = self.eval_cfg
        use_ragas = eval_cfg.get("enabled", False)  # Disabled by default due to compatibility

        if use_ragas:
            try:
                scores = self._run_ragas(questions, answers, contexts, ground_truths)
            except Exception as e:
                logger.warning(f"RAGAS evaluation not available: {e}. Using fallback metrics.")
                scores = self._fallback_metrics(questions, answers, contexts)
        else:
            scores = self._fallback_metrics(questions, answers, contexts)

        # Step 3: Save results
        result = EvaluationResult(
            technique=rag.TECHNIQUE_NAME,
            framework=rag.FRAMEWORK,
            num_samples=len(questions),
            scores=scores,
            metadata={"questions": questions, "answers": answers},
        )
        self._save_results(result)
        return result

    def _run_ragas(
            self,
            questions: List[str],
            answers: List[str],
            contexts: List[List[str]],
            ground_truths: Optional[List[str]],
    ) -> Dict[str, float]:
        """Run actual RAGAS evaluation."""
        from datasets import Dataset
        with _standard_asyncio_loop():
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
            )
        from core.llm_client import get_langchain_llm

        eval_cfg = self.eval_cfg
        metrics_cfg = eval_cfg.get("metrics", {})

        # Build metric list from config
        metric_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_recall": context_recall,
            "context_precision": context_precision,
        }
        selected_metrics = [
            m for name, m in metric_map.items()
            if metrics_cfg.get(name, True)
        ]

        # Remove metrics that require ground truth if not provided
        # context_recall and context_precision both require 'reference' column
        if not ground_truths:
            selected_metrics = [
                m for m in selected_metrics
                if m not in (context_recall, context_precision)
            ]
            # Ensure we have at least some metrics
            if not selected_metrics:
                selected_metrics = [faithfulness, answer_relevancy]

        # Build RAGAS dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        if ground_truths:
            data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data)

        # Configure RAGAS metrics to use LMStudio LLM instead of OpenAI
        try:
            llm = get_langchain_llm()
            # Set LLM for metrics that need it
            for metric in selected_metrics:
                if hasattr(metric, "llm"):
                    metric.llm = llm
        except Exception as e:
            logger.warning(f"Could not configure LLM for RAGAS metrics: {e}")

        with _standard_asyncio_loop():
            result = evaluate(dataset, metrics=selected_metrics)

        return dict(result)

    def _fallback_metrics(
            self,
            questions: List[str],
            answers: List[str],
            contexts: List[List[str]],
    ) -> Dict[str, float]:
        """
        Fallback heuristic metrics when RAGAS is unavailable.
        These provide basic quality signals without external LLM evaluation.
        """
        logger.info("Using fallback heuristic metrics (RAGAS evaluation skipped)")

        scores = {}

        # Answer completeness: ratio of non-empty answers
        non_empty = sum(1 for a in answers if a.strip())
        scores["answer_completeness"] = non_empty / len(answers) if answers else 0.0

        # Context retrieval: ratio of questions with retrieved context
        has_context = sum(1 for c in contexts if c)
        scores["context_retrieval"] = has_context / len(contexts) if contexts else 0.0

        # Answer length signal: longer answers often indicate more thorough responses
        avg_len = sum(len(a.split()) for a in answers) / len(answers) if answers else 0
        # Normalize: 50+ words is good, 100+ is excellent
        scores["answer_length"] = min(avg_len / 100, 1.0)

        # Context density: avg number of contexts retrieved per question
        avg_ctx_count = sum(len(c) for c in contexts) / len(contexts) if contexts else 0
        # Normalize: 5+ contexts is good (typical retrieval k=5)
        scores["context_density"] = min(avg_ctx_count / 10, 1.0)

        return scores

    def _save_results(self, result: EvaluationResult) -> None:
        """Persist evaluation results to disk."""
        output_dir = Path(self.eval_cfg.get("output_dir", "./evaluation_results"))
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = output_dir / f"{result.technique}_{result.framework}_eval.json"
        with open(filename, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Evaluation results saved to {filename}")

    def compare(self, results: List[EvaluationResult]) -> None:
        """Print a side-by-side comparison of multiple evaluation results."""
        if not results:
            return

        metrics = list(results[0].scores.keys())
        col_width = 20

        print("\n" + "=" * (col_width * (len(results) + 1) + 5))
        print(f"  RAG Technique Comparison")
        print("=" * (col_width * (len(results) + 1) + 5))

        # Header
        header = f"  {'Metric':<25}"
        for r in results:
            header += f"{r.technique[:col_width]:<{col_width}}"
        print(header)
        print("-" * (col_width * (len(results) + 1) + 5))

        # Scores
        for metric in metrics:
            row = f"  {metric:<25}"
            for r in results:
                score = r.scores.get(metric, 0.0)
                row += f"{score:.4f}{' ':>{col_width - 6}}"
            print(row)

        print("=" * (col_width * (len(results) + 1) + 5) + "\n")
