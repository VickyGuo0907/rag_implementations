"""Tests for the benchmark dataset loader (no LLM or RAGAS required)."""

import json

from evaluation.run_benchmark import DEFAULT_DATASET, load_dataset


def test_load_dataset_limit(tmp_path):
    p = tmp_path / "q.json"
    p.write_text(json.dumps([{"question": f"q{i}", "ground_truth": f"a{i}"} for i in range(5)]))
    assert load_dataset(str(p), 2) == (["q0", "q1"], ["a0", "a1"])
    assert len(load_dataset(str(p), None)[0]) == 5


def test_shipped_dataset_is_well_formed():
    questions, truths = load_dataset(DEFAULT_DATASET, None)
    assert len(questions) == len(truths) == 10
    assert all(q.strip() and t.strip() for q, t in zip(questions, truths, strict=True))
