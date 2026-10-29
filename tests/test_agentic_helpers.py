"""
Unit tests for Agentic RAG's framework-agnostic ReAct loop: step parsing,
the safe calculator, and the loop itself driven by a scripted fake LLM.
No LLM or vector store required.
"""

import importlib

import pytest

react = importlib.import_module("techniques.13_agentic_rag.react_loop")


# ---------------------------------------------------------------------------
# parse_step
# ---------------------------------------------------------------------------

def test_parse_step_action():
    text = "Thought: need data\nAction: vector_retrieval\nAction Input: what is RAG"
    assert react.parse_step(text) == ("action", "vector_retrieval", "what is RAG")


def test_parse_step_final_answer():
    text = "Thought: I now know the answer\nFinal Answer: 42"
    assert react.parse_step(text) == ("final", "42", "")


def test_parse_step_discards_hallucinated_observation():
    text = (
        "Action: calculator\nAction Input: 2+2\n"
        "Observation: 4\nThought: done\nFinal Answer: 4"
    )
    assert react.parse_step(text) == ("action", "calculator", "2+2")


def test_parse_step_first_marker_wins():
    text = "Final Answer: early\nAction: calculator\nAction Input: 1+1"
    assert react.parse_step(text) == ("final", "early\nAction: calculator\nAction Input: 1+1", "")


def test_parse_step_strips_thinking():
    text = "<think>Action: x\nAction Input: y</think>Final Answer: ok"
    assert react.parse_step(text) == ("final", "ok", "")


def test_parse_step_unformatted_output_becomes_final():
    assert react.parse_step("Just an answer.") == ("final", "Just an answer.", "")


# ---------------------------------------------------------------------------
# safe_calculate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expr,expected", [
    ("2 + 3 * 4", "14"),
    ("(1 + 2) ** 3", "27"),
    ("-5 + 2", "-3"),
    ("7 // 2", "3"),
    ("10 / 4", "2.5"),
])
def test_safe_calculate_valid(expr, expected):
    assert react.safe_calculate(expr) == expected


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hi')",
    "open('x')",
    "1 / 0",
    "2 ** 10000",
    "1 +",
    "abc",
])
def test_safe_calculate_rejects_unsafe_or_invalid(expr):
    assert react.safe_calculate(expr).startswith("Error:")


# ---------------------------------------------------------------------------
# run_react
# ---------------------------------------------------------------------------

def _scripted(responses):
    """Fake LLM that returns canned responses in order and records prompts."""
    prompts = []
    it = iter(responses)

    def complete(prompt):
        prompts.append(prompt)
        return next(it)

    complete.prompts = prompts
    return complete


def _tools():
    return {
        "calculator": react.Tool("calculator", "math", react.safe_calculate),
        "boom": react.Tool("boom", "always fails", lambda _: 1 / 0),
    }


def test_run_react_tool_then_answer():
    llm = _scripted([
        "Thought: compute\nAction: calculator\nAction Input: 6 * 7",
        "Thought: done\nFinal Answer: 42",
    ])
    answer, steps = react.run_react(llm, _tools(), "6*7?")
    assert answer == "42"
    assert [s["observation"] for s in steps] == ["42"]
    assert "Observation: 42" in llm.prompts[1]


def test_run_react_immediate_answer_uses_no_tools():
    llm = _scripted(["Final Answer: hi"])
    assert react.run_react(llm, _tools(), "hello") == ("hi", [])


def test_run_react_unknown_tool_is_reported_not_raised():
    llm = _scripted(["Action: nope\nAction Input: x", "Final Answer: ok"])
    _, steps = react.run_react(llm, _tools(), "q")
    assert steps[0]["observation"].startswith("Error: unknown tool 'nope'")


def test_run_react_tool_exception_becomes_observation():
    llm = _scripted(["Action: boom\nAction Input: x", "Final Answer: recovered"])
    answer, steps = react.run_react(llm, _tools(), "q")
    assert answer == "recovered"
    assert steps[0]["observation"].startswith("Error: tool 'boom' failed")


def test_run_react_forces_final_answer_when_budget_exhausted():
    llm = _scripted([
        "Action: calculator\nAction Input: 1+1",
        "Action: calculator\nAction Input: 2+2",
        " forced result",
    ])
    answer, steps = react.run_react(llm, _tools(), "q", max_iterations=2)
    assert answer == "forced result"
    assert len(steps) == 2
    assert "I have used all my allowed steps" in llm.prompts[-1]


def test_run_react_empty_llm_output_returns_explicit_message():
    answer, steps = react.run_react(_scripted([""]), _tools(), "q")
    assert answer == react.EMPTY_ANSWER
    assert steps == []
