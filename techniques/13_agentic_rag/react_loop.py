"""
Framework-agnostic ReAct loop for Agentic RAG.

Text-based Thought / Action / Observation protocol rather than native tool
calling, because local LMStudio models are unreliable at structured tool calls.
Both the LangChain and LlamaIndex implementations plug their LLM and retriever
into `run_react`, so the agent behavior is identical across frameworks.
"""

import ast
import logging
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ACTION_RE = re.compile(r"Action:\s*([A-Za-z_]\w*)\s*\n\s*Action Input:\s*(.+)", re.IGNORECASE)
_FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.DOTALL | re.IGNORECASE)

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_MAX_EXPONENT = 100
EMPTY_ANSWER = (
    "The agent produced no answer (empty LLM response). Reasoning models can exhaust "
    "max_tokens while thinking; use a non-thinking model or raise lmstudio.max_tokens."
)

REACT_PROMPT = """Answer the question by reasoning step by step. You may use these tools:

{tool_descriptions}

Use EXACTLY this format:

Thought: what you need to find out next
Action: one of [{tool_names}]
Action Input: the input for the tool

You will then receive an Observation. Repeat Thought/Action/Action Input as many
times as needed. When you have enough information, finish with:

Thought: I now know the answer
Final Answer: the answer to the original question

Base factual claims ONLY on tool observations. Stop after writing Action Input.

Question: {question}
{scratchpad}"""

FORCE_FINAL_SUFFIX = (
    "\nThought: I have used all my allowed steps. I must answer with what I have.\n"
    "Final Answer:"
)


@dataclass(frozen=True)
class Tool:
    """A named capability the agent can invoke with a single string input."""
    name: str
    description: str
    fn: Callable[[str], str]


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning traces some LMStudio models emit."""
    return _THINK_TAG_RE.sub("", text).strip()


def parse_step(text: str) -> tuple[str, str, str]:
    """
    Parse one LLM turn into ("final", answer, "") or ("action", name, input).

    Anything after a hallucinated "Observation:" is discarded. Whichever of
    Action / Final Answer appears first wins. Output matching neither format is
    treated as the final answer, so format-ignoring models still return something.
    """
    cleaned = strip_thinking(text).split("Observation:")[0].strip()
    action = _ACTION_RE.search(cleaned)
    final = _FINAL_RE.search(cleaned)
    if action and (not final or action.start() < final.start()):
        return "action", action.group(1), action.group(2).strip()
    if final:
        return "final", final.group(1).strip(), ""
    return "final", cleaned, ""


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate a whitelisted arithmetic AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        value = _eval_node(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    raise ValueError("only numbers and + - * / // % ** are allowed")


def safe_calculate(expression: str) -> str:
    """Evaluate an arithmetic expression via a whitelisted AST (never eval())."""
    try:
        result = _eval_node(ast.parse(expression.strip(), mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
        return f"Error: {exc}"
    return str(result)


def _call_tool(tools: dict[str, Tool], name: str, tool_input: str) -> str:
    """Run a tool, converting unknown names and tool failures into observations."""
    tool = tools.get(name)
    if tool is None:
        return f"Error: unknown tool '{name}'. Available: {', '.join(tools)}."
    try:
        return tool.fn(tool_input)
    except Exception as exc:  # noqa: BLE001 - the agent should see and recover from tool errors
        logger.warning("[agentic_rag] tool '%s' failed: %s", name, exc)
        return f"Error: tool '{name}' failed: {exc}"


def run_react(
    llm_complete: Callable[[str], str],
    tools: dict[str, Tool],
    question: str,
    max_iterations: int = 5,
) -> tuple[str, list[dict]]:
    """
    Run the ReAct loop and return (answer, steps).

    Each step is {"thought", "action", "action_input", "observation"}. If the
    iteration budget runs out, one extra call forces a final answer.
    """
    tool_descriptions = "\n".join(f"- {t.name}: {t.description}" for t in tools.values())
    tool_names = ", ".join(tools)
    steps: list[dict] = []
    scratchpad = ""

    def build_prompt() -> str:
        return REACT_PROMPT.format(
            tool_descriptions=tool_descriptions, tool_names=tool_names,
            question=question, scratchpad=scratchpad,
        )

    for _ in range(max_iterations):
        raw = strip_thinking(llm_complete(build_prompt()))
        kind, a, b = parse_step(raw)
        if kind == "final":
            return a or EMPTY_ANSWER, steps
        observation = _call_tool(tools, a, b)
        thought = raw.split("Action:")[0].replace("Thought:", "").strip()
        steps.append({"thought": thought, "action": a, "action_input": b,
                      "observation": observation})
        scratchpad += f"{raw.split('Observation:')[0].strip()}\nObservation: {observation}\n"

    scratchpad += FORCE_FINAL_SUFFIX
    _, answer, _ = parse_step("Final Answer:" + strip_thinking(llm_complete(build_prompt())))
    return answer or EMPTY_ANSWER, steps
