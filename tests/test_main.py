"""
Tests for the main CLI entry point: command routing and technique listing.
"""

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import main


def test_create_parser_has_all_commands():
    """All documented subcommands are exposed by the CLI parser."""
    parser = main.create_parser()
    sub = {a.dest: a for a in parser._actions if isinstance(a, argparse._SubParsersAction)}
    choices = list(sub.values())[0].choices
    for cmd in ("run", "list", "info", "eval", "config"):
        assert cmd in choices


def test_cmd_list_includes_all_techniques(capsys):
    """cmd_list prints every registry technique."""
    main.cmd_list(argparse.Namespace())
    out = capsys.readouterr().out
    for name in ("Naive RAG", "Adaptive RAG", "GraphRAG", "Multi-modal RAG"):
        assert name in out


def test_cmd_info_known_technique(capsys):
    """cmd_info prints details for a registered technique."""
    main.cmd_info(argparse.Namespace(technique="naive_rag"))
    out = capsys.readouterr().out
    assert "Naive RAG" in out
    assert "langchain" in out.lower()


def test_cmd_info_unknown_technique_exits():
    """cmd_info exits for an unknown technique."""
    with pytest.raises(SystemExit) as exc_info:
        main.cmd_info(argparse.Namespace(technique="not_a_technique"))
    assert exc_info.value.code == 1


def test_load_class_equivalent_to_registry():
    """main.initialize_rag resolves the same classes the registry does."""
    from techniques.registry import load_class

    rag = main.initialize_rag("naive_rag", "langchain", config_path=None)
    assert rag.__class__ is load_class("naive_rag", "langchain")
