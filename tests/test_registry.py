"""
Tests for the technique registry — the single source of truth for
technique metadata and framework→class mappings.
"""

import importlib
import inspect

from techniques.registry import (
    TECHNIQUES,
    IMPLEMENTATIONS,
    FRAMEWORKS,
    get_techniques,
    get_implementations,
    load_class,
    is_implemented,
    is_registered,
)


def test_all_fourteen_techniques_present():
    """The registry lists all 14 techniques."""
    assert len(TECHNIQUES) == 14


def test_implemented_vs_stub_counts():
    """09 techniques implemented, 05 stubs."""
    implemented = get_techniques(status="implemented")
    stubs = get_techniques(status="stub")
    assert len(implemented) == 9
    assert len(stubs) == 5


def test_implemented_techniques_have_both_frameworks():
    """Every implemented technique maps to both LangChain and LlamaIndex classes."""
    for key, meta in TECHNIQUES.items():
        if meta["status"] != "implemented":
            continue
        for fw in FRAMEWORKS:
            assert is_registered(key, fw), f"{key}/{fw} missing from IMPLEMENTATIONS"


def test_load_class_returns_base_rag_subclass():
    """load_class returns a class that subclasses BaseRAG for all combos."""
    from core.base_rag import BaseRAG

    for tech, fw in get_implementations():
        cls = load_class(tech, fw)
        assert issubclass(cls, BaseRAG), f"{cls} does not subclass BaseRAG"
        assert cls.TECHNIQUE_NAME == tech
        assert cls.FRAMEWORK == fw


def test_load_class_unknown_combo_raises_keyerror():
    """An unimplemented combo raises KeyError."""
    try:
        load_class("graph_rag", "langchain")
        assert False, "graph_rag langchain is a stub, should not load"
    except KeyError:
        pass


def test_is_registered_stub_technique():
    """Stub techniques are registered for zero frameworks."""
    assert not is_registered("adaptive_rag", "langchain")
    assert not is_implemented("adaptive_rag")


def test_class_files_exist_on_disk():
    """The module path for every implementation resolves to a real file."""
    for (tech, fw), dotted in IMPLEMENTATIONS.items():
        module_path = dotted.rsplit(".", 1)[0]
        mod = importlib.import_module(module_path)
        assert mod is not None, f"{module_path} failed to import"


def test_all_langchain_impls_inherit_base_and_set_attributes():
    """Every langchain impl class sets TECHNIQUE_NAME + FRAMEWORK consistently."""
    for tech, fw in get_implementations(framework="langchain"):
        cls = load_class(tech, fw)
        assert inspect.isclass(cls)
        assert cls.FRAMEWORK == "langchain"