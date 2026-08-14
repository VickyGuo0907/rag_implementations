"""
Unit tests for the shared RAGResult / Document data models and BaseRAG
interface contract.
"""

from core.base_rag import RAGResult, Document, BaseRAG


def make_result():
    return RAGResult(
        query="q",
        answer="a",
        source_documents=[
            Document(content="doc one", metadata={"source": "x"}, score=0.9),
            Document(content="doc two" * 50, metadata={}, score=0.5),
        ],
        metadata={"k": "v"},
        latency_ms=12.3,
        technique="naive_rag",
        framework="langchain",
    )


def test_rag_result_to_dict():
    """to_dict exposes all key fields with truncated source content."""
    d = make_result().to_dict()
    assert d["query"] == "q"
    assert d["answer"] == "a"
    assert d["technique"] == "naive_rag"
    assert d["framework"] == "langchain"
    assert d["latency_ms"] == 12.3
    assert d["num_sources"] == 2
    assert len(d["sources"]) == 2
    assert d["sources"][0]["score"] == 0.9


def test_document_repr_truncates():
    """Document.__repr__ truncates content to a short preview."""
    doc = Document(content="x" * 500, score=0.5)
    assert len(repr(doc)) < 120


def test_rag_result_print_summary_runs(capsys):
    """print_summary writes formatted output without raising."""
    make_result().print_summary()
    out = capsys.readouterr().out
    assert "naive_rag" in out
    assert "Retrieved Sources" in out


def test_base_rag_get_info_contract():
    """get_info returns technique/framework/description keys."""

    class Dummy(BaseRAG):
        """Dummy docstring."""

        TECHNIQUE_NAME = "dummy"
        FRAMEWORK = "langchain"

        def _build_pipeline(self):
            pass

        def index(self, documents, metadatas=None):
            pass

        def _query(self, question):
            pass

    info = Dummy(config={}).get_info()
    assert set(info.keys()) == {"technique", "framework", "description", "indexed"}
    assert info["technique"] == "dummy"
    assert info["framework"] == "langchain"