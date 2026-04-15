"""
Document Loader & Text Splitter Factory
=========================================
Handles loading documents from various sources and splitting them into chunks.
Supports multiple chunking strategies: recursive, sentence, semantic, proposition.

Usage:
    from core.document_loader import load_documents, get_text_splitter
    docs = load_documents("./data/sample_docs")
    chunks = get_text_splitter().split_documents(docs)
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
from core.config_loader import ConfigLoader


# ---------------------------------------------------------------------------
# Document Loading
# ---------------------------------------------------------------------------

def load_documents(source: str, glob: str = "**/*.*") -> List[Any]:
    """
    Load documents from a file, directory, or URL using LangChain loaders.

    Args:
        source: Path to file or directory, or a URL string.
        glob: Glob pattern for directory loading.

    Returns:
        List of LangChain Document objects.
    """
    source_path = Path(source)

    if source_path.is_dir():
        from langchain_community.document_loaders import DirectoryLoader, TextLoader
        loader = DirectoryLoader(
            str(source_path),
            glob=glob,
            loader_cls=TextLoader,
            show_progress=True,
        )
        return loader.load()

    suffix = source_path.suffix.lower()

    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(str(source_path)).load()

    elif suffix in (".txt", ".md"):
        from langchain_community.document_loaders import TextLoader
        return TextLoader(str(source_path)).load()

    elif suffix == ".csv":
        from langchain_community.document_loaders import CSVLoader
        return CSVLoader(str(source_path)).load()

    elif suffix in (".docx", ".doc"):
        from langchain_community.document_loaders import Docx2txtLoader
        return Docx2txtLoader(str(source_path)).load()

    elif source.startswith("http"):
        from langchain_community.document_loaders import WebBaseLoader
        return WebBaseLoader(source).load()

    else:
        from langchain_community.document_loaders import TextLoader
        return TextLoader(str(source_path)).load()


def load_texts(texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[Any]:
    """
    Wrap raw strings in LangChain Document objects.

    Args:
        texts: List of raw text strings.
        metadatas: Optional metadata dicts (same length as texts).
    """
    from langchain_core.documents import Document as LCDocument
    metas = metadatas or [{}] * len(texts)
    return [LCDocument(page_content=t, metadata=m) for t, m in zip(texts, metas)]


# ---------------------------------------------------------------------------
# Text Splitters
# ---------------------------------------------------------------------------

def get_text_splitter(strategy: Optional[str] = None):
    """
    Returns a LangChain text splitter based on config strategy.

    Strategies:
        - recursive (default): RecursiveCharacterTextSplitter
        - sentence:            SentenceTransformersTokenTextSplitter
        - semantic:            SemanticChunker (embedding-based)
        - token:               TokenTextSplitter

    Returns:
        LangChain TextSplitter instance
    """
    cfg = ConfigLoader.get()
    doc_cfg = cfg.document
    strat = strategy or doc_cfg.get("chunk_strategy", "recursive")
    chunk_size = doc_cfg.get("chunk_size", 512)
    chunk_overlap = doc_cfg.get("chunk_overlap", 50)

    if strat == "recursive":
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=doc_cfg.get("separators", ["\n\n", "\n", " ", ""]),
        )

    elif strat == "sentence":
        from langchain_text_splitters import SentenceTransformersTokenTextSplitter
        return SentenceTransformersTokenTextSplitter(
            chunk_overlap=chunk_overlap,
            tokens_per_chunk=chunk_size,
        )

    elif strat == "semantic":
        from langchain_experimental.text_splitter import SemanticChunker
        from core.embeddings import get_langchain_embeddings
        sem_cfg = doc_cfg.get("semantic", {})
        return SemanticChunker(
            embeddings=get_langchain_embeddings(),
            breakpoint_threshold_type=sem_cfg.get("breakpoint_threshold_type", "percentile"),
            breakpoint_threshold_amount=sem_cfg.get("breakpoint_threshold_amount", 95),
        )

    elif strat == "token":
        from langchain_text_splitters import TokenTextSplitter
        return TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    else:
        raise ValueError(f"Unknown chunk strategy: {strat}")


# ---------------------------------------------------------------------------
# LlamaIndex Document Loading
# ---------------------------------------------------------------------------

def load_llamaindex_documents(source: str):
    """
    Load documents using LlamaIndex readers.

    Returns:
        List of LlamaIndex Document objects.
    """
    from llama_index.core import SimpleDirectoryReader

    source_path = Path(source)
    if source_path.is_dir():
        return SimpleDirectoryReader(str(source_path)).load_data()
    else:
        return SimpleDirectoryReader(input_files=[str(source_path)]).load_data()
