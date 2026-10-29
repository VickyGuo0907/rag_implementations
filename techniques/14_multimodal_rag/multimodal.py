"""
Framework-agnostic multimodal pipeline pieces for Multi-modal RAG.

Strategy ("summarize, embed, answer with the original"):
  * text   — chunked and embedded as usual
  * tables — pipe tables found in the text; a vision-LLM summary is embedded,
             the raw table is kept aside and used at answer time
  * images — markdown ![alt](path) references and images embedded in PDF pages;
             a vision-LLM caption is embedded, the image itself is attached to
             the answer prompt so the model can actually look at it

All embeddings stay text-only, so any configured embedding model works. Both the
LangChain and LlamaIndex implementations share this module and differ only in
vector store / retrieval plumbing.
"""

import base64
import hashlib
import logging
import mimetypes
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp"})
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")
_PDF_IMAGE_DIR = Path(tempfile.gettempdir()) / "rag_multimodal_images"

CAPTION_PROMPT = (
    "Describe this image for a search index. State what it shows, any visible text, "
    "axis labels, numbers, and key trends. Be factual and concise (max 4 sentences)."
)
TABLE_PROMPT = (
    "Summarize this table for a search index in 2-3 sentences: what it measures, the "
    "column names, and the notable values or trends.\n\n{table}"
)
ANSWER_PROMPT = (
    "Answer the question using ONLY the context below and any attached images. "
    "If they do not contain enough information, say so.\n\nContext:\n{context}\n\n"
    "Question: {question}"
)


@dataclass(frozen=True)
class Item:
    """One indexable unit. `text` is what gets embedded (raw text, summary, or caption)."""
    kind: str                      # "text" | "table" | "image"
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Hit:
    """A retrieved unit resolved back to its original content for generation."""
    kind: str
    text: str                      # embedded text (chunk, summary, or caption)
    raw: str                       # table markdown / image path / chunk text
    metadata: dict = field(default_factory=dict)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning traces some LMStudio models emit."""
    return _THINK_TAG_RE.sub("", text).strip()


def extract_tables(text: str) -> list[str]:
    """Return every markdown pipe table (header + separator + rows) found in `text`."""
    tables, block = [], []
    for line in [*text.split("\n"), ""]:
        if line.strip().startswith("|"):
            block.append(line.rstrip())
            continue
        if len(block) >= 3 and _TABLE_SEP_RE.match(block[1]):
            tables.append("\n".join(block))
        block = []
    return tables


def extract_markdown_images(text: str, base_dir: Path) -> list[tuple[str, Path]]:
    """Return (alt_text, resolved_path) for each local, existing image reference."""
    found = []
    for alt, ref in _MD_IMAGE_RE.findall(text):
        if ref.startswith(("http://", "https://", "data:")):
            continue
        path = (base_dir / ref).resolve()
        if path.suffix.lower() in IMAGE_EXTS and path.is_file():
            found.append((alt, path))
        else:
            logger.warning("[multimodal] skipping missing/unsupported image '%s'", ref)
    return found


def extract_pdf_page_images(pdf_path: Path, page: int, min_bytes: int) -> list[tuple[str, Path]]:
    """Save images embedded in one PDF page to a cache dir; skip icons below `min_bytes`."""
    from pypdf import PdfReader

    try:
        images = PdfReader(str(pdf_path)).pages[page].images
        _PDF_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        saved = []
        for img in images:
            if len(img.data) < min_bytes:
                continue
            digest = hashlib.sha1(img.data).hexdigest()[:16]
            path = _PDF_IMAGE_DIR / f"{digest}{Path(img.name).suffix or '.png'}"
            path.write_bytes(img.data)
            saved.append((f"{pdf_path.name} page {page + 1}", path))
        return saved
    except Exception as exc:  # noqa: BLE001 - one bad PDF page must not abort indexing
        logger.warning("[multimodal] could not read images from %s p%s: %s", pdf_path, page, exc)
        return []


def image_data_uri(path: Path) -> str:
    """Encode an image file as a data: URI for an OpenAI-style image_url part."""
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


class VisionLLM:
    """Thin OpenAI-compatible client (LMStudio) for captioning, summarizing, and answering."""

    def __init__(self, lm_cfg: dict, model: str | None = None, max_tokens: int | None = None):
        from openai import OpenAI

        self.model = model or lm_cfg["model"]
        self.max_tokens = max_tokens or lm_cfg.get("max_tokens", 4000)
        self.client = OpenAI(
            base_url=lm_cfg["base_url"], api_key=lm_cfg.get("api_key", "lm-studio"),
            timeout=lm_cfg.get("timeout", 120),
        )

    def _chat(self, content: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, temperature=0, max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        return strip_thinking(resp.choices[0].message.content or "")

    def caption_image(self, path: Path) -> str:
        """Caption an image for retrieval."""
        return self._chat([{"type": "text", "text": CAPTION_PROMPT},
                           {"type": "image_url", "image_url": {"url": image_data_uri(path)}}])

    def summarize_table(self, table: str) -> str:
        """Summarize a table for retrieval."""
        return self._chat([{"type": "text", "text": TABLE_PROMPT.format(table=table)}])

    def answer(self, question: str, context: str, images: list[Path]) -> str:
        """Answer from text context plus attached images."""
        parts: list[dict] = [{"type": "text",
                              "text": ANSWER_PROMPT.format(context=context, question=question)}]
        parts += [{"type": "image_url", "image_url": {"url": image_data_uri(p)}} for p in images]
        return self._chat(parts)


def _describe(describe, arg, fallback: str) -> str:
    """Call a describer; a failed or empty description degrades to `fallback`, never aborts."""
    try:
        return describe(arg) or fallback
    except Exception as exc:  # noqa: BLE001 - e.g. server down, model lacks vision
        logger.warning("[multimodal] description failed (%s); using fallback", exc)
        return fallback


def _image_items(text: str, meta: dict, describer: VisionLLM, cfg: dict) -> list[Item]:
    """Collect image Items referenced by (markdown) or embedded in (PDF) one document."""
    source = Path(meta["source"]) if meta.get("source") else None
    refs: list[tuple[str, Path]] = []
    if source and source.suffix.lower() in (".md", ".txt"):
        refs = extract_markdown_images(text, source.parent)
    elif source and source.suffix.lower() == ".pdf" and isinstance(meta.get("page"), int):
        refs = extract_pdf_page_images(source, meta["page"], cfg.get("min_image_bytes", 5000))
    return [
        Item("image", f"{alt}. " + _describe(describer.caption_image, path, "").strip(),
             {"modality": "image", "source": str(meta.get("source", "")), "raw": str(path)})
        for alt, path in refs
    ]


def _table_items(text: str, meta: dict, describer: VisionLLM) -> list[Item]:
    """Collect table Items (summary embedded, raw table kept) from one document."""
    return [
        Item("table", _describe(describer.summarize_table, t, t[:300]),
             {"modality": "table", "source": str(meta.get("source", "")), "raw": t})
        for t in extract_tables(text)
    ]


def build_items(
    documents: list[str],
    metadatas: list[dict] | None,
    describer: VisionLLM,
    cfg: dict,
) -> list[Item]:
    """
    Turn raw documents into text, table, and image Items.

    Honors `table_extraction` / `image_captioning` in the technique config.
    Text Items keep the full document text (tables stay inline so plain text
    retrieval still works as a fallback).
    """
    metas = metadatas or [{} for _ in documents]
    items: list[Item] = []
    for text, meta in zip(documents, metas, strict=True):
        items.append(Item("text", text, {"modality": "text", **_scalar(meta)}))
        if cfg.get("table_extraction", True):
            items += _table_items(text, meta, describer)
        if cfg.get("image_captioning", True):
            items += _image_items(text, meta, describer, cfg)
    return items


def _scalar(meta: dict) -> dict:
    """Keep only vector-store-safe scalar metadata values."""
    return {k: v for k, v in meta.items() if isinstance(v, str | int | float | bool)}


def build_context(hits: list[Hit], max_images: int) -> tuple[str, list[Path]]:
    """Assemble the answer context and the image files to attach (deduplicated, capped)."""
    blocks, images = [], []
    for hit in hits:
        if hit.kind == "table":
            blocks.append(f"[Table]\n{hit.raw}")
        elif hit.kind == "image":
            blocks.append(f"[Image: {hit.text}]")
            if hit.raw not in map(str, images) and len(images) < max_images:
                images.append(Path(hit.raw))
        else:
            blocks.append(hit.raw)
    return "\n\n---\n\n".join(blocks), images
