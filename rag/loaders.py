"""
loaders.py
----------
Turn course files into plain text "segments" (one per PDF page, PPTX slide,
or whole text/DOCX body) so later steps can keep page/slide numbers as
citation metadata.

Supported: .pdf  .docx  .pptx  .txt  .md
Anything else is skipped with a warning (see ingest.py).

Scanned PDFs (images only) have no extractable text and will come back
empty. They need OCR first; ingest.py reports which files were empty.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".md"}


@dataclass
class Segment:
    text: str
    location: str  # "page 3", "slide 5", or "" for flat text


def _load_pdf(path: Path) -> List[Segment]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    out = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            out.append(Segment(text, f"page {i}"))
    return out


def _load_docx(path: Path) -> List[Segment]:
    import docx

    d = docx.Document(str(path))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    # Tables often hold question lists / marks schemes, so keep them.
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    text = "\n".join(parts).strip()
    return [Segment(text, "")] if text else []


def _load_pptx(path: Path) -> List[Segment]:
    from pptx import Presentation

    prs = Presentation(str(path))
    out = []
    for i, slide in enumerate(prs.slides, start=1):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                t = shape.text_frame.text.strip()
                if t:
                    texts.append(t)
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                texts.append(notes)
        text = "\n".join(texts).strip()
        if text:
            out.append(Segment(text, f"slide {i}"))
    return out


def _load_text(path: Path) -> List[Segment]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return [Segment(text, "")] if text else []


_LOADERS = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".pptx": _load_pptx,
    ".txt": _load_text,
    ".md": _load_text,
}


def load_document(path: Path) -> List[Segment]:
    ext = path.suffix.lower()
    if ext not in _LOADERS:
        raise ValueError(f"Unsupported file type: {ext}")
    return _LOADERS[ext](path)
