"""
chunking.py
-----------
Split text into overlapping chunks without cutting sentences in half where
possible. Size is in characters (roughly 4 characters per token, so the
default 1200 is about 300 tokens, a comfortable unit for RAG context).

Strategy: split on blank lines (paragraphs), then pack paragraphs into a
chunk until the size limit. A single paragraph longer than the limit is split
on sentence boundaries. The tail of each chunk is repeated at the start of
the next one (overlap) so an idea that straddles a boundary is not lost.
"""

from __future__ import annotations

import re
from typing import List

DEFAULT_CHUNK_CHARS = 1200
DEFAULT_OVERLAP_CHARS = 200

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# A new question starts at a line like "1.", "2)", "Q3.", "Q.4", "Question 5:".
_QUESTION_BOUNDARY = re.compile(
    r"(?m)^(?=\s*(?:q(?:uestion)?\.?\s*)?\(?\d{1,3}[.):]\s)", re.I
)


def _split_long(paragraph: str, limit: int) -> List[str]:
    """Break an oversized paragraph into pieces <= limit, on sentence
    boundaries when possible, hard-cutting only if one sentence is too long."""
    pieces, current = [], ""
    for sent in _SENTENCE_SPLIT.split(paragraph):
        while len(sent) > limit:  # one giant "sentence": hard cut
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sent[:limit])
            sent = sent[limit:]
        if len(current) + len(sent) + 1 <= limit:
            current = f"{current} {sent}".strip()
        else:
            if current:
                pieces.append(current)
            current = sent
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str,
               chunk_chars: int = DEFAULT_CHUNK_CHARS,
               overlap_chars: int = DEFAULT_OVERLAP_CHARS) -> List[str]:
    if overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must be smaller than chunk_chars")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: List[str] = []
    for p in paragraphs:
        units.extend(_split_long(p, chunk_chars) if len(p) > chunk_chars else [p])

    chunks, current = [], ""
    for unit in units:
        if len(current) + len(unit) + 2 <= chunk_chars:
            current = f"{current}\n\n{unit}".strip()
        else:
            if current:
                chunks.append(current)
            tail = current[-overlap_chars:] if current else ""
            # Start the next chunk from a word boundary inside the tail.
            if tail and " " in tail:
                tail = tail.split(" ", 1)[1]
            current = f"{tail}\n\n{unit}".strip() if tail else unit
    if current:
        chunks.append(current)
    return chunks


def split_questions(text: str,
                    chunk_chars: int = DEFAULT_CHUNK_CHARS) -> List[str]:
    """Split a question bank / assignment into ONE CHUNK PER QUESTION.

    One question per chunk keeps Bloom's tagging accurate (a 1200-character
    chunk would mix several questions) and makes retrieved examples clean.
    If no numbered questions are found, falls back to normal chunking.
    An over-long question (e.g. with a big code listing) is chunked normally.
    """
    parts = [p.strip() for p in _QUESTION_BOUNDARY.split(text) if p.strip()]
    if len(parts) <= 1:
        return chunk_text(text, chunk_chars)
    out: List[str] = []
    for part in parts:
        out.extend(chunk_text(part, chunk_chars) if len(part) > chunk_chars else [part])
    return out
