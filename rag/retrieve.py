"""
retrieve.py
-----------
The bridge between Stage 2 (mastery) and Stage 3 (generation).

Input : one student's `LevelAssignment.as_dict()` from mastery/level_mapping.py
        {student_id, co_id, mastery, bloom_level_range: [low, high], suggested_verbs, ...}
Output: grounding material for the question-generation prompt:
          - "context"  : textbook / notes / syllabus chunks for the student's CO
          - "examples" : past questions for that CO at the target Bloom's level
                         (style and difficulty references ONLY)
          - "fallbacks": any relaxed filter, so callers and the fairness layer
                         can see when retrieval was thin

Retrieval is deliberately *metadata-filtered first, similarity-ranked second*:
the CO and Bloom's filters guarantee outcome-aware results, and the embedding
similarity only chooses the best chunks inside that filtered set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from embedder import Embedder
from store import VectorStore, build_where

CONTENT_TYPES = ["syllabus", "textbook", "notes", "other"]
EXAMPLE_TYPES = ["question_bank", "assignment"]

# What kind of material to favour, by the top of the student's Bloom's range.
# Low mastery -> foundations; high mastery -> trade-offs and edge cases.
# These phrases are heuristics: tune them once you can see real retrieval output.
TIER_HINTS = {
    "Understand": "basic definition, introduction, core concept explained simply",
    "Analyze": "worked example, application, comparison of approaches",
    "Create": "design trade-offs, edge cases, real-world application, limitations",
}
DEFAULT_HINT = TIER_HINTS["Analyze"]


def _shape(hit: dict) -> dict:
    m = hit["metadata"]
    return {
        "text": hit["text"],
        "source_file": m.get("source_file", ""),
        "location": m.get("location", ""),
        "source_type": m.get("source_type", ""),
        "bloom_level": m.get("bloom_level", ""),
        "co_id": m.get("co_id", ""),
        "score": hit["score"],
    }


class Retriever:
    def __init__(self, db_path, embedder: Embedder, course: Optional[str] = None):
        self.store = VectorStore(db_path, embedder)
        self.course = course
        co_file = Path(db_path) / "co_descriptions.json"
        self.co_desc: Dict[str, str] = (
            json.loads(co_file.read_text(encoding="utf-8")) if co_file.exists() else {}
        )

    # ------------------------------------------------------------------ #

    def _query(self, text: str, k: int, **filters) -> List[dict]:
        where = build_where(course=self.course, **filters)
        return self.store.query(text, k=k, where=where)

    def retrieve_for_student(self, assignment: dict,
                             k_context: int = 4, k_examples: int = 2) -> dict:
        co_id = assignment["co_id"]
        low, high = assignment["bloom_level_range"]
        hint = TIER_HINTS.get(high, DEFAULT_HINT)
        query = f"{self.co_desc.get(co_id, co_id)}. {hint}"
        fallbacks: List[str] = []

        # 1) Grounding content for this CO.
        ctx = self._query(query, k_context, co_id=co_id, source_type=CONTENT_TYPES)
        if len(ctx) < k_context:
            seen = {h["id"] for h in ctx}
            extra = [h for h in self._query(query, k_context, source_type=CONTENT_TYPES)
                     if h["id"] not in seen]
            if extra:
                fallbacks.append(f"content: only {len(ctx)} chunk(s) tagged {co_id}; "
                                 f"filled the rest from untagged/other COs")
            ctx = (ctx + extra)[:k_context]

        # 2) Example questions at the target Bloom's level.
        ex = self._query(query, k_examples, co_id=co_id, source_type=EXAMPLE_TYPES,
                         bloom_level=[low, high])
        if not ex:
            ex = self._query(query, k_examples, co_id=co_id, source_type=EXAMPLE_TYPES)
            if ex:
                fallbacks.append(f"examples: none at {low}-{high} for {co_id}; "
                                 f"used other Bloom's levels")
        if not ex:
            ex = self._query(query, k_examples, source_type=EXAMPLE_TYPES)
            if ex:
                fallbacks.append(f"examples: none for {co_id}; used other COs")

        return {
            "student_id": assignment.get("student_id"),
            "co_id": co_id,
            "bloom_level_range": [low, high],
            "suggested_verbs": assignment.get("suggested_verbs", []),
            "query_used": query,
            "context": [_shape(h) for h in ctx],
            "examples": [_shape(h) for h in ex],
            "fallbacks": fallbacks,
        }


def format_prompt_context(result: dict) -> str:
    """Render retrieval output as the text block that goes into the generation
    prompt. Source tags ([C1], [E1]) let the model's output cite what it used."""
    lines = ["REFERENCE MATERIAL (ground the question in this; do not invent facts):"]
    for i, c in enumerate(result["context"], 1):
        loc = f", {c['location']}" if c["location"] else ""
        lines.append(f"[C{i}] ({c['source_file']}{loc})\n{c['text']}")
    if result["examples"]:
        lines.append("\nEXAMPLE QUESTIONS (match the style and cognitive level only; "
                     "do NOT copy or lightly reword them):")
        for i, e in enumerate(result["examples"], 1):
            lvl = f" [{e['bloom_level']}]" if e["bloom_level"] else ""
            lines.append(f"[E{i}]{lvl} {e['text']}")
    return "\n\n".join(lines)
