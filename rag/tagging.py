"""
tagging.py
----------
Attach the metadata that makes retrieval *outcome-aware*:

  source_type : syllabus | textbook | question_bank | assignment | notes | other
                (guessed from the file path; override with --source-type)
  bloom_level : Remember .. Create, or "" if no cue is found
                (rule-based, from the Bloom's verbs at the start of a question)
  co_id       : which Course Outcome the chunk supports, or "UNTAGGED"
                (needs a CO description file; see `load_co_descriptions`)

These are *heuristic* tags. They are good enough to filter retrieval, but a
teacher should spot-check a sample. `ingest.py` prints a tagging summary so
bad tagging is visible immediately.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional

# Reuse the project's Bloom's verb table so tagging and level assignment agree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mastery"))
from level_mapping import BLOOM_LEVELS, BLOOM_VERBS  # noqa: E402

SOURCE_TYPE_HINTS = [
    ("question_bank", ["question_bank", "question bank", "qbank", "previous_year",
                       "pyq", "past_paper", "past paper", "exam"]),
    ("assignment", ["assignment", "homework", "worksheet", "tutorial", "lab"]),
    ("syllabus", ["syllabus", "curriculum", "course_outcome", "course outcome"]),
    ("textbook", ["textbook", "book", "chapter", "reference"]),
    ("notes", ["notes", "lecture", "slides", "ppt"]),
]


def guess_source_type(path: Path) -> str:
    haystack = str(path).lower().replace("\\", "/")
    for source_type, hints in SOURCE_TYPE_HINTS:
        if any(h in haystack for h in hints):
            return source_type
    return "other"


# Longer / multi-word verbs first so "break down" wins over a stray "use".
_VERB_TO_LEVEL = []
for _level, _verbs in BLOOM_VERBS.items():
    for _v in _verbs:
        _VERB_TO_LEVEL.append((_v, _level))
_VERB_TO_LEVEL.sort(key=lambda x: -len(x[0]))

# Extra verbs commonly seen in exam questions beyond the project's base table.
_EXTRA = {
    "Remember": ["what is", "name", "identify", "write the", "give the"],
    "Understand": ["illustrate", "discuss", "interpret", "differentiate between"],
    "Apply": ["apply", "calculate", "compute", "find", "write a program",
              "evaluate the expression", "trace"],
    "Analyze": ["analyse", "analyze", "distinguish", "why does"],
    "Evaluate": ["evaluate", "judge", "recommend", "which is better"],
    "Create": ["develop", "devise", "build", "create"],
}
for _level, _verbs in _EXTRA.items():
    for _v in _verbs:
        _VERB_TO_LEVEL.append((_v, _level))
_VERB_TO_LEVEL.sort(key=lambda x: -len(x[0]))

_QUESTION_START = re.compile(r"^\s*(?:q(?:uestion)?\.?\s*)?\(?\d+[.)]?\s*|^\s*[a-z][.)]\s*", re.I)


def detect_bloom_level(text: str) -> str:
    """Return the Bloom's level of the first question-like line, else ''.

    Only the opening words of a line are checked, because a verb deep in a
    sentence ("the algorithm uses a stack") is not a question cue.
    """
    for line in text.splitlines():
        line = _QUESTION_START.sub("", line).strip().lower()
        if not line:
            continue
        for verb, level in _VERB_TO_LEVEL:
            if line.startswith(verb + " ") or line.startswith(verb + ","):
                return level
    return ""


# ---------------------------------------------------------------------- #
# CO tagging
# ---------------------------------------------------------------------- #

def load_co_descriptions(path: Optional[Path]) -> Dict[str, str]:
    """Read a JSON file like:
        {"CO1": "Explain stack and queue operations and their applications",
         "CO2": "Implement sorting and searching algorithms"}
    """
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError("CO file must be a non-empty JSON object {co_id: description}")
    return {str(k): str(v) for k, v in data.items()}


def tag_co(chunk_vec, co_vecs: Dict[str, "list"], min_similarity: float) -> str:
    """Pick the CO whose description embedding is closest to the chunk.

    Vectors must be L2-normalised, so the dot product is cosine similarity.
    Below `min_similarity` the chunk is left UNTAGGED rather than forced into
    a wrong CO: a missing tag is safer than a wrong one for fairness checks.
    """
    best_co, best_sim = "UNTAGGED", min_similarity
    for co_id, vec in co_vecs.items():
        sim = sum(a * b for a, b in zip(chunk_vec, vec))
        if sim > best_sim:
            best_co, best_sim = co_id, sim
    return best_co
