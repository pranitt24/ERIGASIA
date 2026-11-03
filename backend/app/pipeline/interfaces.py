"""
The seams between the web app and the ML pipeline.

The web app only ever calls these four interfaces (through registry.get_pipeline()).
To plug in real ML code, implement the matching class in real.py and set
<COMPONENT>_IMPL=real in .env. Nothing in routers/ or the frontend changes.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypedDict


class PlanItem(TypedDict):
    co_id: str     # "CO2"
    topic: str     # "Binary search trees"
    marks: int


class StudentProfile(TypedDict):
    student_id: str
    name: str
    mastery: dict[str, float]          # {"CO1": 0.36, ...}
    levels: dict[str, list[str]]       # {"CO1": ["Remember", "Understand"], ...}  (LevelAssignment.bloom_level_range)


@dataclass
class Chunk:
    text: str
    source: str
    score: float = 0.0


class Retriever(Protocol):
    def retrieve(self, course_id: str, co_id: str, topic: str, k: int = 4) -> list[Chunk]:
        """Top-k knowledge-base chunks for a topic under a CO."""


class QuestionGenerator(Protocol):
    def generate_variant(
        self, plan: list[PlanItem], student: StudentProfile, context: dict[int, list[Chunk]], rev: int = 0
    ) -> list[dict]:
        """
        One question per plan item, in plan order, personalised to `student`.
        Each dict MUST have:  text, bloom_level, co_id, topic, marks
        Each dict MAY have:   rubric  [{"criterion": str, "max_marks": int}]  (sum must equal marks)
        `context[i]` holds retrieved chunks for plan item i. `rev` increments on every Regenerate click.
        The web app fills in question_id, po_ids and a default rubric if they are missing.
        """


class Grader(Protocol):
    def grade(self, questions: list[dict], answers: list[dict], files: list[dict]) -> dict:
        """
        questions: the variant's questions (with rubric)
        answers:   [{"question_id": "q1", "text": "..."}]
        files:     [{"question_id": "q1", "name": "...", "path": "/abs/path"}]   (typed answer may be empty if a file was sent)
        Returns:
          {
            "total_score": float, "max_score": float,
            "per_criterion": [{"criterion": "Q1: Correct concept", "score": 2.0, "max": 3, "comment": "..."}],
            "per_question":  [{"question_id": "q1", "score": 7.5, "max": 10}],
            "feedback": "constructive text",
            "ai_confidence": 0.0-1.0,
          }
        """


class KnowledgeBase(Protocol):
    def ingest(self, path: Path, kind: str, course_id: str) -> None:
        """Chunk, embed and store one uploaded file. Raise on failure (status becomes 'failed')."""
