"""Variant generation: the slow job the teacher starts from 'Generate variants'."""
import time
import uuid

from sqlalchemy.orm import Session

from ..config import settings
from ..course import PO_OF
from ..db import SessionLocal
from ..models import Assignment, User, Variant
from ..pipeline.common import rubric_for
from ..pipeline.registry import get_pipeline
from . import mastery

STEPS = ["Retrieving course material", "Reading each student\u2019s mastery",
         "Drafting questions per student", "Running fairness checks"]
_jobs: dict[str, dict] = {}  # in-memory is fine for one server process; move to Redis/DB if you scale out


def _pause():
    if settings.impl("GENERATOR") == "mock":
        time.sleep(settings.MOCK_STEP_DELAY)


def profile(db: Session, student: User) -> dict:
    vec = mastery.vector(db, student.id)
    return {"student_id": student.id, "name": student.name, "mastery": vec,
            "levels": {co: mastery.level_range(m) for co, m in vec.items()}}


def finalize_questions(raw: list[dict]) -> list[dict]:
    out = []
    for i, q in enumerate(raw):
        out.append({
            "question_id": f"q{i + 1}", "text": q["text"], "co_id": q["co_id"], "po_ids": PO_OF.get(q["co_id"], []),
            "bloom_level": q["bloom_level"], "marks": int(q["marks"]),
            "rubric": q.get("rubric") or rubric_for(q["bloom_level"], int(q["marks"])), "topic": q["topic"],
        })
    return out


def retrieve_context(a: Assignment) -> dict:
    r = get_pipeline().retriever
    return {i: r.retrieve(a.course_id, p["co_id"], p["topic"]) for i, p in enumerate(a.qspec)}


def make_variant(db: Session, a: Assignment, student: User, context: dict, rev: int = 0) -> Variant:
    raw = get_pipeline().generator.generate_variant(a.qspec, profile(db, student), context, rev)
    if len(raw) != len(a.qspec):
        raise ValueError(f"Generator returned {len(raw)} questions, expected {len(a.qspec)}")
    return Variant(id=f"{a.id}-{student.id}", assignment_id=a.id, student_id=student.id, rev=rev,
                   approved=False, questions=finalize_questions(raw))


def new_job(assignment_id: str) -> str:
    jid = f"job-{uuid.uuid4().hex[:10]}"
    _jobs[jid] = {"status": "pending", "step": 0, "steps": STEPS, "assignment_id": assignment_id, "error": None}
    return jid


def get_job(jid: str) -> dict | None:
    return _jobs.get(jid)


def run_generation(job_id: str, assignment_id: str) -> None:
    """Runs in a background thread (FastAPI BackgroundTasks)."""
    job = _jobs[job_id]
    db = SessionLocal()
    a = db.get(Assignment, assignment_id)
    try:
        job.update(status="running", step=0)
        ctx = retrieve_context(a); _pause()
        job["step"] = 1
        students = db.query(User).filter_by(role="student").order_by(User.id).all(); _pause()
        job["step"] = 2
        db.query(Variant).filter_by(assignment_id=a.id).delete()
        for s in students:
            db.add(make_variant(db, a, s, ctx))
        db.commit(); _pause()
        job["step"] = 3; _pause()
        a.status = "in_review"; db.commit()
        job.update(status="done", step=4)
    except Exception as e:  # surface to the UI, leave the assignment editable
        db.rollback()
        a = db.get(Assignment, assignment_id)
        a.status = "draft"; db.commit()
        job.update(status="failed", error=str(e))
    finally:
        db.close()
