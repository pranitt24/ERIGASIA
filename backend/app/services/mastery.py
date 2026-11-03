"""
Mastery access for the whole app. Routers never touch BKT directly; they call these functions.

MASTERY_BACKEND=fallback  -> small built-in BKT stored in SQLite (works today)
MASTERY_BACKEND=ergasia   -> your mastery/ package, via mastery_ergasia.py (fill in the TODOs there)
"""
from datetime import datetime

from sqlalchemy.orm import Session

from ..config import settings
from ..course import CO_IDS
from ..models import MasteryHistory, MasteryState, User

BLOOM = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
P_L0, P_T, P_G, P_S = 0.30, 0.15, 0.20, 0.10  # fallback BKT parameters


def level_range(m: float) -> list[str]:
    """Fallback mapping from mastery to a Bloom's range. Replace with level_mapping.py output when wired."""
    if m < 0.20: return [BLOOM[0], BLOOM[0]]
    if m < 0.40: return [BLOOM[0], BLOOM[1]]
    if m < 0.55: return [BLOOM[1], BLOOM[2]]
    if m < 0.70: return [BLOOM[2], BLOOM[3]]
    if m < 0.85: return [BLOOM[3], BLOOM[4]]
    return [BLOOM[4], BLOOM[5]]


def _use_ergasia() -> bool:
    return settings.MASTERY_BACKEND == "ergasia"


def get(db: Session, student_id: str, co_id: str) -> float:
    if _use_ergasia():
        from . import mastery_ergasia as e
        return e.get_mastery(student_id, co_id)
    row = db.get(MasteryState, (student_id, co_id))
    return row.p_known if row else P_L0


def vector(db: Session, student_id: str) -> dict[str, float]:
    return {co: get(db, student_id, co) for co in CO_IDS}


def update(db: Session, student_id: str, co_id: str, correct: bool) -> float:
    """Call after a question is graded."""
    if _use_ergasia():
        from . import mastery_ergasia as e
        new = e.update_mastery(student_id, co_id, correct)
    else:
        p = get(db, student_id, co_id)
        post = (p * (1 - P_S) / (p * (1 - P_S) + (1 - p) * P_G)) if correct else (p * P_S / (p * P_S + (1 - p) * (1 - P_G)))
        new = post + (1 - post) * P_T
        row = db.get(MasteryState, (student_id, co_id))
        if row:
            row.p_known = new
        else:
            db.add(MasteryState(student_id=student_id, co_id=co_id, p_known=new))
    db.add(MasteryHistory(student_id=student_id, co_id=co_id, p_known=new, at=datetime.utcnow()))
    db.commit()
    return new


def history(db: Session, student_id: str, co_id: str, n: int = 6) -> list[float]:
    rows = (db.query(MasteryHistory).filter_by(student_id=student_id, co_id=co_id)
            .order_by(MasteryHistory.at.desc(), MasteryHistory.id.desc()).limit(n).all())
    vals = [r.p_known for r in reversed(rows)]
    if not vals:
        vals = [get(db, student_id, co_id)]
    return [vals[0]] * (n - len(vals)) + vals  # pad the front so the chart always has n points


def student_ids(db: Session) -> list[str]:
    return [u.id for u in db.query(User).filter_by(role="student").order_by(User.id).all()]


def row_for(db: Session, student_id: str, co_id: str, cohort: list[str]) -> dict:
    """Same shape as LevelAssignment.as_dict() in the mastery module."""
    if _use_ergasia():
        from . import mastery_ergasia as e
        return e.level_row(student_id, co_id, cohort)
    m = get(db, student_id, co_id)
    below = sum(1 for s in cohort if s != student_id and get(db, s, co_id) < m)
    return {"student_id": student_id, "co_id": co_id, "mastery": round(m, 4),
            "bloom_level_range": level_range(m), "percentile": round(below / max(1, len(cohort) - 1), 4)}


def class_rows(db: Session) -> list[dict]:
    cohort = student_ids(db)
    return [row_for(db, s, co, cohort) for s in cohort for co in CO_IDS]
