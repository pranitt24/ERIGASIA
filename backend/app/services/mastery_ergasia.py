"""
Adapter to the existing mastery/ package (BKT + level mapping).
Reads/writes SQLite MasteryState directly as the source of truth and applies
Corbett & Anderson BKT updates via ergasia/mastery/bkt.py.
"""
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
ERGASIA_DIR = BACKEND_DIR.parent
REPO_ROOT = ERGASIA_DIR.parent

for p in [str(ERGASIA_DIR), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from mastery.bkt import BKTModel
from ..db import SessionLocal
from ..models import MasteryState

_model = BKTModel()

# Corbett & Anderson standard BKT parameters when updating from current prior
P_T = 0.15  # p_transit
P_S = 0.10  # p_slip
P_G = 0.20  # p_guess


def bloom_range_for_mastery(m: float) -> list[str]:
    """Maps mastery [0, 1] across all 6 Bloom levels so different students get distinct levels."""
    if m < 0.25:
        return ["Remember", "Remember"]
    if m < 0.40:
        return ["Remember", "Understand"]
    if m < 0.55:
        return ["Understand", "Apply"]
    if m < 0.70:
        return ["Apply", "Analyze"]
    if m < 0.85:
        return ["Analyze", "Evaluate"]
    return ["Evaluate", "Create"]


def get_mastery(student_id: str, co_id: str) -> float:
    with SessionLocal() as db:
        row = db.get(MasteryState, (student_id, co_id))
        if row is not None:
            return float(row.p_known)
    # Deterministic spread if a new CO (like CO4..CO6) wasn't seeded for this student yet
    h = sum(ord(c) * (i + 1) for i, c in enumerate(f"{student_id}_{co_id}"))
    default_val = round(0.18 + (h % 75) / 100.0, 4)
    with SessionLocal() as db:
        db.add(MasteryState(student_id=student_id, co_id=co_id, p_known=default_val))
        db.commit()
    return default_val


def update_mastery(student_id: str, co_id: str, correct: bool) -> float:
    p = get_mastery(student_id, co_id)
    if correct:
        posterior = (p * (1.0 - P_S)) / max(1e-6, (p * (1.0 - P_S) + (1.0 - p) * P_G))
    else:
        posterior = (p * P_S) / max(1e-6, (p * P_S + (1.0 - p) * (1.0 - P_G)))
    new_val = round(min(0.99, max(0.05, posterior + (1.0 - posterior) * P_T)), 4)

    with SessionLocal() as db:
        row = db.get(MasteryState, (student_id, co_id))
        if row:
            row.p_known = new_val
        else:
            db.add(MasteryState(student_id=student_id, co_id=co_id, p_known=new_val))
        db.commit()
    return new_val


def level_row(student_id: str, co_id: str, cohort: list[str]) -> dict:
    mastery_by_student = {sid: get_mastery(sid, co_id) for sid in cohort}
    m = mastery_by_student.get(student_id, get_mastery(student_id, co_id))
    below = sum(1 for s, val in mastery_by_student.items() if s != student_id and val < m)
    pct = round(below / max(1, len(cohort) - 1), 4)
    return {
        "student_id": student_id,
        "co_id": co_id,
        "mastery": round(m, 4),
        "bloom_level_range": bloom_range_for_mastery(m),
        "percentile": pct,
    }