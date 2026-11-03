"""Fairness rule: every variant has the same (CO, topic) coverage and the same total marks.
This one is real, deterministic and needs no ML. Extend it (e.g. difficulty spread) if the team wants."""
from ..models import Assignment, Variant


def check(assignment: Assignment, variants: list[Variant]) -> dict:
    want = sorted((p["co_id"], p["topic"]) for p in assignment.qspec)
    per = []
    for v in variants:
        got = sorted((q["co_id"], q["topic"]) for q in v.questions)
        total = sum(q["marks"] for q in v.questions)
        per.append({"student_id": v.student_id, "topics_ok": got == want, "marks_ok": total == assignment.total, "total": total})
    return {"same_topics": all(p["topics_ok"] for p in per), "same_total_marks": all(p["marks_ok"] for p in per),
            "expected_total": assignment.total, "per_student": per}
