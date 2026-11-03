from sqlalchemy.orm import Session

from ..models import Assignment, Submission, User, Variant


def user_dict(u: User) -> dict:
    return {"id": u.id, "name": u.name, "email": u.email, "role": u.role}


def assignment_dict(a: Assignment, student_count: int | None = None) -> dict:
    d = {"id": a.id, "title": a.title, "course": a.course_id, "due": a.due, "total": a.total,
         "status": a.status, "cos": a.cos, "topics": a.topics}
    if student_count is not None:
        d["student_count"] = student_count
    return d


def variant_dict(v: Variant) -> dict:
    return {"variant_id": v.id, "assignment_id": v.assignment_id, "student_id": v.student_id,
            "rev": v.rev, "approved": v.approved, "questions": v.questions}


def grade_view(s: Submission) -> dict | None:
    if not s.grade or s.status == "submitted":
        return None
    return {**s.grade, "status": s.status}


def queue_item(db: Session, s: Submission) -> dict:
    student = db.get(User, s.student_id)
    a = db.get(Assignment, s.assignment_id)
    v = db.get(Variant, s.variant_id)
    return {"submission": {"id": s.id, "status": s.status, "regrade_reason": s.regrade_reason},
            "student_name": student.name if student else s.student_id, "assignment_title": a.title,
            "answers": s.answers, "questions": v.questions, "grade": grade_view(s)}
