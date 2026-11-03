import json
from datetime import date

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..auth import student_only
from ..config import settings
from ..course import CO_TITLES, COURSE
from ..db import get_db
from ..models import Assignment, Submission, User, Variant
from ..services import grading, mastery
from ..services.serialize import grade_view, user_dict

router = APIRouter(prefix="/student", dependencies=[Depends(student_only)])


@router.get("/assignments")
def my_assignments(user: User = Depends(student_only), db: Session = Depends(get_db)):
    rows = db.query(Assignment).filter(Assignment.status.in_(["released", "closed"])).order_by(Assignment.num).all()
    out = []
    for a in rows:
        v = db.get(Variant, f"{a.id}-{user.id}")
        if not v:
            continue
        sub = db.get(Submission, f"sub-{v.id}")
        score = None
        if sub and sub.grade and sub.status != "submitted":
            g = sub.grade
            score = g["teacher_override"]["total"] if g.get("teacher_override") else g["total_score"]
        out.append({"variant_id": v.id, "assignment_id": a.id, "title": a.title, "course": a.course_id, "due": a.due,
                    "total": a.total, "status": sub.status if sub else "pending", "submission_id": sub.id if sub else None,
                    "score": score, "max": sub.grade["max_score"] if sub and sub.grade else a.total})
    return out


@router.get("/assignments/{variant_id}")
def my_variant(variant_id: str, user: User = Depends(student_only), db: Session = Depends(get_db)):
    v = db.get(Variant, variant_id)
    if not v or v.student_id != user.id:
        raise HTTPException(404, "Assignment not found.")
    a = db.get(Assignment, v.assignment_id)
    if a.status not in ("released", "closed"):
        raise HTTPException(403, "This assignment has not been released yet.")
    sub = db.get(Submission, f"sub-{v.id}")
    return {
        "variant": {"variant_id": v.id, "assignment_id": a.id, "student_id": v.student_id, "rev": v.rev,
                    "approved": v.approved, "questions": v.questions},
        "assignment": {"id": a.id, "title": a.title, "due": a.due, "total": a.total, "course": a.course_id},
        "submission": ({"id": sub.id, "status": sub.status, "submitted_at": sub.submitted_at, "answers": sub.answers} if sub else None),
    }


class AnswerIn(BaseModel):
    question_id: str
    text: str = ""


class FileMetaIn(BaseModel):
    question_id: str
    name: str


class SubmitIn(BaseModel):
    variant_id: str
    answers: list[AnswerIn]
    files: list[FileMetaIn] = []


@router.post("/submissions")
async def submit(background: BackgroundTasks,
                 payload: str = Form(..., description="JSON: {variant_id, answers[], files[]}"),
                 files: list[UploadFile] = File(default=[]),
                 user: User = Depends(student_only), db: Session = Depends(get_db)):
    body = SubmitIn.model_validate_json(payload)
    v = db.get(Variant, body.variant_id)
    if not v or v.student_id != user.id:
        raise HTTPException(404, "Assignment not found.")
    a = db.get(Assignment, v.assignment_id)
    if a.status != "released":
        raise HTTPException(409, "This assignment is not open for submissions.")
    sid = f"sub-{v.id}"
    if db.get(Submission, sid):
        raise HTTPException(409, "You have already submitted this assignment.")
    saved = []
    folder = settings.UPLOAD_DIR / "submissions" / sid
    folder.mkdir(parents=True, exist_ok=True)
    for meta, up in zip(body.files, files):
        dest = folder / f"{meta.question_id}_{up.filename or meta.name}".replace("/", "_")
        dest.write_bytes(await up.read())
        saved.append({"question_id": meta.question_id, "name": meta.name, "path": str(dest)})
    db.add(Submission(id=sid, variant_id=v.id, assignment_id=a.id, student_id=user.id, status="submitted",
                      submitted_at=date.today().isoformat(), answers=[x.model_dump() for x in body.answers], files=saved))
    db.commit()
    background.add_task(grading.run_grading, sid)
    return {"submission_id": sid, "status": "submitted"}


@router.get("/submissions/{sub_id}")
def my_submission(sub_id: str, user: User = Depends(student_only), db: Session = Depends(get_db)):
    s = db.get(Submission, sub_id)
    if not s or s.student_id != user.id:
        raise HTTPException(404, "Submission not found.")
    a = db.get(Assignment, s.assignment_id)
    v = db.get(Variant, s.variant_id)
    return {"submission": {"id": s.id, "status": s.status, "submitted_at": s.submitted_at, "regrade_reason": s.regrade_reason},
            "assignment_title": a.title, "answers": s.answers, "questions": v.questions, "grade": grade_view(s)}


class RegradeIn(BaseModel):
    reason: str


@router.post("/submissions/{sub_id}/regrade")
def request_regrade(sub_id: str, body: RegradeIn, user: User = Depends(student_only), db: Session = Depends(get_db)):
    s = db.get(Submission, sub_id)
    if not s or s.student_id != user.id:
        raise HTTPException(404, "Submission not found.")
    if s.status != "ai_graded":
        raise HTTPException(409, "A re-check can only be requested on an AI-graded submission.")
    s.status, s.regrade_reason = "regrade_requested", body.reason.strip()[:1000]
    db.commit()
    return {"status": s.status}


@router.get("/progress")
def my_progress(user: User = Depends(student_only), db: Session = Depends(get_db)):
    cohort = mastery.student_ids(db)
    cos = []
    for co in COURSE["cos"]:
        row = mastery.row_for(db, user.id, co["co_id"], cohort)
        cos.append({**row, "title": CO_TITLES[co["co_id"]], "trend": mastery.history(db, user.id, co["co_id"], 6)})
    by_co = {c["co_id"]: c["mastery"] for c in cos}
    modules = [{"name": m["name"], "co_id": m["co_id"], "completed": round(min(1.0, by_co[m["co_id"]] * m["weight"]), 3)}
               for m in COURSE["modules"]]
    return {"student": user_dict(user), "cos": cos, "modules": modules}
