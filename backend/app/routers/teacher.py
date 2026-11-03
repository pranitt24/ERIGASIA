from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..auth import teacher_only
from ..course import CO_IDS, CO_TITLES, COURSE, PO_OF
from ..db import get_db
from ..models import Assignment, KBFile, Submission, User, Variant
from ..pipeline.common import rubric_for
from ..services import fairness, generation, kb, mastery
from ..services.planner import plan_questions
from ..services.serialize import assignment_dict, queue_item, variant_dict

router = APIRouter(prefix="/teacher", dependencies=[Depends(teacher_only)])
TARGET = 0.6  # mastery share used for the CO-PO table; see INTEGRATION.md (attainment method is an open decision)


def _students(db: Session) -> list[User]:
    return db.query(User).filter_by(role="student").order_by(User.id).all()


def _assignment(db: Session, aid: str) -> Assignment:
    a = db.get(Assignment, aid)
    if not a:
        raise HTTPException(404, "Assignment not found.")
    return a


def _variants(db: Session, aid: str) -> list[Variant]:
    return db.query(Variant).filter_by(assignment_id=aid).order_by(Variant.student_id).all()


def _queue(db: Session) -> list[dict]:
    subs = db.query(Submission).filter(Submission.status.in_(["ai_graded", "regrade_requested"])).all()
    subs.sort(key=lambda s: (s.status != "regrade_requested", (s.grade or {}).get("ai_confidence", 1)))
    return [queue_item(db, s) for s in subs if s.grade]


# ---------- course + dashboard ----------
@router.get("/course")
def course():
    return {"id": COURSE["id"], "name": COURSE["name"], "cos": CO_TITLES, "po_of": PO_OF}


@router.get("/dashboard")
def dashboard(db: Session = Depends(get_db)):
    rows = db.query(Assignment).order_by(Assignment.num).all()
    return {"assignments": [assignment_dict(a) for a in rows], "queue": _queue(db)[:4], "student_count": len(_students(db))}


# ---------- assignments ----------
class CreateAssignmentIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    course: str = "CS201"
    cos: list[str] = Field(min_length=1)
    topics: list[str] = []
    total: int = Field(ge=10, le=500)
    due: str


@router.post("/assignments")
def create_assignment(body: CreateAssignmentIn, db: Session = Depends(get_db)):
    unknown = [c for c in body.cos if c not in CO_IDS]
    if unknown:
        raise HTTPException(422, f"Unknown course outcome: {', '.join(unknown)}")
    num = (db.query(Assignment).count() and max(a.num for a in db.query(Assignment).all()) or 0) + 1
    plan = plan_questions(body.cos, body.topics, body.total)
    a = Assignment(id=f"A{num}", num=num, course_id=body.course, title=body.title.strip(), due=body.due, total=body.total,
                   status="draft", cos=sorted(set(body.cos)), topics=sorted({p["topic"] for p in plan}), qspec=plan)
    db.add(a); db.commit()
    return assignment_dict(a, len(_students(db)))


@router.get("/assignments/{aid}")
def get_assignment(aid: str, db: Session = Depends(get_db)):
    return assignment_dict(_assignment(db, aid), len(_students(db)))


@router.post("/assignments/{aid}/generate")
def generate(aid: str, background: BackgroundTasks, db: Session = Depends(get_db)):
    a = _assignment(db, aid)
    if a.status not in ("draft", "in_review"):
        raise HTTPException(409, f"Cannot generate variants while the assignment is {a.status}.")
    a.status = "generating"; db.commit()
    jid = generation.new_job(aid)
    background.add_task(generation.run_generation, jid, aid)
    return {"job_id": jid}


@router.get("/assignments/{aid}/variants")
def variants(aid: str, db: Session = Depends(get_db)):
    a = _assignment(db, aid)
    vs = _variants(db, aid)
    names = {u.id: u.name for u in _students(db)}
    return {"assignment": assignment_dict(a),
            "variants": [{**variant_dict(v), "student_name": names.get(v.student_id, v.student_id),
                          "mastery": mastery.vector(db, v.student_id)} for v in vs],
            "fairness": fairness.check(a, vs)}


class PatchVariantIn(BaseModel):
    question_id: str | None = None
    text: str | None = None
    marks: int | None = Field(default=None, ge=1, le=500)
    approved: bool | None = None


def _editable(db: Session, vid: str) -> tuple[Variant, Assignment]:
    v = db.get(Variant, vid)
    if not v:
        raise HTTPException(404, "Variant not found.")
    a = db.get(Assignment, v.assignment_id)
    if a.status != "in_review":
        raise HTTPException(409, "Variants can only be changed while the assignment is in review.")
    return v, a


@router.patch("/variants/{vid}")
def patch_variant(vid: str, body: PatchVariantIn, db: Session = Depends(get_db)):
    v, a = _editable(db, vid)
    if body.question_id:
        qs = [dict(q) for q in v.questions]
        q = next((x for x in qs if x["question_id"] == body.question_id), None)
        if not q:
            raise HTTPException(404, "Question not found.")
        if body.text is not None:
            if not body.text.strip():
                raise HTTPException(422, "Question text cannot be empty.")
            q["text"] = body.text.strip()
        if body.marks is not None:
            q["marks"] = body.marks
            q["rubric"] = rubric_for(q["bloom_level"], body.marks)
        v.questions = qs
        v.approved = False  # any edit needs a fresh approval
    if body.approved is not None:
        v.approved = body.approved
    db.commit()
    return {"variant": variant_dict(v), "fairness": fairness.check(a, _variants(db, a.id))}


@router.post("/variants/{vid}/regenerate")
def regenerate(vid: str, db: Session = Depends(get_db)):
    v, a = _editable(db, vid)
    student = db.get(User, v.student_id)
    try:
        nv = generation.make_variant(db, a, student, generation.retrieve_context(a), rev=v.rev + 1)
    except Exception as e:
        raise HTTPException(502, f"Generation failed: {e}")
    v.questions, v.rev, v.approved = nv.questions, nv.rev, False
    db.commit()
    return {"variant": variant_dict(v), "fairness": fairness.check(a, _variants(db, a.id))}


@router.post("/assignments/{aid}/release")
def release(aid: str, db: Session = Depends(get_db)):
    a = _assignment(db, aid)
    if a.status != "in_review":
        raise HTTPException(409, f"Cannot release an assignment that is {a.status}.")
    vs = _variants(db, aid)
    f = fairness.check(a, vs)
    if not (f["same_topics"] and f["same_total_marks"]):
        raise HTTPException(409, "Fix the fairness issues before releasing.")
    if not vs or any(not v.approved for v in vs):
        raise HTTPException(409, "Approve every variant before releasing.")
    a.status = "released"; db.commit()
    return {"status": "released"}


# ---------- grading ----------
@router.get("/grading-queue")
def grading_queue(db: Session = Depends(get_db)):
    return _queue(db)


class PatchGradeIn(BaseModel):
    scores: list[float | None] | None = None
    feedback: str | None = None
    confirm: bool = False


@router.patch("/grades/{sub_id}")
def patch_grade(sub_id: str, body: PatchGradeIn, user: User = Depends(teacher_only), db: Session = Depends(get_db)):
    s = db.get(Submission, sub_id)
    if not s or not s.grade:
        raise HTTPException(404, "Graded submission not found.")
    g = dict(s.grade)
    crits = [dict(c) for c in g["per_criterion"]]
    if body.scores is not None:
        if len(body.scores) != len(crits):
            raise HTTPException(422, "Send one score per rubric criterion.")
        for c, sc in zip(crits, body.scores):
            if sc is not None:
                c["score"] = max(0.0, min(float(sc), float(c["max"])))
    total = sum(c["score"] for c in crits)
    g["per_criterion"] = crits
    if body.feedback is not None:
        g["feedback"] = body.feedback
    g["teacher_override"] = {"total": total, "by": user.name} if total != g["total_score"] else None
    s.grade = g
    if body.confirm:
        s.status = "teacher_confirmed"
    db.commit()
    return {**g, "status": s.status}


# ---------- analytics ----------
@router.get("/analytics/mastery")
def analytics_mastery(db: Session = Depends(get_db)):
    return {"cos": CO_TITLES, "students": [{"id": u.id, "name": u.name} for u in _students(db)], "rows": mastery.class_rows(db)}


@router.get("/analytics/co-po")
def analytics_co_po(db: Session = Depends(get_db)):
    """Placeholder attainment: share of students with mastery >= TARGET. Real OBE attainment is normally
    computed from marks per CO; swap this body once the team decides the method."""
    rows = mastery.class_rows(db)
    n = max(1, len(_students(db)))
    out = []
    for co in COURSE["cos"]:
        share = sum(1 for r in rows if r["co_id"] == co["co_id"] and r["mastery"] >= TARGET) / n
        level = 3 if share >= 0.7 else 2 if share >= 0.6 else 1 if share >= 0.5 else 0
        out.append({"co_id": co["co_id"], "title": co["title"], "po_ids": co["po_ids"], "pct_at_target": round(share, 4),
                    "attainment_level": level, "target_met": level >= 2})
    return {"target": TARGET, "rows": out}


# ---------- knowledge base ----------
def _kb_dict(k: KBFile) -> dict:
    return {"id": k.id, "name": k.name, "kind": k.kind, "size": k.size, "status": k.status}


@router.get("/knowledge-base")
def kb_list(db: Session = Depends(get_db)):
    return [_kb_dict(k) for k in db.query(KBFile).order_by(KBFile.created_at.desc()).all()]


@router.post("/knowledge-base")
async def kb_upload(background: BackgroundTasks, file: UploadFile = File(...), db: Session = Depends(get_db)):
    row = kb.save_upload(db, file.filename or "upload", await file.read())
    background.add_task(kb.run_ingest, row.id)
    return _kb_dict(row)
