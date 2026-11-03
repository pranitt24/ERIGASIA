"""Auto-grading: runs in the background right after a student submits."""
import traceback
from ..config import settings
from ..db import SessionLocal
from ..models import Submission, Variant
from ..pipeline.registry import get_pipeline
from . import mastery


def run_grading(submission_id: str) -> None:
    db = SessionLocal()
    try:
        sub = db.get(Submission, submission_id)
        if not sub:
            return
        variant = db.get(Variant, sub.variant_id)
        print(f"[grading] Grading submission {submission_id} for student {sub.student_id}...")
        result = get_pipeline().grader.grade(variant.questions, sub.answers, sub.files)
        result.setdefault("teacher_override", None)
        result["submission_id"] = sub.id
        sub.grade = result
        sub.status = "ai_graded"
        db.commit()
        print(f"[grading] Saved AI grade for {submission_id}: {result.get('total_score')}/{result.get('max_score')}")

        # Update BKT mastery per question without risking rolling back the saved grade
        try:
            by_id = {q["question_id"]: q for q in variant.questions}
            for pq in result.get("per_question", []):
                q = by_id.get(pq.get("question_id"))
                if q:
                    correct = pq["max"] > 0 and (pq["score"] / pq["max"]) >= settings.BKT_CORRECT_THRESHOLD
                    mastery.update(db, sub.student_id, q["co_id"], correct)
        except Exception as bkt_err:
            print(f"[grading] BKT mastery update warning for {submission_id}: {bkt_err}")
    except Exception as e:
        db.rollback()
        print(f"[grading] ERROR for {submission_id}: {e}")
        traceback.print_exc()
    finally:
        db.close()