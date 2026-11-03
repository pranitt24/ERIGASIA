import json
from typing import Dict, Any, Optional


GRADING_SYSTEM_PROMPT = (
    "You are Ergasia AI, a strict and constructive engineering assignment grader. "
    "Evaluate the student's submission solely against the provided question and grading rubric. "
    "Award partial credit where appropriate. Output ONLY valid JSON matching this exact schema:\n"
    "{\n"
    '  "total_score": number,\n'
    '  "max_marks": number,\n'
    '  "criterion_scores": [\n'
    '    {"criterion": "string", "awarded": number, "max": number, "justification": "string"}\n'
    "  ],\n"
    '  "overall_feedback": "string",\n'
    '  "confidence": number\n'
    "}"
)


class AssignmentGrader:
    """
    Stage 5 + Closed Feedback Loop:
      1. Grades student answer against the Stage 3 rubric using `lora_grading_adapter`.
      2. Flags low-confidence or boundary grades for periodic faculty spot-checks.
      3. Updates the student's per-CO mastery in BKTModel.
    """
    def __init__(
        self,
        llm_engine,
        bkt_model: Optional[Any] = None,
        mastery_pass_threshold: float = 0.60,
        spot_check_confidence_threshold: float = 0.75,
    ):
        self.llm = llm_engine
        self.bkt_model = bkt_model
        self.mastery_pass_threshold = mastery_pass_threshold
        self.spot_check_confidence_threshold = spot_check_confidence_threshold

    def grade_and_update_mastery(
        self,
        student_id: str,
        question_record: Dict[str, Any],
        student_answer: str,
        mode: str = "grading",
        update_bkt: bool = True,
    ) -> Dict[str, Any]:
        """
        Grades a single submission and triggers the BKT mastery update.
        """
        q_data = question_record["question_data"]
        co_id = q_data["co_id"]
        max_marks = q_data["total_marks"]
        rubric_json = json.dumps(q_data["rubric"], indent=2)

        user_prompt = (
            f"Course Outcome: {co_id} (Bloom's Level: {q_data['bloom_level']})\n"
            f"Question ({max_marks} Marks):\n{q_data['question_text']}\n\n"
            f"Grading Rubric:\n{rubric_json}\n\n"
            f"Student Submission:\n{student_answer}\n\n"
            "Evaluate the submission against each rubric criterion and return the JSON grade report."
        )

        res = self.llm.generate_json(
            system_prompt=GRADING_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            mode=mode,
        )

        grade_payload = self._normalize_grade(res.get("data"), q_data)
        score_ratio = grade_payload["total_score"] / max_marks if max_marks > 0 else 0.0

        # Binary observation for standard Corbett & Anderson BKT update (1 = mastered step, 0 = not yet)
        bkt_observation = 1 if score_ratio >= self.mastery_pass_threshold else 0

        prior_mastery = question_record.get("bkt_mastery", 0.5)
        posterior_mastery = prior_mastery

        if update_bkt and self.bkt_model is not None:
            posterior_mastery = self._apply_bkt_update(student_id, co_id, bkt_observation, prior_mastery)

        needs_faculty_spot_check = (
            not res["valid_json"]
            or grade_payload["confidence"] < self.spot_check_confidence_threshold
            or (0.48 <= score_ratio <= 0.52)  # Borderline pass/fail cases
        )

        return {
            "student_id": student_id,
            "co_id": co_id,
            "schema_valid": res["valid_json"],
            "total_score": grade_payload["total_score"],
            "max_marks": max_marks,
            "score_ratio": round(score_ratio, 4),
            "criterion_scores": grade_payload["criterion_scores"],
            "overall_feedback": grade_payload["overall_feedback"],
            "needs_faculty_spot_check": needs_faculty_spot_check,
            "bkt_feedback_loop": {
                "observation": bkt_observation,
                "prior_mastery": round(prior_mastery, 4),
                "posterior_mastery": round(posterior_mastery, 4),
                "mastery_delta": round(posterior_mastery - prior_mastery, 4),
            },
        }

    def _apply_bkt_update(self, student_id: str, co_id: str, observation: int, fallback_prior: float) -> float:
        """Calls ergasia/mastery/bkt.py update method regardless of exact signature."""
        if hasattr(self.bkt_model, "update"):
            try:
                updated = self.bkt_model.update(student_id, co_id, bool(observation))
                if isinstance(updated, (int, float)):
                    return float(updated)
                if hasattr(self.bkt_model, "get_mastery"):
                    return float(self.bkt_model.get_mastery(student_id, co_id))
            except TypeError:
                # Fallback if update takes integer 0/1
                updated = self.bkt_model.update(student_id, co_id, int(observation))
                if isinstance(updated, (int, float)):
                    return float(updated)
        return fallback_prior

    @staticmethod
    def _normalize_grade(data: Any, q_data: Dict[str, Any]) -> Dict[str, Any]:
        max_marks = float(q_data["total_marks"])
        if not isinstance(data, dict):
            return {
                "total_score": 0.0,
                "max_marks": max_marks,
                "criterion_scores": [],
                "overall_feedback": "Automated parse fallback; flagged for faculty review.",
                "confidence": 0.0,
            }

        c_scores = data.get("criterion_scores", [])
        if c_scores and isinstance(c_scores, list):
            computed_total = sum(float(c.get("awarded", 0.0)) for c in c_scores)
        else:
            computed_total = float(data.get("total_score", 0.0))

        # Clamp score between [0, max_marks]
        computed_total = max(0.0, min(max_marks, round(computed_total, 2)))

        return {
            "total_score": computed_total,
            "max_marks": max_marks,
            "criterion_scores": c_scores,
            "overall_feedback": str(data.get("overall_feedback", "Evaluated against rubric criteria.")),
            "confidence": float(data.get("confidence", 0.85)),
        }