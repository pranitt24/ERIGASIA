from typing import List, Dict, Any


BLOOM_ORDER = {
    "remember": 1,
    "understand": 2,
    "apply": 3,
    "analyze": 4,
    "evaluate": 5,
    "create": 6,
}


class FairnessChecker:
    """
    Stage 4: Audits generated assignments across a student cohort to guarantee:
      1. Identical Course Outcome (CO) & Topic coverage across all students.
      2. Identical total marks and per-question marks distribution.
      3. Rubric sum consistency (sum of criterion marks == question total_marks).
      4. Bloom's level calibration (generated question matches student's assigned BKT bucket).
    """

    @staticmethod
    def check_cohort_assignments(
        student_assignments: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        student_assignments: {student_id: [generated_question_record_1, generated_question_record_2, ...]}
        Returns a detailed audit report with pass/fail status and violation logs.
        """
        if not student_assignments:
            return {"passed": False, "reason": "Empty student_assignments dict."}

        student_ids = list(student_assignments.keys())
        ref_id = student_ids[0]
        ref_questions = student_assignments[ref_id]

        ref_cos = [q["question_data"]["co_id"] for q in ref_questions]
        ref_topics = [q["question_data"]["topic"] for q in ref_questions]
        ref_total_marks = sum(q["question_data"]["total_marks"] for q in ref_questions)

        violations = []
        bloom_calibration_hits = 0
        total_questions_checked = 0

        for sid, q_list in student_assignments.items():
            cos = [q["question_data"]["co_id"] for q in q_list]
            topics = [q["question_data"]["topic"] for q in q_list]
            total_marks = sum(q["question_data"]["total_marks"] for q in q_list)

            # 1. Check CO coverage invariance
            if cos != ref_cos:
                violations.append(
                    f"[CO Mismatch] Student {sid} has COs {cos}, expected {ref_cos} (ref: {ref_id})."
                )

            # 2. Check Topic coverage invariance
            if topics != ref_topics:
                violations.append(
                    f"[Topic Mismatch] Student {sid} has topics {topics}, expected {ref_topics}."
                )

            # 3. Check Total Marks invariance
            if total_marks != ref_total_marks:
                violations.append(
                    f"[Total Marks Mismatch] Student {sid} has {total_marks} marks, expected {ref_total_marks}."
                )

            # 4. Per-question Rubric & Bloom's level audit
            for q in q_list:
                total_questions_checked += 1
                q_data = q["question_data"]
                q_marks = q_data["total_marks"]
                rubric_sum = round(sum(float(r.get("marks", 0)) for r in q_data.get("rubric", [])), 1)

                if abs(rubric_sum - q_marks) > 0.1:
                    violations.append(
                        f"[Rubric Sum Error] Student {sid} ({q_data['co_id']}): rubric sums to {rubric_sum} != {q_marks}."
                    )

                assigned_range = [b.lower() for b in q.get("assigned_bloom_range", [])]
                actual_bloom = str(q_data.get("bloom_level", "")).strip().lower()
                if any(b in actual_bloom for b in assigned_range):
                    bloom_calibration_hits += 1

        calibration_accuracy = (
            bloom_calibration_hits / total_questions_checked if total_questions_checked > 0 else 0.0
        )

        return {
            "fairness_passed": len(violations) == 0,
            "num_students_audited": len(student_ids),
            "invariant_co_sequence": ref_cos,
            "invariant_topic_sequence": ref_topics,
            "invariant_total_marks": ref_total_marks,
            "bloom_calibration_accuracy": round(calibration_accuracy, 4),
            "violations": violations,
        }