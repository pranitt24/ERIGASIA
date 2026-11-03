import os
import sys
import json
import inspect

# 1. Ensure both `ergasia_code` and `ergasia_code/ergasia` are on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ERGASIA_PKG_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(ERGASIA_PKG_DIR, ".."))

for p in [REPO_ROOT, ERGASIA_PKG_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ergasia.llm.engine import ErgasiaLLMEngine
from ergasia.pipeline.generator import QuestionGenerator
from ergasia.pipeline.fairness import FairnessChecker
from ergasia.pipeline.grader import AssignmentGrader

# Import Stage 2 BKT & Level Mapping
try:
    from ergasia.mastery.bkt import BKTModel
    from ergasia.mastery.level_mapping import assign_levels_for_class
    HAS_STAGE2_MODULE = True
except ImportError as e:
    print(f"[Warning] Could not import Stage 2 module: {e}")
    HAS_STAGE2_MODULE = False


def find_adapter_dir(adapter_name: str) -> str:
    """Locates the trained adapter folder whether it's in root, finetuning/, or pipeline/."""
    search_bases = [
        REPO_ROOT,
        ERGASIA_PKG_DIR,
        os.path.join(ERGASIA_PKG_DIR, "finetuning"),
        os.path.join(REPO_ROOT, "finetuning"),
        CURRENT_DIR,
    ]
    for base in search_bases:
        candidate = os.path.join(base, adapter_name)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(REPO_ROOT, adapter_name)


def build_demo_cohort():
    """Creates 3 students across the 3 BKT mastery tiers for CO1."""
    if HAS_STAGE2_MODULE:
        bkt = BKTModel()
        student_histories = [
            ("S001_Low", [0, 0, 0]),
            ("S002_Mid", [1, 0, 1]),
            ("S003_High", [1, 1, 1, 1]),
        ]
        mastery_by_student = {}
        for sid, history in student_histories:
            m = 0.5
            for obs in history:
                res = bkt.update(sid, "CO1", bool(obs))
                if isinstance(res, (int, float)):
                    m = float(res)
            if hasattr(bkt, "get_mastery"):
                m = float(bkt.get_mastery(sid, "CO1"))
            mastery_by_student[sid] = m

        # Inspect parameter order of assign_levels_for_class
        sig = list(inspect.signature(assign_levels_for_class).parameters.keys())
        if sig and "co" in sig[0].lower():
            assignments = assign_levels_for_class("CO1", mastery_by_student)
        else:
            assignments = assign_levels_for_class(mastery_by_student, "CO1")

        if isinstance(assignments, dict):
            assignments = list(assignments.values())

        payloads = [p.as_dict() if hasattr(p, "as_dict") else p for p in assignments]
        return bkt, payloads

    # Fallback Stage 3 input contract payloads
    return None, [
        {
            "student_id": "S001_Low", "co_id": "CO1", "mastery": 0.28,
            "bloom_level_range": ["Remember", "Understand"],
            "percentile": 0.20, "percentile_band": "low",
            "flag_mismatch": False,
            "suggested_verbs": ["explain", "summarize", "describe", "classify"]
        },
        {
            "student_id": "S002_Mid", "co_id": "CO1", "mastery": 0.55,
            "bloom_level_range": ["Apply", "Analyze"],
            "percentile": 0.55, "percentile_band": "mid",
            "flag_mismatch": False,
            "suggested_verbs": ["apply", "solve", "analyze", "compare"]
        },
        {
            "student_id": "S003_High", "co_id": "CO1", "mastery": 0.84,
            "bloom_level_range": ["Evaluate", "Create"],
            "percentile": 0.90, "percentile_band": "high",
            "flag_mismatch": False,
            "suggested_verbs": ["design", "evaluate", "formulate", "justify"]
        },
    ]


def main():
    gen_dir = find_adapter_dir("lora_generation_adapter")
    grade_dir = find_adapter_dir("lora_grading_adapter")
    print("1. Initializing Ergasia 4-bit LLM Engine + Dual LoRA Adapters...")
    print(f"   -> Generation Adapter: {gen_dir} (Exists: {os.path.exists(gen_dir)})")
    print(f"   -> Grading Adapter   : {grade_dir} (Exists: {os.path.exists(grade_dir)})")

    engine = ErgasiaLLMEngine(
        gen_adapter_dir=gen_dir,
        grade_adapter_dir=grade_dir,
    )

    bkt_model, stage3_payloads = build_demo_cohort()
    generator = QuestionGenerator(llm_engine=engine)
    grader = AssignmentGrader(llm_engine=engine, bkt_model=bkt_model)

    topic = "Process Synchronization, Semaphores, and Deadlock Prevention"
    cohort_assignments = {}

    print("\n2. [Stage 3] Generating Mastery-Calibrated Assignments...")
    for payload in stage3_payloads:
        sid = payload["student_id"]
        q_record = generator.generate_for_student(
            stage3_payload=payload,
            topic=topic,
            target_marks=10,
            mode="generation",
        )
        cohort_assignments[sid] = [q_record]
        print(
            f"  -> {sid} (Mastery: {payload['mastery']:.2f} | Target: {payload['bloom_level_range']}) "
            f"=> Generated Bloom: {q_record['question_data']['bloom_level']} | "
            f"Marks: {q_record['question_data']['total_marks']}"
        )
        print(f"     Question: {q_record['question_data']['question_text'][:140]}...")

    print("\n3. [Stage 4] Running Cohort Fairness Audit...")
    fairness_report = FairnessChecker.check_cohort_assignments(cohort_assignments)
    print(json.dumps(fairness_report, indent=2))

    print("\n4. [Stage 5 + Feedback Loop] Grading Submissions & Updating BKT Mastery...")
    sample_answer = (
        "A semaphore is an integer variable accessed through atomic wait(P) and signal(V) operations "
        "to enforce mutual exclusion. Deadlocks can be prevented by breaking the circular wait condition "
        "through a strict global ordering of resource acquisition."
    )

    for sid, q_list in cohort_assignments.items():
        grade_report = grader.grade_and_update_mastery(
            student_id=sid,
            question_record=q_list[0],
            student_answer=sample_answer,
            mode="grading",
        )
        loop = grade_report["bkt_feedback_loop"]
        print(
            f"  -> {sid}: Score = {grade_report['total_score']}/{grade_report['max_marks']} | "
            f"BKT Mastery: {loop['prior_mastery']:.3f} -> {loop['posterior_mastery']:.3f} "
            f"(Delta: {loop['mastery_delta']:+.3f})"
        )
        print(f"     Feedback: {grade_report['overall_feedback'][:140]}...")


if __name__ == "__main__":
    main()