import os
import sys
import json
import math
from typing import List, Dict, Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ERGASIA_PKG_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(ERGASIA_PKG_DIR, ".."))

for p in [REPO_ROOT, ERGASIA_PKG_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ergasia.llm.engine import ErgasiaLLMEngine
from ergasia.pipeline.generator import GENERATION_SYSTEM_PROMPT
from ergasia.pipeline.grader import GRADING_SYSTEM_PROMPT
from ergasia.pipeline.demo_e2e import find_adapter_dir


def compute_qwk(y_true: List[int], y_pred: List[int], min_rating: int = 0, max_rating: int = 10) -> float:
    """Computes Quadratic Weighted Kappa (QWK) in pure Python (no sklearn dependency required)."""
    num_ratings = max_rating - min_rating + 1
    conf_mat = [[0] * num_ratings for _ in range(num_ratings)]
    n = len(y_true)
    if n == 0:
        return 0.0

    for t, p in zip(y_true, y_pred):
        t_idx = max(0, min(num_ratings - 1, int(round(t)) - min_rating))
        p_idx = max(0, min(num_ratings - 1, int(round(p)) - min_rating))
        conf_mat[t_idx][p_idx] += 1

    hist_true = [sum(conf_mat[i][j] for j in range(num_ratings)) for i in range(num_ratings)]
    hist_pred = [sum(conf_mat[i][j] for i in range(num_ratings)) for j in range(num_ratings)]

    numerator = 0.0
    denominator = 0.0
    for i in range(num_ratings):
        for j in range(num_ratings):
            expected = (hist_true[i] * hist_pred[j]) / float(n)
            weight = ((i - j) ** 2) / float((num_ratings - 1) ** 2)
            numerator += weight * conf_mat[i][j]
            denominator += weight * expected

    if denominator == 0.0:
        return 1.0
    return 1.0 - (numerator / denominator)


# Evaluation Benchmark Cases (Generation + Grading)
GENERATION_TEST_CASES = [
    {"co_id": "CO1", "mastery": 0.20, "bloom_range": ["Remember", "Understand"], "verbs": ["define", "explain"], "topic": "Process Control Blocks and Context Switching", "marks": 10},
    {"co_id": "CO1", "mastery": 0.35, "bloom_range": ["Remember", "Understand"], "verbs": ["summarize", "classify"], "topic": "Types of Operating System Kernels (Monolithic vs Microkernel)", "marks": 10},
    {"co_id": "CO2", "mastery": 0.52, "bloom_range": ["Apply", "Analyze"], "verbs": ["apply", "calculate"], "topic": "CPU Scheduling Algorithms (SRTF and Round Robin Turnaround Time)", "marks": 10},
    {"co_id": "CO2", "mastery": 0.65, "bloom_range": ["Apply", "Analyze"], "verbs": ["analyze", "compare"], "topic": "Banker's Algorithm for Deadlock Avoidance", "marks": 10},
    {"co_id": "CO3", "mastery": 0.82, "bloom_range": ["Evaluate", "Create"], "verbs": ["evaluate", "justify"], "topic": "Page Replacement Policies under Thrashing Workloads", "marks": 10},
    {"co_id": "CO3", "mastery": 0.91, "bloom_range": ["Evaluate", "Create"], "verbs": ["design", "formulate"], "topic": "Lock-Free Concurrent Data Structures vs Semaphore Synchronization", "marks": 10},
]

GRADING_TEST_CASES = [
    {
        "co_id": "CO1", "bloom_level": "Understand", "max_marks": 10,
        "question": "Explain the four necessary conditions for a deadlock to occur and how breaking circular wait prevents deadlocks.",
        "rubric": [
            {"criterion": "Four Necessary Conditions", "marks": 6, "expected_points": "Mutual exclusion, hold and wait, no preemption, circular wait."},
            {"criterion": "Circular Wait Prevention", "marks": 4, "expected_points": "Impose a total ordering of resource types and require requesting in increasing order."}
        ],
        "student_answer": "The four Coffman conditions are mutual exclusion, hold and wait, no preemption, and circular wait. We prevent circular wait by assigning a strict numerical order to all resources and forcing processes to request resources in increasing order.",
        "human_gold_score": 10
    },
    {
        "co_id": "CO1", "bloom_level": "Understand", "max_marks": 10,
        "question": "Explain the four necessary conditions for a deadlock to occur and how breaking circular wait prevents deadlocks.",
        "rubric": [
            {"criterion": "Four Necessary Conditions", "marks": 6, "expected_points": "Mutual exclusion, hold and wait, no preemption, circular wait."},
            {"criterion": "Circular Wait Prevention", "marks": 4, "expected_points": "Impose a total ordering of resource types and require requesting in increasing order."}
        ],
        "student_answer": "Deadlock happens when processes wait for each other in a circle (circular wait) and hold resources (hold and wait). You can stop it by restarting the operating system.",
        "human_gold_score": 3
    },
    {
        "co_id": "CO2", "bloom_level": "Apply", "max_marks": 10,
        "question": "Compare Preemptive Shortest Job First (SRTF) and Round Robin scheduling in terms of response time and starvation.",
        "rubric": [
            {"criterion": "Response Time Analysis", "marks": 5, "expected_points": "RR guarantees bounded response time via time quantum; SRTF minimizes average waiting time."},
            {"criterion": "Starvation Analysis", "marks": 5, "expected_points": "SRTF can starve long processes if short jobs keep arriving; RR is starvation-free."}
        ],
        "student_answer": "Round Robin uses a fixed time quantum so every job gets CPU time quickly, giving great response time and zero starvation. However, SRTF always picks the shortest remaining burst, which gives optimal average wait time but can starve long processes indefinitely.",
        "human_gold_score": 9
    },
    {
        "co_id": "CO2", "bloom_level": "Apply", "max_marks": 10,
        "question": "Compare Preemptive Shortest Job First (SRTF) and Round Robin scheduling in terms of response time and starvation.",
        "rubric": [
            {"criterion": "Response Time Analysis", "marks": 5, "expected_points": "RR guarantees bounded response time via time quantum; SRTF minimizes average waiting time."},
            {"criterion": "Starvation Analysis", "marks": 5, "expected_points": "SRTF can starve long processes if short jobs keep arriving; RR is starvation-free."}
        ],
        "student_answer": "Round robin schedules processes in a queue. SRTF runs the shortest job first. SRTF can cause starvation for big processes, while Round Robin does not starve processes.",
        "human_gold_score": 6
    },
    {
        "co_id": "CO3", "bloom_level": "Evaluate", "max_marks": 10,
        "question": "Evaluate why LRU page replacement approximates optimal replacement better than FIFO and state when Belady's anomaly occurs.",
        "rubric": [
            {"criterion": "LRU vs FIFO Locality Evaluation", "marks": 6, "expected_points": "LRU exploits temporal locality; FIFO ignores usage frequency."},
            {"criterion": "Belady's Anomaly Explanation", "marks": 4, "expected_points": "Increasing page frames increases page faults in FIFO; LRU is a stack algorithm and immune."}
        ],
        "student_answer": "FIFO just swaps out the oldest page in memory even if it is heavily used. I am not sure what Belady's anomaly is.",
        "human_gold_score": 2
    },
]


def evaluate_generation(engine: ErgasiaLLMEngine, mode: str) -> Dict[str, float]:
    valid_schema_count = 0
    bloom_hit_count = 0
    rubric_sum_match_count = 0
    n = len(GENERATION_TEST_CASES)

    for case in GENERATION_TEST_CASES:
        target_bloom = case["bloom_range"][-1]
        user_prompt = (
            f"Target Course Outcome: {case['co_id']}\n"
            f"Topic: {case['topic']}\n"
            f"Student BKT Mastery: {case['mastery']:.2f}\n"
            f"Target Bloom's Taxonomy Level: {target_bloom} (Allowed range: {' - '.join(case['bloom_range'])})\n"
            f"Recommended Action Verbs: {', '.join(case['verbs'])}\n"
            f"Total Marks: {case['marks']}\n\n"
            f"Retrieved Grounding Context:\nCourse notes on {case['topic']} for {case['co_id']}.\n\n"
            "Generate the calibrated engineering question, CO-PO mapping, and grading rubric in strict JSON."
        )
        res = engine.generate_json(GENERATION_SYSTEM_PROMPT, user_prompt, mode=mode)
        data = res.get("data")
        if res["valid_json"] and isinstance(data, dict):
            req = {"question_text", "co_id", "bloom_level", "total_marks", "rubric"}
            if req.issubset(data.keys()) and isinstance(data["rubric"], list) and len(data["rubric"]) > 0:
                valid_schema_count += 1
                gen_bloom = str(data.get("bloom_level", "")).lower()
                if any(b.lower() in gen_bloom for b in case["bloom_range"]):
                    bloom_hit_count += 1
                r_sum = sum(float(r.get("marks", 0)) for r in data["rubric"] if isinstance(r, dict))
                if abs(r_sum - case["marks"]) <= 0.1:
                    rubric_sum_match_count += 1

    return {
        "schema_validity_rate": round(valid_schema_count / n, 4),
        "bloom_calibration_acc": round(bloom_hit_count / n, 4),
        "rubric_sum_acc": round(rubric_sum_match_count / n, 4),
    }


def evaluate_grading(engine: ErgasiaLLMEngine, mode: str) -> Dict[str, float]:
    gold_scores = []
    pred_scores = []
    valid_schema_count = 0
    n = len(GRADING_TEST_CASES)

    for case in GRADING_TEST_CASES:
        user_prompt = (
            f"Course Outcome: {case['co_id']} (Bloom's Level: {case['bloom_level']})\n"
            f"Question ({case['max_marks']} Marks):\n{case['question']}\n\n"
            f"Grading Rubric:\n{json.dumps(case['rubric'], indent=2)}\n\n"
            f"Student Submission:\n{case['student_answer']}\n\n"
            "Evaluate the submission against each rubric criterion and return the JSON grade report."
        )
        res = engine.generate_json(GRADING_SYSTEM_PROMPT, user_prompt, mode=mode)
        data = res.get("data")
        gold_scores.append(case["human_gold_score"])

        if res["valid_json"] and isinstance(data, dict) and ("total_score" in data or "criterion_scores" in data):
            valid_schema_count += 1
            c_scores = data.get("criterion_scores", [])
            if isinstance(c_scores, list) and len(c_scores) > 0:
                score = sum(float(c.get("awarded", 0)) for c in c_scores if isinstance(c, dict))
            else:
                score = float(data.get("total_score", 0))
            score = max(0.0, min(float(case["max_marks"]), score))
            pred_scores.append(round(score))
        else:
            pred_scores.append(0)

    qwk = compute_qwk(gold_scores, pred_scores, min_rating=0, max_rating=10)
    mae = sum(abs(g - p) for g, p in zip(gold_scores, pred_scores)) / float(n)
    return {
        "grading_schema_validity": round(valid_schema_count / n, 4),
        "grading_qwk": round(qwk, 4),
        "grading_mae": round(mae, 4),
        "gold_scores": gold_scores,
        "pred_scores": pred_scores,
    }


def main():
    gen_dir = find_adapter_dir("lora_generation_adapter")
    grade_dir = find_adapter_dir("lora_grading_adapter")
    engine = ErgasiaLLMEngine(gen_adapter_dir=gen_dir, grade_adapter_dir=grade_dir)

    print("\nRunning Ablation 1/2: Prompted RAG Baseline (Base Qwen2.5-7B-Instruct, Adapters Disabled)...")
    base_gen = evaluate_generation(engine, mode="base")
    base_grade = evaluate_grading(engine, mode="base")

    print("\nRunning Ablation 2/2: Fine-Tuned RAG (Ergasia QLoRA Adapters Enabled)...")
    ft_gen = evaluate_generation(engine, mode="generation")
    ft_grade = evaluate_grading(engine, mode="grading")

    report = {
        "Prompted_RAG_Baseline": {**base_gen, **base_grade},
        "FineTuned_RAG_Ergasia": {**ft_gen, **ft_grade},
    }

    out_path = os.path.join(CURRENT_DIR, "ablation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 75)
    print(f"{'Metric':<35} | {'Prompted RAG (Base)':<18} | {'Fine-Tuned RAG (LoRA)':<18}")
    print("-" * 75)
    print(f"{'Gen Schema Validity Rate':<35} | {base_gen['schema_validity_rate']:<18.2%} | {ft_gen['schema_validity_rate']:<18.2%}")
    print(f"{'Bloom Calibration Accuracy':<35} | {base_gen['bloom_calibration_acc']:<18.2%} | {ft_gen['bloom_calibration_acc']:<18.2%}")
    print(f"{'Rubric Sum Consistency':<35} | {base_gen['rubric_sum_acc']:<18.2%} | {ft_gen['rubric_sum_acc']:<18.2%}")
    print(f"{'Grading Schema Validity':<35} | {base_grade['grading_schema_validity']:<18.2%} | {ft_grade['grading_schema_validity']:<18.2%}")
    print(f"{'Grading QWK (Target >= 0.70)':<35} | {base_grade['grading_qwk']:<18.4f} | {ft_grade['grading_qwk']:<18.4f}")
    print(f"{'Grading Mean Absolute Error (MAE)':<35} | {base_grade['grading_mae']:<18.2f} | {ft_grade['grading_mae']:<18.2f}")
    print("=" * 75)
    print(f"Saved full ablation results to: {out_path}")


if __name__ == "__main__":
    main()