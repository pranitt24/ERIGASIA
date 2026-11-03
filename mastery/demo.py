"""
demo.py
-------
End-to-end smoke test for the mastery modeling stage, using synthetic data.
No GPU, no API key, no dataset needed — run this anytime to sanity-check
the BKT + level-assignment logic as it evolves.

Run:
    python demo.py
"""

import random
from bkt import BKTModel, BKTParams, fit_co_params
from level_mapping import assign_levels_for_class


random.seed(42)

COS = ["CO1", "CO2", "CO3"]
N_STUDENTS = 24
N_OPPORTUNITIES = 8  # questions seen per student per CO, in their history


def make_synthetic_responses(true_skill: float, n: int) -> list:
    """Simulate a response sequence for a student with a latent true_skill
    in [0,1]: higher true_skill -> more likely correct, with some noise
    (slip/guess) baked in, and a mild learning trend over opportunities."""
    responses = []
    skill = true_skill
    for i in range(n):
        p_correct = min(max(skill * 0.85 + 0.08, 0.05), 0.95)  # slip/guess-ish noise
        responses.append(random.random() < p_correct)
        skill = min(skill + random.uniform(0.0, 0.03), 1.0)  # slight learning drift
    return responses


def main():
    # 1. Synthesize students with varied latent ability per CO
    student_ids = [f"S{idx:03d}" for idx in range(N_STUDENTS)]
    latent_skill = {
        sid: {co: random.betavariate(2, 2) for co in COS}  # spread across [0,1]
        for sid in student_ids
    }

    raw_sequences = {
        co: {
            sid: make_synthetic_responses(latent_skill[sid][co], N_OPPORTUNITIES)
            for sid in student_ids
        }
        for co in COS
    }

    # 2. Fit BKT parameters per CO from the synthetic class data
    model = BKTModel()
    print("=== Fitting BKT parameters per CO (grid search) ===")
    for co in COS:
        sequences = list(raw_sequences[co].values())
        fitted = fit_co_params(sequences, grid_steps=5)
        model.set_co_params(co, fitted)
        print(f"{co}: p_init={fitted.p_init:.2f} p_transit={fitted.p_transit:.2f} "
              f"p_slip={fitted.p_slip:.2f} p_guess={fitted.p_guess:.2f}")

    # 3. Replay each student's response history through BKT to get mastery
    for co in COS:
        for sid in student_ids:
            model.update_sequence(sid, co, raw_sequences[co][sid])

    # 4. Cold-start check: a brand-new student with zero observations should
    #    fall back to the CO's prior (p_init), not error out.
    cold_start_mastery = model.get_mastery("BRAND_NEW_STUDENT", "CO1")
    print(f"\nCold-start mastery for a new student on CO1 (should == p_init): "
          f"{cold_start_mastery:.3f}")

    # 5. Level assignment per CO, across the whole class
    print("\n=== Level assignment (threshold + percentile cross-check) ===")
    mismatch_count = 0
    for co in COS:
        mastery_by_student = {sid: model.get_mastery(sid, co) for sid in student_ids}
        assignments = assign_levels_for_class(co, mastery_by_student)

        print(f"\n--- {co} ---")
        print(f"{'student':8} {'mastery':8} {'bloom range':22} {'pct':6} {'band':5} {'flag'}")
        band_counts = {"low": 0, "mid": 0, "high": 0}
        for sid in student_ids:
            a = assignments[sid]
            band_counts[a.percentile_band] += 1
            flag = "  <-- MISMATCH" if not a.agrees_with_percentile else ""
            if flag:
                mismatch_count += 1
            print(f"{sid:8} {a.mastery:8.3f} "
                  f"{a.bucket_low + '-' + a.bucket_high:22} "
                  f"{a.percentile:6.2f} {a.percentile_band:5}{flag}")

        print(f"Class spread across bands: {band_counts}")

    print(f"\nTotal threshold/percentile mismatches flagged for review: {mismatch_count} "
          f"/ {N_STUDENTS * len(COS)}")
    print("(A high mismatch count would mean the absolute thresholds need "
          "retuning against real data — that's exactly what this check is for.)")

    # 6. Example of what Stage 3 (generation) would actually consume
    example_sid, example_co = student_ids[0], COS[0]
    assignment = assign_levels_for_class(
        example_co, {sid: model.get_mastery(sid, example_co) for sid in student_ids}
    )[example_sid]
    print(f"\n=== Example Stage-3 input payload for {example_sid} / {example_co} ===")
    print(assignment.as_dict())


if __name__ == "__main__":
    main()
