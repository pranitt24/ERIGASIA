# Mastery Modeling (Stage 2)

Zero external dependencies — pure Python standard library. Runs identically
on the laptop, the desktop, or any cloud instance.

## Files

- **`bkt.py`** — Bayesian Knowledge Tracing core. Per-(student, CO) mastery
  state, standard BKT update equations, a brute-force grid-search parameter
  fitter (`fit_co_params`), and JSON save/load for persisting state between
  sessions.
- **`level_mapping.py`** — Maps a mastery probability to a target Bloom's
  Taxonomy range using absolute threshold buckets (primary signal, tunable
  via `DEFAULT_THRESHOLDS`), cross-checked against the student's percentile
  rank within the class (safety net, flags disagreement for review). Also
  returns suggested Bloom's verbs to steer Stage 3 prompt generation.
- **`demo.py`** — End-to-end smoke test with synthetic student response data:
  fits BKT params per CO, replays response histories, checks cold-start
  behaviour for a brand-new student, assigns levels across a synthetic
  class, and prints the exact payload Stage 3 (generation) would consume.

## Run it

```bash
python demo.py
```

## Design notes / what to revisit once real data arrives

1. **Parameter fitting is grid search, not EM/Baum-Welch.** Simple, zero
   dependencies, easy to defend in a report — but coarse. Swap in a proper
   EM fitter (e.g. via `pyBKT` or a hand-rolled Baum-Welch) once you have
   enough real response sequences per CO to make the extra precision worth
   it. `grid_steps` in `fit_co_params` controls resolution if you want finer
   grids in the meantime.

2. **Threshold boundaries (`DEFAULT_THRESHOLDS` in `level_mapping.py`) are
   placeholders** (0.40 / 0.70 splitting Remember-Understand / Apply-Analyze
   / Evaluate-Create). The synthetic demo run shows a real failure mode:
   with noisy fitted params, ~30% of assignments disagreed with the
   percentile cross-check. Re-tune these thresholds once real student
   performance data is available, using the mismatch rate as your
   calibration signal — if mismatches stay high, move toward the percentile-
   based assignment (option 4 from the design discussion) as primary instead
   of the absolute thresholds.

3. **Cold start** currently falls back to the CO's fitted `p_init` (i.e. the
   class-average prior) for any student with zero observations — matches
   the "class-average cold start" behaviour from the proposal. Confirmed
   working in `demo.py`.

4. **Next integration point:** `assign_levels_for_class()`'s output
   (`LevelAssignment.as_dict()`) is the contract Stage 3's RAG/generation
   module should consume — `bloom_level_range` + `suggested_verbs` feed the
   generation prompt, `co_id` selects which knowledge-base chunks to
   retrieve.
