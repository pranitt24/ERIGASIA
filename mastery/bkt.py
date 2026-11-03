"""
bkt.py
------
Bayesian Knowledge Tracing (BKT) for Ergasia's Student Mastery Modeling stage.

Maintains a per-student, per-Course-Outcome (CO) mastery probability P(L_t),
updated after each observed response (correct / incorrect) on a question
tagged with that CO.

Standard BKT has four parameters per skill (here: per CO):
    p_init    : P(L_0)        - prior probability the student already knows the skill
    p_transit : P(T)          - probability of learning the skill between opportunities
    p_slip    : P(S)          - probability of an incorrect answer despite knowing the skill
    p_guess   : P(G)          - probability of a correct answer despite not knowing the skill

Update equations (standard BKT, Corbett & Anderson 1994):

    Posterior given evidence:
        if correct:
            P(L_t | evidence) = P(L_{t-1})(1 - p_slip)
                                 / [P(L_{t-1})(1 - p_slip) + (1 - P(L_{t-1})) * p_guess]
        if incorrect:
            P(L_t | evidence) = P(L_{t-1}) * p_slip
                                 / [P(L_{t-1}) * p_slip + (1 - P(L_{t-1})) * (1 - p_guess)]

    Then apply the learning transition:
        P(L_t) = P(L_t | evidence) + (1 - P(L_t | evidence)) * p_transit

No external dependencies beyond the standard library, so this runs anywhere
(including the bare laptop, no GPU needed).
"""

from __future__ import annotations

import json
import itertools
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional


DEFAULT_PARAMS = {
    "p_init": 0.30,
    "p_transit": 0.15,
    "p_slip": 0.10,
    "p_guess": 0.20,
}


@dataclass
class BKTParams:
    p_init: float = DEFAULT_PARAMS["p_init"]
    p_transit: float = DEFAULT_PARAMS["p_transit"]
    p_slip: float = DEFAULT_PARAMS["p_slip"]
    p_guess: float = DEFAULT_PARAMS["p_guess"]

    def validate(self) -> None:
        for name, val in asdict(self).items():
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"BKT parameter {name}={val} out of [0,1] range")


@dataclass
class COState:
    """Running BKT state for one (student, CO) pair."""
    co_id: str
    mastery: float
    n_observations: int = 0
    history: List[Tuple[int, float]] = field(default_factory=list)
    # history: list of (correct:0/1, mastery_after_update)


class BKTModel:
    """
    Tracks BKT mastery per CO, shared across students, and per-student state.

    Usage:
        model = BKTModel()
        model.set_co_params("CO1", BKTParams(p_init=0.25, p_transit=0.12,
                                              p_slip=0.10, p_guess=0.20))
        model.update("student_42", "CO1", correct=True)
        mastery = model.get_mastery("student_42", "CO1")
    """

    def __init__(self, default_params: Optional[BKTParams] = None):
        self.default_params = default_params or BKTParams()
        self.default_params.validate()

        # co_id -> BKTParams  (per-skill parameters, fit or set externally)
        self._co_params: Dict[str, BKTParams] = {}

        # (student_id, co_id) -> COState
        self._state: Dict[Tuple[str, str], COState] = {}

    # ------------------------------------------------------------------ #
    # Parameter management
    # ------------------------------------------------------------------ #

    def set_co_params(self, co_id: str, params: BKTParams) -> None:
        params.validate()
        self._co_params[co_id] = params

    def get_co_params(self, co_id: str) -> BKTParams:
        return self._co_params.get(co_id, self.default_params)

    # ------------------------------------------------------------------ #
    # Core update
    # ------------------------------------------------------------------ #

    def _get_or_init_state(self, student_id: str, co_id: str) -> COState:
        key = (student_id, co_id)
        if key not in self._state:
            p = self.get_co_params(co_id)
            self._state[key] = COState(co_id=co_id, mastery=p.p_init)
        return self._state[key]

    def update(self, student_id: str, co_id: str, correct: bool) -> float:
        """
        Observe one response (correct/incorrect) from a student on a question
        tagged with co_id, update that student's mastery for the CO, and
        return the new mastery probability.
        """
        params = self.get_co_params(co_id)
        state = self._get_or_init_state(student_id, co_id)
        p_prev = state.mastery

        if correct:
            numerator = p_prev * (1 - params.p_slip)
            denominator = numerator + (1 - p_prev) * params.p_guess
        else:
            numerator = p_prev * params.p_slip
            denominator = numerator + (1 - p_prev) * (1 - params.p_guess)

        # Guard against division by zero in degenerate parameter settings
        p_evidence = numerator / denominator if denominator > 1e-12 else p_prev

        p_new = p_evidence + (1 - p_evidence) * params.p_transit
        p_new = min(max(p_new, 0.0), 1.0)

        state.mastery = p_new
        state.n_observations += 1
        state.history.append((1 if correct else 0, p_new))

        return p_new

    def update_sequence(self, student_id: str, co_id: str,
                         responses: List[bool]) -> float:
        """Convenience: replay a sequence of responses, return final mastery."""
        for r in responses:
            self.update(student_id, co_id, r)
        return self.get_mastery(student_id, co_id)

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    def get_mastery(self, student_id: str, co_id: str) -> float:
        """
        Current mastery estimate. If the student has no observations yet for
        this CO, falls back to the CO's prior (p_init) — this is the
        'class-average cold start' behaviour for new students, since p_init
        can itself be fit from the whole class's historical data.
        """
        key = (student_id, co_id)
        if key in self._state:
            return self._state[key].mastery
        return self.get_co_params(co_id).p_init

    def get_mastery_vector(self, student_id: str, co_ids: List[str]) -> Dict[str, float]:
        """Mastery across a list of COs — this is the per-student mastery vector
        that feeds Stage 3 (question generation) and the level-assignment logic."""
        return {co: self.get_mastery(student_id, co) for co in co_ids}

    def get_class_mastery(self, co_id: str, student_ids: List[str]) -> Dict[str, float]:
        """All students' mastery for one CO — used for percentile ranking."""
        return {s: self.get_mastery(s, co_id) for s in student_ids}

    def n_observations(self, student_id: str, co_id: str) -> int:
        key = (student_id, co_id)
        return self._state[key].n_observations if key in self._state else 0

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "default_params": asdict(self.default_params),
            "co_params": {co: asdict(p) for co, p in self._co_params.items()},
            "state": {
                f"{s}||{co}": {
                    "co_id": st.co_id,
                    "mastery": st.mastery,
                    "n_observations": st.n_observations,
                    "history": st.history,
                }
                for (s, co), st in self._state.items()
            },
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "BKTModel":
        with open(path) as f:
            data = json.load(f)
        model = cls(default_params=BKTParams(**data["default_params"]))
        for co, p in data["co_params"].items():
            model.set_co_params(co, BKTParams(**p))
        for key, st in data["state"].items():
            s, co = key.split("||")
            model._state[(s, co)] = COState(
                co_id=st["co_id"],
                mastery=st["mastery"],
                n_observations=st["n_observations"],
                history=[tuple(h) for h in st["history"]],
            )
        return model


# ---------------------------------------------------------------------- #
# Parameter fitting (grid search over log-likelihood)
# ---------------------------------------------------------------------- #

def _sequence_log_likelihood(responses: List[bool], params: BKTParams) -> float:
    """Log-likelihood of one response sequence under a fixed BKT parameter set,
    marginalising mastery step by step (no hidden state is observed directly)."""
    import math
    p_L = params.p_init
    ll = 0.0
    for r in responses:
        p_correct = p_L * (1 - params.p_slip) + (1 - p_L) * params.p_guess
        p_correct = min(max(p_correct, 1e-9), 1 - 1e-9)
        ll += math.log(p_correct) if r else math.log(1 - p_correct)

        if r:
            num = p_L * (1 - params.p_slip)
            den = num + (1 - p_L) * params.p_guess
        else:
            num = p_L * params.p_slip
            den = num + (1 - p_L) * (1 - params.p_guess)
        p_ev = num / den if den > 1e-12 else p_L
        p_L = p_ev + (1 - p_ev) * params.p_transit
    return ll


def fit_co_params(sequences: List[List[bool]],
                   grid_steps: int = 5) -> BKTParams:
    """
    Fit BKT parameters for one CO via brute-force grid search maximising total
    log-likelihood across all students' response sequences for that CO.

    This is intentionally simple (grid search, not EM/Baum-Welch) so it has
    zero extra dependencies and is easy to explain/defend in a project report.
    Swap in a proper EM fitter later if you want tighter parameter estimates.

    `grid_steps` controls grid resolution per parameter (5 -> 5^4 = 625
    combinations; fine for a few hundred sequences, runs in well under a
    second with no GPU).
    """
    grid = [i / (grid_steps - 1) * 0.9 + 0.05 for i in range(grid_steps)]  # avoid 0/1 edges

    best_params = None
    best_ll = float("-inf")

    for p_init, p_transit, p_slip, p_guess in itertools.product(grid, repeat=4):
        # Constraint that keeps BKT identifiable: guessing shouldn't beat
        # knowing-with-a-slip (a common sanity constraint in BKT fitting).
        if p_slip >= 0.5 or p_guess >= 0.5:
            continue
        params = BKTParams(p_init, p_transit, p_slip, p_guess)
        total_ll = sum(_sequence_log_likelihood(seq, params) for seq in sequences if seq)
        if total_ll > best_ll:
            best_ll = total_ll
            best_params = params

    return best_params or BKTParams()
