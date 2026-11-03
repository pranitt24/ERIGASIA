"""
level_mapping.py
-----------------
Maps a student's BKT mastery probability for a CO to a target Bloom's
Taxonomy level for question generation (Stage 3 input).

Approach (per project discussion):
  1. PRIMARY signal: absolute threshold buckets on mastery probability.
     Simple, transparent, defensible to faculty / NBA-OBE reviewers.
  2. SAFETY NET: percentile rank within the class for the same CO, logged
     alongside the bucket decision so we can sanity-check that bucket
     boundaries actually produce a spread across the class rather than
     everyone clustering in one bucket. Can be promoted to the primary
     signal later (or blended) if thresholds prove miscalibrated.

Output also carries which CO-PO linked Bloom's verbs to feed into the
generation prompt, so this module's output plugs directly into Stage 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


# Bloom's levels in increasing cognitive complexity, with a few representative
# question-verb cues each (used later to steer the generation prompt).
BLOOM_LEVELS = [
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create",
]

BLOOM_VERBS = {
    "Remember":   ["define", "list", "state", "recall"],
    "Understand": ["explain", "summarize", "describe", "classify"],
    "Apply":      ["implement", "solve", "use", "demonstrate"],
    "Analyze":    ["compare", "differentiate", "break down", "examine"],
    "Evaluate":   ["justify", "critique", "assess", "argue"],
    "Create":     ["design", "formulate", "construct", "propose"],
}

# Default absolute thresholds on BKT mastery -> a *pair* of Bloom's levels
# (low end, high end) so generation can pick a level within that band rather
# than a single rigid point. Tune these once you have real performance data.
DEFAULT_THRESHOLDS: List[Tuple[float, str, str]] = [
    # (upper_bound_exclusive, low_level, high_level)
    (0.40, "Remember", "Understand"),
    (0.70, "Apply", "Analyze"),
    (1.01, "Evaluate", "Create"),  # 1.01 so mastery == 1.0 still matches
]

# Percentile bands used only as a cross-check / class-spread sanity signal.
DEFAULT_PERCENTILE_BANDS: List[Tuple[float, str]] = [
    (0.30, "low"),
    (0.70, "mid"),
    (1.01, "high"),
]


@dataclass
class LevelAssignment:
    student_id: str
    co_id: str
    mastery: float
    bucket_low: str
    bucket_high: str
    percentile: float
    percentile_band: str
    agrees_with_percentile: bool
    suggested_verbs: List[str]

    def as_dict(self) -> dict:
        return {
            "student_id": self.student_id,
            "co_id": self.co_id,
            "mastery": round(self.mastery, 4),
            "bloom_level_range": [self.bucket_low, self.bucket_high],
            "percentile": round(self.percentile, 4),
            "percentile_band": self.percentile_band,
            "flag_mismatch": not self.agrees_with_percentile,
            "suggested_verbs": self.suggested_verbs,
        }


def _bucket_for_mastery(mastery: float,
                         thresholds: List[Tuple[float, str, str]]) -> Tuple[str, str]:
    for upper, low, high in thresholds:
        if mastery < upper:
            return low, high
    return thresholds[-1][1], thresholds[-1][2]


def _percentile_rank(value: float, population: List[float]) -> float:
    """Fraction of the population strictly below `value` (0..1). If the
    population has 1 or fewer students, returns 0.5 (neutral)."""
    if len(population) <= 1:
        return 0.5
    below = sum(1 for v in population if v < value)
    return below / (len(population) - 1) if len(population) > 1 else 0.5


def _band_for_percentile(pct: float,
                          bands: List[Tuple[float, str]]) -> str:
    for upper, name in bands:
        if pct < upper:
            return name
    return bands[-1][1]


# Rough ordinal position of each bucket-pair's *midpoint* on the same low/mid/high
# scale as the percentile bands, used only to flag disagreement for review.
_BUCKET_BAND_ORDER = {
    ("Remember", "Understand"): "low",
    ("Apply", "Analyze"): "mid",
    ("Evaluate", "Create"): "high",
}


def assign_level(
    student_id: str,
    co_id: str,
    mastery: float,
    class_mastery_values: List[float],
    thresholds: List[Tuple[float, str, str]] = None,
    percentile_bands: List[Tuple[float, str]] = None,
) -> LevelAssignment:
    """
    Core entry point: given one student's mastery for a CO and the whole
    class's mastery values for that same CO, return a LevelAssignment.

    `class_mastery_values` should include this student's own value.
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS
    percentile_bands = percentile_bands or DEFAULT_PERCENTILE_BANDS

    low, high = _bucket_for_mastery(mastery, thresholds)
    pct = _percentile_rank(mastery, class_mastery_values)
    band = _band_for_percentile(pct, percentile_bands)

    expected_band = _BUCKET_BAND_ORDER.get((low, high), "mid")
    agrees = (expected_band == band)

    verbs = BLOOM_VERBS[high]  # lean toward the higher end of the assigned range

    return LevelAssignment(
        student_id=student_id,
        co_id=co_id,
        mastery=mastery,
        bucket_low=low,
        bucket_high=high,
        percentile=pct,
        percentile_band=band,
        agrees_with_percentile=agrees,
        suggested_verbs=verbs,
    )


def assign_levels_for_class(
    co_id: str,
    mastery_by_student: Dict[str, float],
    thresholds: List[Tuple[float, str, str]] = None,
    percentile_bands: List[Tuple[float, str]] = None,
) -> Dict[str, LevelAssignment]:
    """Convenience: assign levels for every student in a class, for one CO,
    in a single pass (so percentile ranking is computed against the whole
    class at once)."""
    values = list(mastery_by_student.values())
    return {
        sid: assign_level(sid, co_id, m, values, thresholds, percentile_bands)
        for sid, m in mastery_by_student.items()
    }
