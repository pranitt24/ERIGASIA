BLOOM = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def rubric_for(bloom: str, marks: int) -> list[dict]:
    """Default 3-criterion rubric (30/40/30). The generator may return its own rubric instead."""
    bi = BLOOM.index(bloom) if bloom in BLOOM else 2
    names = (["Correct concept", "Explanation", "Clarity"] if bi < 2
             else ["Correct method", "Working shown", "Final result"] if bi == 2
             else ["Valid analysis", "Evidence", "Justification"])
    a, b = round(marks * 0.3), round(marks * 0.4)
    return [{"criterion": n, "max_marks": m} for n, m in zip(names, [a, b, marks - a - b])]
