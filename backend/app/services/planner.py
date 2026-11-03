def plan_questions(cos: list[str], topics: list[str], total: int) -> list[dict]:
    """Turn the teacher's setup into the question plan that is IDENTICAL for every student.
    Round-robin over COs and topics so each CO and each topic (up to 6) is covered."""
    cos = sorted(cos)
    topics = [t.strip() for t in topics if t.strip()] or ["General"]
    n = max(len(cos), min(6, len(topics)))
    plan = [{"co_id": cos[i % len(cos)], "topic": topics[i % len(topics)]} for i in range(n)]
    base = total // n
    for p in plan:
        p["marks"] = base
    plan[-1]["marks"] = total - base * (n - 1)
    return plan
