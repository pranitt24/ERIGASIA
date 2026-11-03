"""Stand-ins so the whole app works end to end before the real pipeline exists."""
import hashlib
import random
import time
from pathlib import Path

from ..config import settings
from .common import BLOOM, clamp, rubric_for
from .interfaces import Chunk

LEVEL_TEXT = [
    "State the key definitions and properties of {t}.",
    "Explain how {t} works, using a short example.",
    "Apply {t} to a small problem of your choice and show every step.",
    "Analyze the trade-offs of {t} by comparing at least two approaches.",
    "Evaluate when {t} is the right choice, and defend your position.",
]
SUFFIX = ["", " Use a small worked example.", " Keep your answer under 150 words.", " Refer to a real system you know."]
FEEDBACK = {
    "hi": "Strong work. Your reasoning is clear and well structured. To push higher, tie each conclusion back to the trade-off the question asks about.",
    "mid": "You have the core idea. Your working skips a step, so show every step to secure the method marks.",
    "lo": "The main concept is partly there, but the explanation is thin. Revisit the lecture notes on this topic and try the practice set before the next assignment.",
}


def _seed(*parts: str) -> random.Random:
    return random.Random(int(hashlib.sha256("|".join(parts).encode()).hexdigest(), 16) % (2**32))


class MockRetriever:
    def retrieve(self, course_id, co_id, topic, k=4):
        return [Chunk(text=f"(mock context about {topic})", source="mock", score=1.0)]


class MockGenerator:
    def generate_variant(self, plan, student, context, rev=0):
        out = []
        for item in plan:
            hi = student["levels"][item["co_id"]][1]
            idx = min(BLOOM.index(hi), 4)
            out.append({
                "text": LEVEL_TEXT[idx].format(t=item["topic"]) + SUFFIX[rev % len(SUFFIX)],
                "bloom_level": hi, "co_id": item["co_id"], "topic": item["topic"], "marks": item["marks"],
            })
        return out


class MockGrader:
    def grade(self, questions, answers, files):
        per_crit, per_q, confs = [], [], []
        for i, q in enumerate(questions):
            ans = next((a.get("text", "") for a in answers if a["question_id"] == q["question_id"]), "")
            has_file = any(f["question_id"] == q["question_id"] for f in files)
            r = _seed(q["text"], ans)
            quality = clamp(0.25 + min(len(ans.split()), 80) / 130 + (0.25 if has_file else 0) + (r.random() - 0.5) * 0.2)
            q_total = 0.0
            for c in q["rubric"]:
                sc = clamp(round(c["max_marks"] * clamp(quality + (r.random() - 0.5) * 0.3) * 2) / 2, 0, c["max_marks"])
                q_total += sc
                per_crit.append({"criterion": f"Q{i + 1}: {c['criterion']}", "score": sc, "max": c["max_marks"],
                                 "comment": "Fully addressed." if sc >= c["max_marks"] * 0.8
                                 else "Partly there; one step is missing." if sc >= c["max_marks"] * 0.5
                                 else "Needs more detail here."})
            per_q.append({"question_id": q["question_id"], "score": q_total, "max": q["marks"]})
            confs.append(clamp(0.35 + quality * 0.6 + (r.random() - 0.5) * 0.1))
        total = sum(p["score"] for p in per_q)
        mx = sum(p["max"] for p in per_q) or 1
        frac = total / mx
        return {"total_score": total, "max_score": mx, "per_criterion": per_crit, "per_question": per_q,
                "feedback": FEEDBACK["hi" if frac > 0.75 else "mid" if frac > 0.5 else "lo"],
                "ai_confidence": round(sum(confs) / len(confs), 2) if confs else 0.5}


class MockKB:
    def ingest(self, path: Path, kind: str, course_id: str) -> None:
        time.sleep(settings.MOCK_STEP_DELAY * 2)


def build():
    from types import SimpleNamespace
    return SimpleNamespace(retriever=MockRetriever(), generator=MockGenerator(), grader=MockGrader(), kb=MockKB())
