"""python -m app.seed [--reset] -> Seeds VIT PEIT21T Artificial Intelligence course, 10 students across all 6 Bloom tiers, and 1 AI draft assignment."""
import math
import re
import sys
from datetime import datetime, timedelta

from .auth import hash_password
from .course import CO_IDS, COURSE
from .db import Base, SessionLocal, engine
from .models import Assignment, KBItem, MasteryHistory, MasteryState, User
from .services.planner import plan_questions

PASSWORD = "ergasia"

# 10 students spanning all 6 Bloom taxonomy tiers across CO1..CO6:
# (<0.25 Remember | 0.25-0.40 Understand | 0.40-0.55 Apply | 0.55-0.70 Analyze | 0.70-0.85 Evaluate | >0.85 Create)
STUDENTS = [
    ("S001", "Ananya Rao",    [0.20, 0.34, 0.48, 0.62, 0.76, 0.88]),  # Spans all 6 levels across CO1-CO6
    ("S002", "Rohan Mehta",   [0.89, 0.78, 0.64, 0.50, 0.35, 0.21]),  # High on CO1/CO2 (Create/Evaluate), Low on CO5/CO6
    ("S003", "Isha Nair",     [0.33, 0.22, 0.37, 0.29, 0.44, 0.31]),  # Remember / Understand tier
    ("S004", "Kabir Shah",    [0.64, 0.68, 0.58, 0.66, 0.61, 0.67]),  # Analyze tier
    ("S005", "Diya Menon",    [0.92, 0.88, 0.90, 0.86, 0.91, 0.89]),  # Create tier across all COs
    ("S006", "Vihaan Patel",  [0.18, 0.19, 0.21, 0.17, 0.23, 0.20]),  # Remember tier across all COs
    ("S007", "Meera Joshi",   [0.46, 0.51, 0.44, 0.49, 0.52, 0.47]),  # Apply tier
    ("S008", "Arjun Desai",   [0.75, 0.81, 0.73, 0.79, 0.77, 0.82]),  # Evaluate tier
    ("S009", "Sara Khan",     [0.58, 0.31, 0.87, 0.43, 0.74, 0.19]),  # Mixed profile
    ("S010", "Neil D’Souza",  [0.36, 0.63, 0.24, 0.81, 0.49, 0.90]),  # Mixed profile
]


def email_for(name: str) -> str:
    return re.sub(r"[^a-z.]", "", name.lower().replace(" ", ".")) + "@college.edu"


def run(reset: bool = False) -> None:
    if reset:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    db = SessionLocal()
    if db.query(User).count():
        print("Database already has users. Use --reset to start over.")
        db.close()
        return

    pw = hash_password(PASSWORD)
    db.add(User(id="T001", name="Prof. Shashikant Mahajan", email="meera.iyer@college.edu", password_hash=pw, role="teacher"))
    now = datetime.utcnow()

    for sid, name, ms in STUDENTS:
        db.add(User(id=sid, name=name, email=email_for(name), password_hash=pw, role="student"))
        for i, co in enumerate(CO_IDS):
            m = ms[i % len(ms)]
            db.add(MasteryState(student_id=sid, co_id=co, p_known=m))
            for k in range(6):
                v = max(0.05, min(0.99, m - (5 - k) * 0.04 + math.sin(k * 1.7 + i * 2) * 0.015)) if k < 5 else m
                db.add(MasteryHistory(student_id=sid, co_id=co, p_known=round(v, 4), at=now - timedelta(weeks=5 - k)))

    cos = ["CO1", "CO2", "CO3"]
    topics = [
        "PEAS Representation & Intelligent Agents",
        "A* Heuristic Search & Minimax Alpha-Beta Pruning",
        "First-Order Logic, Unification & Resolution",
    ]
    db.add(
        Assignment(
            id="A1",
            num=1,
            course_id=COURSE["id"],
            title="Intelligent Agents, Heuristic Search & Logical Reasoning",
            due=(now + timedelta(days=14)).date().isoformat(),
            total=30,
            status="draft",
            cos=cos,
            topics=topics,
            qspec=plan_questions(cos, topics, 30),
        )
    )

    db.add(KBItem(id="kb-aap-ai", name="AAP_AI_SemV_INFT_PEIT21T.pdf", kind="Syllabus", size="1.4 MB", status="indexed", path="raw_data"))
    db.commit()
    db.close()
    print(f"Seeded PEIT21T Artificial Intelligence! Teacher: meera.iyer@college.edu / {PASSWORD} | Student: ananya.rao@college.edu / {PASSWORD}")


if __name__ == "__main__":
    run("--reset" in sys.argv)