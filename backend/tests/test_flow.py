"""End-to-end smoke test on the mock pipeline:  pytest -q   (from backend/)"""
import json
import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["UPLOAD_DIR"] = f"{_tmp}/uploads"
os.environ["MOCK_STEP_DELAY"] = "0"
os.environ["PIPELINE"] = "mock"

from fastapi.testclient import TestClient  # noqa: E402

from app import seed  # noqa: E402
from app.main import app  # noqa: E402


def login(c, email, role):
    r = c.post("/api/auth/login", json={"email": email, "password": "ergasia", "role": role})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["token"]}


def test_full_flow():
    seed.run(reset=True)
    c = TestClient(app)
    t = login(c, "meera.iyer@college.edu", "teacher")

    assert c.post("/api/auth/login", json={"email": "meera.iyer@college.edu", "password": "x"}).status_code == 401
    assert c.post("/api/auth/login", json={"email": "meera.iyer@college.edu", "password": "ergasia", "role": "student"}).status_code == 403

    r = c.post("/api/teacher/assignments", headers=t, json={"title": "Test", "cos": ["CO1", "CO2"], "topics": ["Stacks", "Big-O"], "total": 40, "due": "2026-12-01"})
    aid = r.json()["id"]
    jid = c.post(f"/api/teacher/assignments/{aid}/generate", headers=t).json()["job_id"]
    assert c.get(f"/api/jobs/{jid}", headers=t).json()["status"] == "done"

    v = c.get(f"/api/teacher/assignments/{aid}/variants", headers=t).json()
    assert len(v["variants"]) == 10 and v["fairness"]["same_total_marks"] and v["fairness"]["same_topics"]
    assert c.post(f"/api/teacher/assignments/{aid}/release", headers=t).status_code == 409  # not approved yet

    # breaking marks must trip the fairness check; regenerating must repair it
    vid = v["variants"][0]["variant_id"]
    broke = c.patch(f"/api/teacher/variants/{vid}", headers=t, json={"question_id": "q1", "marks": 5}).json()
    assert not broke["fairness"]["same_total_marks"]
    fixed = c.post(f"/api/teacher/variants/{vid}/regenerate", headers=t).json()
    assert fixed["fairness"]["same_total_marks"]

    for x in v["variants"]:
        c.patch(f"/api/teacher/variants/{x['variant_id']}", headers=t, json={"approved": True})
    s = login(c, "ananya.rao@college.edu", "student")
    assert all(a["assignment_id"] != aid for a in c.get("/api/student/assignments", headers=s).json())  # hidden until release
    assert c.post(f"/api/teacher/assignments/{aid}/release", headers=t).json()["status"] == "released"

    mine = [a for a in c.get("/api/student/assignments", headers=s).json() if a["assignment_id"] == aid][0]
    var = c.get(f"/api/student/assignments/{mine['variant_id']}", headers=s).json()
    answers = [{"question_id": q["question_id"], "text": "A reasonably detailed answer " * 10} for q in var["variant"]["questions"]]
    r = c.post("/api/student/submissions", headers=s, data={"payload": json.dumps({"variant_id": mine["variant_id"], "answers": answers, "files": []})})
    assert r.status_code == 200, r.text
    sub = c.get(f"/api/student/submissions/{r.json()['submission_id']}", headers=s).json()
    assert sub["grade"] and sub["submission"]["status"] == "ai_graded"

    q = c.get("/api/teacher/grading-queue", headers=t).json()
    assert any(i["submission"]["id"] == sub["submission"]["id"] for i in q)
    n = len(sub["grade"]["per_criterion"])
    g = c.patch(f"/api/teacher/grades/{sub['submission']['id']}", headers=t, json={"scores": [0] * n, "confirm": True}).json()
    assert g["status"] == "teacher_confirmed" and g["teacher_override"]["total"] == 0

    p = c.get("/api/student/progress", headers=s).json()
    assert len(p["cos"]) == 3 and len(p["cos"][0]["trend"]) == 6
    assert c.get("/api/teacher/analytics/co-po", headers=t).status_code == 200
    assert c.get("/api/student/assignments", headers=t).status_code == 403
