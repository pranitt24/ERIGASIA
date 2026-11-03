from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String, primary_key=True)  # S001 / T001
    name: Mapped[str] = mapped_column(String)
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String)
    role: Mapped[str] = mapped_column(String)  # student | teacher


class Assignment(Base):
    __tablename__ = "assignments"
    id: Mapped[str] = mapped_column(String, primary_key=True)  # A1, A2 ...
    num: Mapped[int] = mapped_column(Integer, index=True)
    course_id: Mapped[str] = mapped_column(String, default="CS201")
    title: Mapped[str] = mapped_column(String)
    due: Mapped[str] = mapped_column(String)  # ISO date
    total: Mapped[int] = mapped_column(Integer)
    # draft -> generating -> in_review -> released -> closed
    status: Mapped[str] = mapped_column(String, default="draft")
    cos: Mapped[list] = mapped_column(JSON)
    topics: Mapped[list] = mapped_column(JSON)
    qspec: Mapped[list] = mapped_column(JSON)  # [{co_id, topic, marks}], identical for every student
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Variant(Base):
    """One personalised assignment for one student."""
    __tablename__ = "variants"
    id: Mapped[str] = mapped_column(String, primary_key=True)  # "A1-S001"
    assignment_id: Mapped[str] = mapped_column(String, index=True)
    student_id: Mapped[str] = mapped_column(String, index=True)
    rev: Mapped[int] = mapped_column(Integer, default=0)
    approved: Mapped[bool] = mapped_column(Boolean, default=False)
    questions: Mapped[list] = mapped_column(JSON)


class Submission(Base):
    __tablename__ = "submissions"
    id: Mapped[str] = mapped_column(String, primary_key=True)  # "sub-A1-S001"
    variant_id: Mapped[str] = mapped_column(String, index=True)
    assignment_id: Mapped[str] = mapped_column(String, index=True)
    student_id: Mapped[str] = mapped_column(String, index=True)
    # submitted -> ai_graded -> teacher_confirmed   (or regrade_requested)
    status: Mapped[str] = mapped_column(String, default="submitted")
    submitted_at: Mapped[str] = mapped_column(String)
    answers: Mapped[list] = mapped_column(JSON)
    files: Mapped[list] = mapped_column(JSON, default=list)
    regrade_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    grade: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class MasteryState(Base):
    """Only used by the built-in fallback BKT. Your real mastery/ module keeps its own state."""
    __tablename__ = "mastery_state"
    student_id: Mapped[str] = mapped_column(String, primary_key=True)
    co_id: Mapped[str] = mapped_column(String, primary_key=True)
    p_known: Mapped[float] = mapped_column(Float)


class MasteryHistory(Base):
    __tablename__ = "mastery_history"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[str] = mapped_column(String, index=True)
    co_id: Mapped[str] = mapped_column(String)
    p_known: Mapped[float] = mapped_column(Float)
    at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class KBFile(Base):
    __tablename__ = "kb_files"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    kind: Mapped[str] = mapped_column(String)
    size: Mapped[str] = mapped_column(String)
    path: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="indexing")  # indexing | indexed | failed
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
