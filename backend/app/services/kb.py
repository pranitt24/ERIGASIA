import re
import uuid
from pathlib import Path

from ..config import settings
from ..db import SessionLocal
from ..models import KBFile
from ..pipeline.registry import get_pipeline


def guess_kind(name: str) -> str:
    if re.search(r"syllab", name, re.I): return "Syllabus"
    if re.search(r"bank|quest", name, re.I): return "Question bank"
    return "Course material"


def human_size(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n > 1e6 else f"{max(1, round(n / 1e3))} KB"


def save_upload(db, filename: str, data: bytes, course_id: str = "CS201") -> KBFile:
    kid = f"k{uuid.uuid4().hex[:8]}"
    folder = settings.UPLOAD_DIR / "kb"
    folder.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w.\-]", "_", filename)
    path = folder / f"{kid}_{safe}"
    path.write_bytes(data)
    row = KBFile(id=kid, name=filename, kind=guess_kind(filename), size=human_size(len(data)), path=str(path), status="indexing")
    db.add(row); db.commit()
    return row


def run_ingest(kb_id: str, course_id: str = "CS201") -> None:
    db = SessionLocal()
    try:
        row = db.get(KBFile, kb_id)
        get_pipeline().kb.ingest(Path(row.path), row.kind, course_id)
        row.status = "indexed"
    except Exception as e:
        row = db.get(KBFile, kb_id)
        row.status = "failed"
        print(f"[kb] ingest failed for {kb_id}: {e}")
    finally:
        db.commit(); db.close()
