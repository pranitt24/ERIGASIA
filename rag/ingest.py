"""
ingest.py
---------
Build (or extend) the vector database from a folder of course materials.

    python ingest.py --src ./materials --db ./vectordb --course DSA \
                     --co-file ./co_descriptions.json

Folder layout hint: put files under folders whose names say what they are,
e.g. materials/textbook/..., materials/question_bank/..., materials/assignments/...,
materials/syllabus/..., materials/notes/... . The source type is guessed from
the path (see tagging.SOURCE_TYPE_HINTS); use --source-type to force one type
for everything in --src.

Re-running is safe: chunk IDs are derived from content, so unchanged files
are overwritten with identical data rather than duplicated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from chunking import (DEFAULT_CHUNK_CHARS, DEFAULT_OVERLAP_CHARS, chunk_text,
                      split_questions)
from embedder import Embedder, make_embedder
from loaders import SUPPORTED_EXTENSIONS, load_document
from store import VectorStore
from tagging import (detect_bloom_level, guess_source_type, load_co_descriptions,
                     tag_co)

QUESTION_TYPES = {"question_bank", "assignment"}
EMBED_BATCH = 64


def _chunk_id(course: str, source_file: str, location: str, idx: int, text: str) -> str:
    raw = f"{course}|{source_file}|{location}|{idx}|{text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def ingest(src: Path, db: Path, course: str,
           co_file: Optional[Path] = None,
           embedder: Optional[Embedder] = None,
           force_source_type: Optional[str] = None,
           chunk_chars: int = DEFAULT_CHUNK_CHARS,
           overlap_chars: int = DEFAULT_OVERLAP_CHARS,
           min_co_similarity: float = 0.40,
           verbose: bool = True) -> dict:
    src, db = Path(src), Path(db)
    embedder = embedder or make_embedder("st")
    store = VectorStore(db, embedder)

    co_desc = load_co_descriptions(co_file)
    co_vecs: Dict[str, List[float]] = {}
    if co_desc:
        ids = list(co_desc)
        co_vecs = dict(zip(ids, embedder.embed_documents([co_desc[i] for i in ids])))
        # Saved next to the DB so retrieval can build CO-aware queries later.
        (db / "co_descriptions.json").write_text(
            json.dumps(co_desc, indent=2, ensure_ascii=False), encoding="utf-8")

    records: List[dict] = []
    empty_files, skipped_files, failed_files = [], [], []

    files = sorted(p for p in src.rglob("*") if p.is_file())
    for path in files:
        rel = str(path.relative_to(src))
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            skipped_files.append(rel)
            continue
        try:
            segments = load_document(path)
        except Exception as e:  # corrupt / password-protected file etc.
            failed_files.append((rel, f"{type(e).__name__}: {e}"))
            continue
        if not segments:
            empty_files.append(rel)
            continue

        source_type = force_source_type or guess_source_type(path.relative_to(src))
        is_question_type = source_type in QUESTION_TYPES
        n = 0
        for seg in segments:
            pieces = (split_questions(seg.text, chunk_chars) if is_question_type
                      else chunk_text(seg.text, chunk_chars, overlap_chars))
            for piece in pieces:
                records.append({
                    "id": _chunk_id(course, rel, seg.location, n, piece),
                    "text": piece,
                    "meta": {
                        "course": course,
                        "source_file": rel,
                        "source_type": source_type,
                        "location": seg.location,
                        "chunk_index": n,
                        "bloom_level": detect_bloom_level(piece) if is_question_type else "",
                    },
                })
                n += 1

    if not records:
        raise RuntimeError(f"No usable text found under {src}. "
                           f"Empty: {empty_files}, failed: {failed_files}")

    # Embed in batches, then tag each chunk with its best-matching CO.
    vectors: List[List[float]] = []
    texts = [r["text"] for r in records]
    for i in range(0, len(texts), EMBED_BATCH):
        vectors.extend(embedder.embed_documents(texts[i:i + EMBED_BATCH]))
    for rec, vec in zip(records, vectors):
        rec["meta"]["co_id"] = (tag_co(vec, co_vecs, min_co_similarity)
                                if co_vecs else "UNTAGGED")

    store.upsert([r["id"] for r in records], texts, vectors,
                 [r["meta"] for r in records])

    summary = {
        "files_ingested": len({r["meta"]["source_file"] for r in records}),
        "chunks": len(records),
        "by_source_type": dict(Counter(r["meta"]["source_type"] for r in records)),
        "by_co": dict(Counter(r["meta"]["co_id"] for r in records)),
        "by_bloom": dict(Counter(r["meta"]["bloom_level"] or "(none)" for r in records)),
        "empty_files": empty_files,
        "unsupported_skipped": skipped_files,
        "failed_files": failed_files,
        "db_total_chunks": store.count(),
    }
    if verbose:
        print_summary(summary, bool(co_vecs))
    return summary


def print_summary(s: dict, had_co_file: bool) -> None:
    print("\n=== Ingestion summary ===")
    print(f"files ingested : {s['files_ingested']}")
    print(f"chunks added   : {s['chunks']}   (database now holds {s['db_total_chunks']})")
    print(f"by source type : {s['by_source_type']}")
    print(f"by CO          : {s['by_co']}")
    print(f"by Bloom's     : {s['by_bloom']}  (question banks/assignments only)")
    if s["empty_files"]:
        print(f"\nNO TEXT FOUND in {len(s['empty_files'])} file(s) (scanned PDFs need OCR):")
        for f in s["empty_files"]:
            print(f"  - {f}")
    if s["failed_files"]:
        print(f"\nFAILED to read {len(s['failed_files'])} file(s):")
        for f, err in s["failed_files"]:
            print(f"  - {f}: {err}")
    if s["unsupported_skipped"]:
        print(f"\nSkipped {len(s['unsupported_skipped'])} unsupported file(s), e.g. "
              f"{s['unsupported_skipped'][:5]}")
    if not had_co_file:
        print("\nNOTE: no --co-file given, so every chunk is UNTAGGED. CO-aware retrieval "
              "needs a CO description file.")
    elif s["by_co"].get("UNTAGGED", 0) > 0.5 * s["chunks"]:
        print("\nWARNING: over half the chunks are UNTAGGED. Lower --min-co-sim or make "
              "the CO descriptions more specific, then re-ingest.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest course materials into the vector DB")
    ap.add_argument("--src", required=True, type=Path, help="folder of materials")
    ap.add_argument("--db", required=True, type=Path, help="vector DB folder (created if missing)")
    ap.add_argument("--course", required=True, help="course code/name, e.g. DSA")
    ap.add_argument("--co-file", type=Path, help='JSON {"CO1": "description", ...}')
    ap.add_argument("--source-type", help="force one source type for all files")
    ap.add_argument("--embedder", choices=["st", "hash"], default="st")
    ap.add_argument("--model", help="sentence-transformers model name")
    ap.add_argument("--device", help="cuda / cpu (default: auto)")
    ap.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    ap.add_argument("--overlap-chars", type=int, default=DEFAULT_OVERLAP_CHARS)
    ap.add_argument("--min-co-sim", type=float, default=0.40)
    ap.add_argument("--reset", action="store_true", help="delete the DB folder first")
    args = ap.parse_args()

    if args.reset and args.db.exists():
        shutil.rmtree(args.db)
    ingest(args.src, args.db, args.course, args.co_file,
           make_embedder(args.embedder, args.model, args.device),
           args.source_type, args.chunk_chars, args.overlap_chars, args.min_co_sim)


if __name__ == "__main__":
    main()
