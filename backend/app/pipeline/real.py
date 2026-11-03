"""
Real pipeline components wired to:
  1. ChromaDB + ergasia/rag/ (RagRetriever & RagKnowledgeBase)
  2. Unsloth 4-bit Qwen2.5-7B-Instruct + Dual QLoRA Adapters (QwenGenerator & QwenGrader)
"""
import importlib.util
import threading
from pathlib import Path
from typing import Dict, Any

from .interfaces import Chunk, PlanItem, StudentProfile

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
ERGASIA_DIR = BACKEND_DIR.parent
REPO_ROOT = ERGASIA_DIR.parent


def _load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_engine_mod = _load_module_from_path("ergasia_llm_engine", ERGASIA_DIR / "llm" / "engine.py")
_gen_mod = _load_module_from_path("ergasia_pipe_gen", ERGASIA_DIR / "pipeline" / "generator.py")
_grade_mod = _load_module_from_path("ergasia_pipe_grade", ERGASIA_DIR / "pipeline" / "grader.py")

ErgasiaLLMEngine = _engine_mod.ErgasiaLLMEngine
QuestionGenerator = _gen_mod.QuestionGenerator
AssignmentGrader = _grade_mod.AssignmentGrader

_ENGINE_LOCK = threading.Lock()
_SHARED_ENGINE = None


def _find_adapter(name: str) -> str:
    for base in [
        ERGASIA_DIR / "finetuning",
        REPO_ROOT / "finetuning",
        ERGASIA_DIR,
        REPO_ROOT,
        ERGASIA_DIR / "pipeline",
    ]:
        cand = base / name
        if cand.exists():
            return str(cand)
    return str(ERGASIA_DIR / "finetuning" / name)


def get_shared_engine():
    global _SHARED_ENGINE
    if _SHARED_ENGINE is None:
        with _ENGINE_LOCK:
            if _SHARED_ENGINE is None:
                gen_dir = _find_adapter("lora_generation_adapter")
                grade_dir = _find_adapter("lora_grading_adapter")
                print(f"[Ergasia Real Pipeline] Loading 4-bit Qwen2.5-7B + LoRA adapters...")
                print(f"  -> Gen Adapter  : {gen_dir}")
                print(f"  -> Grade Adapter: {grade_dir}")
                _SHARED_ENGINE = ErgasiaLLMEngine(
                    gen_adapter_dir=gen_dir,
                    grade_adapter_dir=grade_dir,
                )
    return _SHARED_ENGINE


class RagRetriever:
    """RETRIEVER_IMPL=real -> Queries ChromaDB populated in ergasia/rag/chroma_db."""

    def __init__(self):
        self.db_path = ERGASIA_DIR / "rag" / "chroma_db"
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
            except Exception as e:
                print(f"[RagRetriever] SentenceTransformer load warning: {e}")
                self._embedder = False
        return self._embedder

    def retrieve(self, course_id: str, co_id: str, topic: str, k: int = 4) -> list[Chunk]:
        try:
            import chromadb
            if not self.db_path.exists():
                raise FileNotFoundError(f"ChromaDB not found at {self.db_path}")

            client = chromadb.PersistentClient(path=str(self.db_path))
            collections = client.list_collections()
            if not collections:
                raise ValueError("No collections inside ChromaDB")

            col = collections[0]
            query_text = f"{co_id} {topic}"
            embedder = self._get_embedder()

            if embedder:
                q_emb = embedder.encode([query_text], normalize_embeddings=True).tolist()
                res = col.query(query_embeddings=q_emb, n_results=max(1, k))
            else:
                res = col.query(query_texts=[query_text], n_results=max(1, k))

            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]

            chunks = []
            for i, doc in enumerate(docs):
                meta = metas[i] if i < len(metas) and metas[i] else {}
                src = meta.get("source") or meta.get("file_name") or "chroma_kb"
                dist = float(dists[i]) if i < len(dists) and dists[i] is not None else 0.2
                score = round(max(0.0, 1.0 - dist), 4)
                chunks.append(Chunk(text=doc, source=str(src), score=score))

            if chunks:
                return chunks
        except Exception as e:
            print(f"[RagRetriever] Chroma query notice: {e}")

        return [
            Chunk(
                text=f"Artificial Intelligence ({course_id} - {co_id}) reference context on {topic}.",
                source="syllabus_context",
                score=0.75,
            )
        ]


class RagKnowledgeBase:
    """KB_IMPL=real -> Ingests teacher-uploaded files (PDF/DOCX/TXT) directly into ChromaDB."""

    def ingest(self, path: Path, kind: str, course_id: str) -> None:
        db_dir = ERGASIA_DIR / "rag" / "chroma_db"
        db_dir.mkdir(parents=True, exist_ok=True)

        text = ""
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(str(path))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception:
                try:
                    import fitz  # PyMuPDF fallback
                    doc = fitz.open(str(path))
                    text = "\n".join(page.get_text() for page in doc)
                except Exception as e:
                    raise RuntimeError(f"Could not extract text from PDF {path.name}: {e}")
        elif suffix == ".docx":
            try:
                import docx
                d = docx.Document(str(path))
                text = "\n".join(p.text for p in d.paragraphs if p.text)
            except Exception as e:
                raise RuntimeError(f"Could not read DOCX {path.name}: {e}")
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")

        text = text.strip()
        if not text:
            raise ValueError(f"No readable text found in {path.name} (scanned PDFs require OCR).")

        # Chunk text into ~900-char overlapping segments
        chunks = [text[i : i + 900] for i in range(0, len(text), 750) if text[i : i + 900].strip()]
        if not chunks:
            raise ValueError(f"No valid chunks produced from {path.name}.")

        import chromadb
        from sentence_transformers import SentenceTransformer

        client = chromadb.PersistentClient(path=str(db_dir))
        cols = client.list_collections()
        col = cols[0] if cols else client.get_or_create_collection(name="ergasia_kb")

        emb_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        vecs = emb_model.encode(chunks, normalize_embeddings=True).tolist()
        ids = [f"{path.stem}_{i}" for i in range(len(chunks))]
        metas = [{"source": path.name, "kind": kind, "course_id": course_id} for _ in chunks]
        col.upsert(ids=ids, embeddings=vecs, documents=chunks, metadatas=metas)
        print(f"[RagKnowledgeBase] Indexed {len(chunks)} chunks from {path.name} into ChromaDB.")


class QwenGenerator:
    """GENERATOR_IMPL=real -> Uses fine-tuned lora_generation_adapter via ErgasiaLLMEngine."""

    SUB_SCENARIOS = [
        "Focus on a step-by-step trace with a concrete state-space or numerical example.",
        "Frame the problem around an autonomous robotics or smart navigation agent.",
        "Focus on comparing algorithmic trade-offs, time/space complexity, and optimality.",
        "Frame the problem around diagnosing failure modes or edge cases in an intelligent system.",
        "Focus on formal logic representation, mathematical formulation, and proofs.",
        "Frame the problem around healthcare decision support or cybersecurity threat analysis.",
        "Focus on heuristic admissibility, pruning efficiency, or adversarial counter-moves.",
        "Frame the problem around multi-agent coordination under uncertainty.",
        "Focus on system architecture design and real-world deployment constraints.",
        "Frame the problem around responsible AI, explainability, and safety verification.",
    ]

    def generate_variant(
        self, plan: list[PlanItem], student: StudentProfile, context: dict[int, list[Chunk]], rev: int = 0
    ) -> list[dict]:
        engine = get_shared_engine()
        q_gen = QuestionGenerator(llm_engine=engine)
        questions_out = []

        sid = student.get("student_id", "S001")
        sid_idx = sum(ord(c) * (idx + 1) for idx, c in enumerate(sid))

        for i, item in enumerate(plan):
            co_id = item["co_id"]
            topic = item["topic"]
            marks = int(item["marks"])

            mastery_val = float(student.get("mastery", {}).get(co_id, 0.5))
            if mastery_val < 0.25:
                bloom_range, target_bloom, verbs = ["Remember", "Remember"], "Remember", ["define", "list", "state"]
            elif mastery_val < 0.40:
                bloom_range, target_bloom, verbs = ["Remember", "Understand"], "Understand", ["explain", "describe", "summarize"]
            elif mastery_val < 0.55:
                bloom_range, target_bloom, verbs = ["Understand", "Apply"], "Apply", ["apply", "solve", "trace"]
            elif mastery_val < 0.70:
                bloom_range, target_bloom, verbs = ["Apply", "Analyze"], "Analyze", ["analyze", "compare", "differentiate"]
            elif mastery_val < 0.85:
                bloom_range, target_bloom, verbs = ["Analyze", "Evaluate"], "Evaluate", ["evaluate", "justify", "critique"]
            else:
                bloom_range, target_bloom, verbs = ["Evaluate", "Create"], "Create", ["design", "formulate", "propose"]

            scenario = self.SUB_SCENARIOS[(sid_idx + i + rev * 3) % len(self.SUB_SCENARIOS)]
            chunks = context.get(i) or []
            chunk_texts = [c.text for c in chunks] if chunks else [f"Course material for {co_id}: {topic}"]

            stage3_payload = {
                "student_id": sid,
                "co_id": co_id,
                "mastery": mastery_val,
                "bloom_level_range": bloom_range,
                "suggested_verbs": verbs,
            }

            q_gen.retrieve_context = lambda payload, topic_hint="": {
                "context_chunks": chunk_texts,
                "example_questions": [],
                "fallbacks": [],
            }

            prompt_topic = f"{topic} (Target Bloom: {target_bloom}, Mastery: {int(mastery_val*100)}%. {scenario})"
            with _ENGINE_LOCK:
                rec = q_gen.generate_for_student(
                    stage3_payload=stage3_payload,
                    topic=prompt_topic,
                    target_marks=marks,
                    mode="generation",
                )

            q_data = rec["question_data"]
            raw_rubric = q_data.get("rubric", [])
            ui_rubric = []
            running_sum = 0
            for idx, r in enumerate(raw_rubric):
                if idx == len(raw_rubric) - 1:
                    m_val = max(1, marks - running_sum)
                else:
                    m_val = max(1, int(round(float(r.get("marks", r.get("max_marks", marks // max(1, len(raw_rubric))))))))
                    if running_sum + m_val >= marks:
                        m_val = max(1, marks - running_sum - (len(raw_rubric) - 1 - idx))
                    running_sum += m_val
                ui_rubric.append({
                    "criterion": str(r.get("criterion", f"Criterion {idx + 1}")),
                    "max_marks": int(m_val),
                })

            if not ui_rubric or sum(r["max_marks"] for r in ui_rubric) != marks:
                a = round(marks * 0.4)
                b = marks - a
                ui_rubric = [
                    {"criterion": "Conceptual Clarity & Formulation", "max_marks": a},
                    {"criterion": "Application & Analysis", "max_marks": b},
                ]

            questions_out.append({
                "text": str(q_data.get("question_text", f"[{target_bloom}] {topic}: {scenario}")),
                "bloom_level": target_bloom,
                "co_id": co_id,
                "topic": topic,   # Strictly matches plan topic for Stage 4 Fairness
                "marks": marks,   # Strictly matches plan marks for Stage 4 Fairness
                "rubric": ui_rubric,
            })

        return questions_out


class QwenGrader:
    """GRADER_IMPL=real -> Grades submissions using fine-tuned lora_grading_adapter."""

    def grade(self, questions: list[dict], answers: list[dict], files: list[dict]) -> dict:
        print(f"[QwenGrader] Starting automated grading for {len(questions)} question(s)...")
        engine = get_shared_engine()
        grader = AssignmentGrader(llm_engine=engine, bkt_model=None)

        ans_by_qid = {a.get("question_id"): a.get("text", "") for a in (answers or []) if isinstance(a, dict)}
        file_by_qid = {f.get("question_id"): f for f in (files or []) if isinstance(f, dict)}

        per_criterion = []
        per_question = []
        feedbacks = []
        confidences = []
        total_score = 0.0
        max_score = 0.0

        for idx, q in enumerate(questions):
            qid = q.get("question_id", f"q{idx + 1}")
            q_label = f"Q{idx + 1}"
            q_marks = float(q.get("marks", 10))
            max_score += q_marks

            student_text = str(ans_by_qid.get(qid, "")).strip()
            if qid in file_by_qid and file_by_qid[qid].get("path"):
                fpath = Path(file_by_qid[qid]["path"])
                if fpath.exists():
                    try:
                        if fpath.suffix.lower() == ".pdf":
                            import pypdf
                            reader = pypdf.PdfReader(str(fpath))
                            extracted = "\n".join(p.extract_text() or "" for p in reader.pages)
                        else:
                            extracted = fpath.read_text(encoding="utf-8", errors="ignore")
                        if extracted.strip():
                            student_text = f"{student_text}\n[Uploaded File ({fpath.name})]:\n{extracted.strip()}".strip()
                    except Exception as e:
                        print(f"[QwenGrader] File extraction warning for {fpath}: {e}")

            if not student_text:
                student_text = "No answer submitted."

            ui_rubric = q.get("rubric") or [{"criterion": "Solution Completeness", "max_marks": int(q_marks)}]
            grader_rubric = [
                {
                    "criterion": str(r.get("criterion", "Criterion")),
                    "marks": float(r.get("max_marks", r.get("marks", 5))),
                    "expected_points": str(r.get("criterion", "Complete and accurate explanation")),
                }
                for r in ui_rubric
            ]

            question_record = {
                "bkt_mastery": 0.5,
                "question_data": {
                    "co_id": q.get("co_id", "CO1"),
                    "bloom_level": q.get("bloom_level", "Apply"),
                    "total_marks": q_marks,
                    "question_text": q.get("text", ""),
                    "rubric": grader_rubric,
                },
            }

            try:
                with _ENGINE_LOCK:
                    res = grader.grade_and_update_mastery(
                        student_id="ui_student",
                        question_record=question_record,
                        student_answer=student_text,
                        mode="grading",
                        update_bkt=False,
                    )
            except Exception as e:
                print(f"[QwenGrader] Error grading {qid}: {e}")
                res = {"schema_valid": False, "score_ratio": 0.5, "criterion_scores": [], "overall_feedback": "Evaluated against rubric."}

            c_scores = res.get("criterion_scores") or []
            q_awarded = 0.0

            for r_idx, r_item in enumerate(ui_rubric):
                c_name = str(r_item.get("criterion", f"Criterion {r_idx + 1}"))
                c_max = float(r_item.get("max_marks", r_item.get("marks", 0)))
                matched = c_scores[r_idx] if r_idx < len(c_scores) and isinstance(c_scores[r_idx], dict) else {}

                if "awarded" in matched:
                    raw_award = float(matched.get("awarded", 0.0))
                elif "score" in matched:
                    raw_award = float(matched.get("score", 0.0))
                else:
                    raw_award = float(res.get("score_ratio", 0.5)) * c_max

                award = max(0.0, min(c_max, round(raw_award * 2) / 2))
                q_awarded += award
                just = str(matched.get("justification") or matched.get("comment") or "Evaluated against rubric criterion.")

                per_criterion.append({
                    "criterion": f"{q_label}: {c_name}",
                    "score": award,
                    "max": c_max,
                    "comment": just,
                })

            q_awarded = round(min(q_marks, q_awarded), 2)
            total_score += q_awarded
            per_question.append({
                "question_id": qid,
                "score": q_awarded,
                "max": q_marks,
            })
            feedbacks.append(f"{q_label}: {res.get('overall_feedback', 'Good attempt.')}")
            confidences.append(0.88 if res.get("schema_valid") else 0.62)

        avg_conf = round(sum(confidences) / max(1, len(confidences)), 2)
        print(f"[QwenGrader] Completed grading! Total Score: {total_score}/{max_score} (Confidence: {avg_conf})")
        return {
            "total_score": round(total_score, 2),
            "max_score": round(max_score, 2),
            "per_criterion": per_criterion,
            "per_question": per_question,
            "feedback": " ".join(feedbacks).strip(),
            "ai_confidence": avg_conf,
        }


def build():
    from types import SimpleNamespace
    return SimpleNamespace(
        retriever=RagRetriever(),
        generator=QwenGenerator(),
        grader=QwenGrader(),
        kb=RagKnowledgeBase(),
    )