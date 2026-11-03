"""
test_rag.py
-----------
End-to-end test of the RAG pipeline on small synthetic course materials.
Needs no downloads: uses the offline HashingEmbedder, which matches shared
words rather than meaning. So this proves the PLUMBING (loading, chunking,
tagging, filtering, persistence, the mastery->retrieval contract) but says
nothing about retrieval QUALITY. Judge quality only with the real embedder
on your real materials.

Run:  python test_rag.py
"""

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "mastery"))

from chunking import chunk_text, split_questions            # noqa: E402
from embedder import HashingEmbedder                         # noqa: E402
from ingest import ingest                                    # noqa: E402
from level_mapping import assign_level                       # noqa: E402
from loaders import load_document                            # noqa: E402
from retrieve import Retriever, format_prompt_context        # noqa: E402
from store import VectorStore                                # noqa: E402
from tagging import detect_bloom_level, guess_source_type    # noqa: E402

CO_DESCRIPTIONS = {
    "CO1": "Explain stack and queue data structures and their operations",
    "CO2": "Implement sorting algorithms such as quicksort and mergesort",
    "CO3": "Analyze graph traversal algorithms like BFS and DFS",
}

STACK_TEXT = (
    "A stack is a linear data structure that follows last in first out order. "
    "The stack operations are push, pop and peek. A queue follows first in first out "
    "order and supports enqueue and dequeue operations.\n\n"
    "Stacks are used for function calls, undo features and expression evaluation. "
    "Queues are used in scheduling and breadth first search."
)
SORT_TEXT = (
    "Quicksort picks a pivot and partitions the array into smaller and larger elements. "
    "Mergesort splits the array in halves, sorts each half and merges the sorted halves. "
    "Sorting algorithms differ in time complexity and stability.\n\n"
    "Quicksort has average time complexity n log n while mergesort guarantees n log n."
)
GRAPH_TEXT = (
    "Breadth first search explores a graph level by level using a queue. "
    "Depth first search explores a graph as deep as possible using a stack or recursion. "
    "Graph traversal algorithms are used for connectivity and shortest paths."
)
QBANK_TEXT = """1. Define a stack and list its operations.
2. Explain how a queue differs from a stack with an example.
3. Implement a stack using an array and demonstrate push and pop.
4. Compare quicksort and mergesort for time complexity and stability.
5. Justify the choice of a queue over a stack for breadth first search.
6. Design a data structure that supports stack operations and minimum retrieval in constant time.
7. Implement quicksort and trace it on the array 5 2 9 1 7.
"""


def make_materials(root: Path) -> None:
    (root / "textbook").mkdir(parents=True)
    (root / "question_bank").mkdir()
    (root / "assignments").mkdir()
    (root / "notes").mkdir()
    (root / "empty_scan").mkdir()
    (root / "misc").mkdir()

    (root / "textbook" / "ch1_stacks_queues.txt").write_text(STACK_TEXT)

    # DOCX textbook chapter
    import docx
    d = docx.Document()
    for para in SORT_TEXT.split("\n\n"):
        d.add_paragraph(para)
    d.save(root / "textbook" / "ch2_sorting.docx")

    # PPTX lecture notes
    from pptx import Presentation
    prs = Presentation()
    for title, body in [("Graph traversal", GRAPH_TEXT), ("Stacks", STACK_TEXT)]:
        s = prs.slides.add_slide(prs.slide_layouts[1])
        s.shapes.title.text = title
        s.placeholders[1].text = body
    prs.save(root / "notes" / "lecture_graphs.pptx")

    # PDF question bank (real PDF, text extractable)
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(root / "question_bank" / "qbank.pdf"), pagesize=A4)
    y = 800
    for line in QBANK_TEXT.splitlines():
        c.drawString(40, y, line[:110])
        y -= 18
    c.save()

    # Assignment as plain text
    (root / "assignments" / "assignment1.txt").write_text(
        "1. Explain the working of a queue using an example.\n"
        "2. Apply depth first search to the graph given in class and write the visit order.\n")

    # A file with no extractable text, an unsupported type, and a corrupt file
    (root / "empty_scan" / "scanned.txt").write_text("   \n")
    (root / "misc" / "diagram.png").write_bytes(b"\x89PNG\r\n")
    (root / "misc" / "broken.pdf").write_bytes(b"this is not a pdf")


def check(cond: bool, msg: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        check.failed += 1


check.failed = 0


def main() -> int:
    # ---- unit-level checks -------------------------------------------- #
    chunks = chunk_text("Sentence one is here. " * 200, chunk_chars=300, overlap_chars=50)
    check(all(len(c) <= 360 for c in chunks) and len(chunks) > 5,
          f"chunker respects size limit ({len(chunks)} chunks)")
    check(len(split_questions(QBANK_TEXT)) == 7, "question bank splits into 7 questions")
    check(detect_bloom_level("3. Implement a stack using an array") == "Apply",
          "Bloom's: 'Implement' -> Apply")
    check(detect_bloom_level("5. Justify the choice of a queue") == "Evaluate",
          "Bloom's: 'Justify' -> Evaluate")
    check(detect_bloom_level("1. Define a stack") == "Remember",
          "Bloom's: 'Define' -> Remember")
    check(detect_bloom_level("2. Apply depth first search to the graph") == "Apply",
          "Bloom's: 'Apply' -> Apply (regression: verb was missing)")
    check(detect_bloom_level("The algorithm uses a stack to define order") == "",
          "Bloom's: verb mid-sentence is ignored")
    check(guess_source_type(Path("question_bank/x.pdf")) == "question_bank",
          "source type guessed from folder name")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mats, db = tmp / "materials", tmp / "vectordb"
        make_materials(mats)

        # all four loaders return text
        for rel in ["textbook/ch1_stacks_queues.txt", "textbook/ch2_sorting.docx",
                    "notes/lecture_graphs.pptx", "question_bank/qbank.pdf"]:
            segs = load_document(mats / rel)
            check(len(segs) > 0 and all(s.text for s in segs), f"loader reads {rel}")
        check(load_document(mats / "notes/lecture_graphs.pptx")[1].location == "slide 2",
              "pptx keeps slide numbers")

        co_file = tmp / "co.json"
        co_file.write_text(json.dumps(CO_DESCRIPTIONS))
        emb = HashingEmbedder()

        # ---- ingest ---------------------------------------------------- #
        s = ingest(mats, db, "DSA", co_file, emb, min_co_similarity=0.05, verbose=True)
        # 5 files have text; the 6th (empty scan) is reported but not ingested.
        check(s["files_ingested"] == 5, f"5 files ingested (got {s['files_ingested']})")
        check(s["by_bloom"].get("(none)", 0) == s["chunks"] - 9,
              "every question/assignment chunk got a Bloom's level "
              f"(untagged = {s['by_bloom'].get('(none)', 0)}, expected {s['chunks'] - 9} non-question chunks)")
        check("empty_scan/scanned.txt" in s["empty_files"], "empty file reported")
        check(any("broken.pdf" in f for f, _ in s["failed_files"]), "corrupt file reported, not fatal")
        check("misc/diagram.png" in s["unsupported_skipped"], "unsupported file skipped")
        check(s["by_source_type"].get("question_bank") == 7, "7 question-bank chunks")
        check(s["by_co"].get("UNTAGGED", 0) < s["chunks"], "some chunks received a CO tag")

        # ---- idempotent re-ingest -------------------------------------- #
        before = VectorStore(db, emb).count()
        ingest(mats, db, "DSA", co_file, emb, min_co_similarity=0.05, verbose=False)
        after = VectorStore(db, emb).count()
        check(before == after, f"re-ingest adds no duplicates ({before} -> {after})")

        # ---- embedder mismatch protection ------------------------------ #
        try:
            VectorStore(db, HashingEmbedder(dim=256))
            check(False, "mismatched embedder is rejected")
        except ValueError:
            check(True, "mismatched embedder is rejected")

        # ---- retrieval from a mastery payload -------------------------- #
        r = Retriever(db, emb, course="DSA")
        low_mastery = assign_level("S001", "CO1", 0.20, [0.2, 0.5, 0.9]).as_dict()
        res = r.retrieve_for_student(low_mastery, k_context=3, k_examples=2)
        check(res["bloom_level_range"] == ["Remember", "Understand"], "contract fields carried through")
        check(len(res["context"]) == 3, f"3 context chunks (got {len(res['context'])})")
        check(all(c["source_type"] in ("textbook", "notes", "syllabus", "other")
                  for c in res["context"]), "context excludes question banks")
        check(all(e["source_type"] in ("question_bank", "assignment")
                  for e in res["examples"]), "examples come only from question sources")
        check(all(e["bloom_level"] in ("Remember", "Understand") for e in res["examples"]),
              "examples match the low-mastery Bloom's range: "
              f"{[e['bloom_level'] for e in res['examples']]}")

        high_mastery = assign_level("S002", "CO1", 0.95, [0.2, 0.5, 0.95]).as_dict()
        res_hi = r.retrieve_for_student(high_mastery, k_context=3, k_examples=2)
        check(all(e["bloom_level"] in ("Evaluate", "Create") for e in res_hi["examples"]),
              "high-mastery examples are Evaluate/Create: "
              f"{[e['bloom_level'] for e in res_hi['examples']]}")
        # Fairness-relevant: both students got material for the SAME CO.
        check(all(c["co_id"] == "CO1" for c in res["context"] + res_hi["context"]
                  if c["co_id"] != "UNTAGGED") or bool(res["fallbacks"] or res_hi["fallbacks"]),
              "same CO for both students (or a fallback was reported)")

        # fallback path: a CO with no example questions at that level
        odd = assign_level("S003", "CO3", 0.50, [0.2, 0.5, 0.9]).as_dict()
        res_odd = r.retrieve_for_student(odd, k_context=2, k_examples=2)
        print("\nfallbacks for CO3 case:", res_odd["fallbacks"])
        check(isinstance(res_odd["fallbacks"], list), "fallbacks are reported as a list")

        # empty database is handled
        empty = VectorStore(tmp / "emptydb", emb)
        check(empty.query("anything") == [], "empty DB returns no hits, no crash")

        print("\n--- sample prompt block (low-mastery student) ---")
        print(format_prompt_context(res)[:900])

    print(f"\n{'ALL CHECKS PASSED' if not check.failed else str(check.failed) + ' CHECK(S) FAILED'}")
    return 1 if check.failed else 0


if __name__ == "__main__":
    sys.exit(main())
