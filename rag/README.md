# RAG pipeline (Stage 1 knowledge base + retrieval for Stage 3)

Turns the team's course materials into a searchable, **outcome-aware** vector
database, and retrieves grounding material for each student's assignment.

Scripts import each other directly, so **run everything from inside this folder**
(`cd ergasia/rag`). The retriever also imports `../mastery/` automatically.

## Files

| File | Purpose |
|---|---|
| `loaders.py` | PDF / DOCX / PPTX / TXT / MD to text segments (keeps page or slide numbers) |
| `chunking.py` | Overlapping paragraph chunker, plus a one-question-per-chunk splitter for question banks |
| `tagging.py` | `source_type` (from folder names), `bloom_level` (rule-based, from the opening verb), `co_id` (nearest CO description) |
| `embedder.py` | Swappable embedders: real (`bge-small-en-v1.5`) and an offline test stand-in |
| `store.py` | ChromaDB wrapper. Refuses to query a DB with a different embedder than it was built with |
| `ingest.py` | CLI: folder of materials to vector DB, with a diagnostic summary |
| `retrieve.py` | `Retriever.retrieve_for_student()` takes the mastery module's `LevelAssignment.as_dict()` and returns context + example questions |
| `test_rag.py` | End-to-end test on synthetic files (offline) |

## Setup (on the 24 GB desktop)

```bash
pip install -r requirements.txt        # install a CUDA build of PyTorch first
python test_rag.py                     # should end with ALL CHECKS PASSED
```

## 1. Prepare your materials folder

Folder names tell the pipeline what each file is. Use any of these words in the path:

```
materials/
  syllabus/        syllabus.pdf
  textbook/        ch1.pdf  ch2.pdf
  notes/           lecture1.pptx
  question_bank/   previous_year_papers.pdf  qbank.docx
  assignments/     assignment1.docx
```

Question banks and assignments should have **numbered questions** (`1.`, `2)`, `Q3.`).
Each question becomes one chunk, which keeps Bloom's tagging accurate.

## 2. Write the CO description file

Copy `co_descriptions.example.json` and replace it with your real Course Outcomes.
**This file drives CO tagging**, so write each outcome in specific, topic-rich words
(copy from the syllabus). Vague outcomes produce vague tags.

## 3. Ingest

```bash
python ingest.py --src ./materials --db ./vectordb --course DSA \
                 --co-file ./co_descriptions.json --device cuda
```

Re-running is safe (no duplicates). Use `--reset` to rebuild from scratch.
**Read the summary it prints**: it shows chunks per source type, per CO, per Bloom's
level, and lists files with no text (scanned PDFs need OCR), unreadable files and skipped types.

## 4. Retrieve for a student

```python
from embedder import make_embedder
from retrieve import Retriever, format_prompt_context
from level_mapping import assign_level          # ../mastery

r = Retriever("./vectordb", make_embedder("st", device="cuda"), course="DSA")
a = assign_level("S001", "CO1", mastery=0.2, class_mastery_values=[0.2, 0.5, 0.9]).as_dict()
result = r.retrieve_for_student(a)
prompt_block = format_prompt_context(result)   # goes into the generation prompt
print(result["fallbacks"])                     # empty list = retrieval was healthy
```

## What was tested, and what was NOT

Tested here (offline, synthetic files): all four file loaders, chunking, question
splitting, Bloom's tagging rules, CO tagging mechanics, idempotent re-ingestion,
embedder-mismatch protection, metadata filtering (CO / source type / Bloom's level),
fallback reporting, and the mastery to retrieval contract.

**Not tested, so verify on your side:**
- **The real embedder** (`sentence-transformers`). The offline test used a word-matching
  stand-in, so *retrieval quality is unmeasured*. After ingesting real materials, run 10-20
  sample queries per CO and check the top results make sense.
- **CO tagging accuracy** on real content. The default `--min-co-sim 0.40` is a starting
  guess. If the summary shows many UNTAGGED chunks, lower it; if tags look wrong, raise it
  or sharpen the CO descriptions. A teacher should spot-check ~20 tagged chunks.
- **Bloom's tagging** is keyword-based, so unusual phrasings ("With the help of an
  example, ...") get no level. Extend the verb lists in `tagging.py` for your institution's wording.
- **Real PDFs** vary a lot (two-column layouts, headers/footers repeated on every page,
  tables). Check a few chunks from your textbook PDFs for garbled text.

## Known limitations / things to watch

- **Duplicate content** across sources (slides copied from the textbook) can fill the
  top-k with near-identical chunks. If you see this, add de-duplication or MMR re-ranking in `retrieve.py`.
- **Example questions can leak.** Past questions are retrieved as *style references*; the
  generation prompt must forbid copying them (the prompt block already says so, but check the outputs).
- **Fairness:** retrieval is filtered by CO, so students share topics. Any fallback
  (listed in `result["fallbacks"]`) means a student got looser material. Log these and
  feed them to the Stage 4 fairness check.
- Scanned PDFs have no text layer and need OCR before ingestion.
