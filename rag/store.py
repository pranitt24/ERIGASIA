"""
store.py
--------
Thin wrapper over a persistent ChromaDB collection.

Why a wrapper: the rest of the project should not care that Chroma is the
backend (FAISS or pgvector could replace it), and the wrapper adds one
safety check that matters: a database built with one embedding model must
never be queried with another, or every similarity score is meaningless.
That check is done with a small `embedder.txt` file next to the database.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from embedder import Embedder

DEFAULT_COLLECTION = "ergasia_kb"
_UPSERT_BATCH = 512


def build_where(**conditions) -> Optional[dict]:
    """build_where(co_id="CO1", source_type=["textbook","notes"]) -> Chroma filter.
    A list value means "any of"; None values are ignored."""
    clauses = []
    for key, value in conditions.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            clauses.append({key: {"$in": list(value)}})
        else:
            clauses.append({key: value})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class VectorStore:
    def __init__(self, db_path, embedder: Embedder,
                 collection: str = DEFAULT_COLLECTION):
        import chromadb  # lazy import

        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self._check_embedder_matches()

        self.client = chromadb.PersistentClient(path=str(self.db_path / "chroma"))
        self.col = self.client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}
        )

    def _check_embedder_matches(self) -> None:
        marker = self.db_path / "embedder.txt"
        if marker.exists():
            built_with = marker.read_text().strip()
            if built_with != self.embedder.name:
                raise ValueError(
                    f"This database was built with embedder '{built_with}' but you "
                    f"are using '{self.embedder.name}'. Use the same embedder, or "
                    f"re-ingest into a new database."
                )
        else:
            marker.write_text(self.embedder.name)

    # ------------------------------------------------------------------ #

    def upsert(self, ids: List[str], texts: List[str],
               vectors: List[List[float]], metadatas: List[dict]) -> None:
        for i in range(0, len(ids), _UPSERT_BATCH):
            s = slice(i, i + _UPSERT_BATCH)
            self.col.upsert(ids=ids[s], documents=texts[s],
                            embeddings=vectors[s], metadatas=metadatas[s])

    def query(self, text: str, k: int = 5, where: Optional[dict] = None) -> List[dict]:
        """Return up to k hits: {id, text, metadata, score}. score is cosine
        similarity in [-1, 1] (higher is closer)."""
        total = self.count()
        if total == 0:
            return []
        res = self.col.query(
            query_embeddings=[self.embedder.embed_query(text)],
            n_results=min(k, total),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for i, doc_id in enumerate(res["ids"][0]):
            hits.append({
                "id": doc_id,
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "score": round(1.0 - res["distances"][0][i], 4),
            })
        return hits

    def count(self) -> int:
        return self.col.count()

    def stats(self) -> Dict[str, Dict[str, int]]:
        """Counts per source_type / co_id / bloom_level / course, for sanity checks."""
        data = self.col.get(include=["metadatas"])
        metas = data["metadatas"] or []
        return {
            field: dict(Counter(m.get(field, "") or "(none)" for m in metas))
            for field in ("course", "source_type", "co_id", "bloom_level")
        }
