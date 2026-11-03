"""
embedder.py
-----------
Swappable text embedders. Everything else in the pipeline talks to the small
`Embedder` interface, so the model can change without touching other code.

  SentenceTransformerEmbedder : the REAL one. Default BAAI/bge-small-en-v1.5
                                (384-d, fast, good retrieval quality, runs on
                                the 24 GB GPU or even CPU).
  HashingEmbedder             : a deterministic, download-free stand-in used
                                ONLY for offline tests of the pipeline
                                mechanics. It matches shared words, not
                                meaning, so never use it for real retrieval.

All vectors are L2-normalised, so dot product == cosine similarity.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List


class Embedder:
    name: str = "base"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class SentenceTransformerEmbedder(Embedder):
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5",
                 device: str | None = None, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self.name = f"st:{model_name}"
        self._query_prefix = self.BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vecs = self.model.encode(texts, batch_size=self.batch_size,
                                 normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([self._query_prefix + text])[0]


class HashingEmbedder(Embedder):
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hash:{dim}"

    @staticmethod
    def _features(text: str) -> List[str]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        bigrams = [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        return tokens + bigrams

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for feat in self._features(text):
            h = int(hashlib.md5(feat.encode("utf-8")).hexdigest(), 16)
            sign = 1.0 if (h >> 64) & 1 else -1.0
            vec[h % self.dim] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]


def make_embedder(kind: str = "st", model_name: str | None = None,
                  device: str | None = None) -> Embedder:
    if kind == "hash":
        return HashingEmbedder()
    if kind == "st":
        return SentenceTransformerEmbedder(model_name or "BAAI/bge-small-en-v1.5",
                                           device=device)
    raise ValueError(f"Unknown embedder kind: {kind!r} (use 'st' or 'hash')")
