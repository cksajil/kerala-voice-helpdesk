"""Hybrid retriever: BM25 (words + char 3-grams) fused with dense BGE-M3 by
reciprocal-rank fusion. Dense is optional: if sentence-transformers is not
installed, or HELPDESK_DENSE=0, it runs BM25 only (handy for tests and CI).

BM25 matters here: fee amounts, form names and words like 'ട്രേഡ് ലൈസൻസ്'
are exact-match signals that dense models blur.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from .normalise import normalise, tokens

KB_PATH = Path(__file__).resolve().parent.parent / "data" / "kb.jsonl"
DENSE_MODEL = os.getenv("HELPDESK_EMBED_MODEL", "BAAI/bge-m3")
BM25_MIN = float(os.getenv("HELPDESK_BM25_MIN", "4.0"))   # tune on eval set
COS_MIN = float(os.getenv("HELPDESK_COS_MIN", "0.55"))    # tune on eval set


@dataclass
class Hit:
    doc: dict
    rrf: float
    bm25: float
    cos: float | None


def load_kb(path: Path = KB_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class Retriever:
    def __init__(self, docs: list[dict] | None = None, dense: bool | None = None):
        self.docs = docs or load_kb()
        # Example questions are indexed (BM25 and dense) but never sent to the LLM.
        # The title counts twice: short, generic entries otherwise outrank the specific one
        # ("ട്രേഡ് ലൈസൻസ് ലഭിക്കാൻ..." went to the Akshaya entry). Chosen on the eval set and
        # the 7 Nov Space log together (eval hit@1 15/15, Space 4/6; the 2 misses are ASR errors).
        texts = [" ".join([d["title"], d["title"], d["text"], *d.get("questions", [])]) for d in self.docs]
        self.bm25 = BM25Okapi([tokens(t) for t in texts])
        if dense is None:
            dense = os.getenv("HELPDESK_DENSE", "1") == "1"
        self.encoder = None
        if dense:
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer(DENSE_MODEL, device="cpu")
                self.doc_vecs = self.encoder.encode(
                    [self._p("passage") + normalise(t) for t in texts], normalize_embeddings=True)
            except Exception as e:  # broken torch/numpy installs raise more than ImportError
                import warnings
                warnings.warn(f"Dense retrieval disabled, using BM25 only: {type(e).__name__}: {e}")
                self.encoder = None

    @staticmethod
    def _p(kind: str) -> str:
        """multilingual-e5 models need 'query: ' / 'passage: ' prefixes; BGE-M3 does not."""
        return f"{kind}: " if "e5" in DENSE_MODEL.lower() else ""

    def search(self, query: str, k: int = 3, rrf_k: int = 60) -> list[Hit]:
        bm = self.bm25.get_scores(tokens(query))
        order_bm = sorted(range(len(self.docs)), key=lambda i: -bm[i])
        rrf = {i: 1 / (rrf_k + r) for r, i in enumerate(order_bm)}
        cos = None
        if self.encoder is not None:
            q = self.encoder.encode([self._p("query") + normalise(query)], normalize_embeddings=True)[0]
            cos = self.doc_vecs @ q
            for r, i in enumerate(sorted(range(len(self.docs)), key=lambda i: -cos[i])):
                rrf[i] += 1 / (rrf_k + r)
        top = sorted(rrf, key=lambda i: -rrf[i])[:k]
        return [Hit(self.docs[i], rrf[i], float(bm[i]),
                    None if cos is None else float(cos[i])) for i in top]

    @staticmethod
    def confident(hits: list[Hit]) -> bool:
        """Gate before the LLM. With a dense model, use cosine only: BM25 scores
        are not comparable across queries (common words like 'ഫീസ്' score high
        even for questions the KB cannot answer). Without one, fall back to BM25."""
        if not hits:
            return False
        best_cos = max((h.cos for h in hits if h.cos is not None), default=None)
        if best_cos is not None:
            return best_cos >= COS_MIN
        return max(h.bm25 for h in hits) >= BM25_MIN
