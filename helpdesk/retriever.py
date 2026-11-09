"""Hybrid retriever: BM25 (words + char 3-grams) fused with dense BGE-M3 by
reciprocal-rank fusion. Dense is optional: if sentence-transformers is not
installed, or HELPDESK_DENSE=0, it runs BM25 only (handy for tests and CI).

BM25 matters here: fee amounts, form names and words like 'ട്രേഡ് ലൈസൻസ്'
are exact-match signals that dense models blur.
"""
from __future__ import annotations

import json
import re
import os
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from .intent import STRICT_FACETS, detect_facet, detect_services, fix_asr
from .normalise import normalise, tokens

KB_PATH = Path(__file__).resolve().parent.parent / "data" / "kb.jsonl"
# Tier 2: passages built from official documents by ingest/ (not hand-written). Optional.
DOCS_PATH = Path(__file__).resolve().parent.parent / "data" / "docs.jsonl"
# Curated Tier-1 entries win ties: they are checked and written for speech.
TIER1_BOOST = float(os.getenv("HELPDESK_TIER1_BOOST", "1.5"))
DENSE_MODEL = os.getenv("HELPDESK_EMBED_MODEL", "BAAI/bge-m3")
# The question's service filters the candidates and its facet boosts entries that answer
# it (see intent.py). Boost chosen on the eval set and the 7-8 Nov Space logs.
FACET_BOOST = float(os.getenv("HELPDESK_FACET_BOOST", "0.6"))


def mentions_service(query: str) -> bool:
    return bool(detect_services(query))


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


def load_all() -> list[dict]:
    """Tier 1 (kb.jsonl) then Tier 2 (docs.jsonl, if built). Set HELPDESK_TIER2=0 to leave it out."""
    docs = [{**d, "tier": 1} for d in load_kb()]
    if os.getenv("HELPDESK_TIER2", "1") == "1" and DOCS_PATH.exists():
        docs += [{**d, "tier": 2} for d in load_kb(DOCS_PATH)]
    return docs


# Words too common to identify a service on their own (open retrieval, see Retriever.open_match).
_GENERIC = ["സർട്ടിഫിക്കറ്റ", "സർട്ടിഫിക്ക", "അപേക്ഷ", "ലൈസൻസ", "രജിസ്ട്രേഷ", "രജിസ്റ്റ", "പഞ്ചായത്ത",
            "നഗരസഭ", "എത്ര", "എന്ത", "എങ്ങനെ", "എവിടെ", "ലഭിക്ക", "കിട്ട", "വേണ്ട", "വേണം", "വേണോ",
            "ആവശ്യ", "ചെയ്യ", "നൽക", "കേരള", "ദിവസ", "ഫീസ", "രൂപ", "രേഖ", "സേവന", "ഓഫീസ", "എടുക്ക",
            "ഉള്ള", "ഇപ്പോ", "എനിക്ക", "എന്റെ", "നിങ്ങ", "കുറിച്ച", "പറ്റി", "സംബന്ധ", "വിവര", "ഏതൊക്ക",
            "എന്തൊക്ക", "എന്നിവ", "ഉണ്ട", "കഴിയ", "പറയ", "അറിയ", "സമയ", "തുക", "പണം", "കാര്യ", "പുതിയ",
            "തുടങ്ങ", "ആരംഭ", "പോക", "വര", "ഇന്ന", "നാളെ", "എപ്പോ", "ആരാ", "ഏതാ", "പദ്ധതി", "ജോലി"]


def _content_words(text: str) -> list[str]:
    return [w for w in normalise(text, strip_punct=True).split()
            if len(w) >= 4 and not any(w.startswith(normalise(g)) for g in _GENERIC)]


def _stem_match(a: str, b: str) -> bool:
    """Same word up to inflection: 'പരസ്യം' / 'പരസ്യങ്ങൾ', 'ചികിത്സ' / 'ചികിത്സയ്ക്ക്'."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    shorter = min(len(a), len(b))
    # One word is the other plus a suffix ('വേതന' / 'വേതനം'), or they share most of the shorter
    # word ('കുത്തിവെപ്പ്' / 'കുത്തിവെയ്പ്'); 'തൊഴിലില്ലായ്മ' / 'തൊഴിലാളി' share only 'തൊഴില'.
    return a == b or (n >= 4 and n == shorter) or (n >= 5 and n >= 0.65 * shorter)


class Retriever:
    def __init__(self, docs: list[dict] | None = None, dense: bool | None = None):
        self.docs = docs or load_all()
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
                self.doc_vecs = self._doc_vectors([self._p("passage") + normalise(t) for t in texts])
            except Exception as e:  # broken torch/numpy installs raise more than ImportError
                import warnings
                warnings.warn(f"Dense retrieval disabled, using BM25 only: {type(e).__name__}: {e}")
                self.encoder = None

    def _doc_vectors(self, texts: list[str]):
        """Passage embeddings, cached next to the data: with Tier-2 documents there are
        hundreds of passages, and encoding them on a CPU at every start takes minutes."""
        import hashlib

        import numpy as np
        key = hashlib.sha1((DENSE_MODEL + "\n" + "\n".join(texts)).encode()).hexdigest()[:12]
        path = KB_PATH.parent / f"emb-{re.sub(r'[^A-Za-z0-9]+', '-', DENSE_MODEL)}-{key}.npy"
        if path.exists():
            return np.load(path)
        vecs = self.encoder.encode(texts, normalize_embeddings=True, batch_size=32)
        try:
            for old in path.parent.glob(f"emb-{re.sub(r'[^A-Za-z0-9]+', '-', DENSE_MODEL)}-*.npy"):
                old.unlink()
            np.save(path, vecs)
        except OSError:
            pass                                     # read-only disk: just keep them in memory
        return vecs

    @staticmethod
    def _p(kind: str) -> str:
        """multilingual-e5 models need 'query: ' / 'passage: ' prefixes; BGE-M3 does not."""
        return f"{kind}: " if "e5" in DENSE_MODEL.lower() else ""

    def search(self, query: str, k: int = 3, rrf_k: int = 60, services: list[str] | None = None,
               facet: str | None = None) -> list[Hit]:
        """BM25 (+ dense) over the entries of the question's services, facet-boosted.
        services/facet default to what intent.py detects in the query; pass them to
        override (follow-ups carry the previous question's service)."""
        query = fix_asr(query)
        if services is None:
            services = detect_services(query)
        if facet is None:
            facet = detect_facet(query)
        pool = [i for i, d in enumerate(self.docs) if not services or d.get("service") in services]
        if not pool:
            pool = list(range(len(self.docs)))
        scores = self.bm25.get_scores(tokens(query))
        bm = {i: scores[i] * (1 + FACET_BOOST if facet and facet in self.docs[i].get("facets", []) else 1)
              * (TIER1_BOOST if self.docs[i].get("tier", 1) == 1 else 1) for i in pool}
        order_bm = sorted(pool, key=lambda i: -bm[i])
        rrf = {i: 1 / (rrf_k + r) for r, i in enumerate(order_bm)}
        cos = None
        if self.encoder is not None:
            qv = self.encoder.encode([self._p("query") + normalise(query)], normalize_embeddings=True)[0]
            cos = self.doc_vecs @ qv
            for r, i in enumerate(sorted(pool, key=lambda i: -cos[i])):
                rrf[i] += 1 / (rrf_k + r)
        top, seen = [], set()
        for i in sorted(rrf, key=lambda i: -rrf[i]):
            key = normalise(self.docs[i]["title"], strip_punct=True)   # same row from several charters
            if key in seen:
                continue
            seen.add(key)
            top.append(i)
            if len(top) == k:
                break
        return [Hit(self.docs[i], rrf[i], float(bm[i]),
                    None if cos is None else float(cos[i])) for i in top]

    @staticmethod
    def title_overlap(query: str, doc: dict) -> int:
        """How many distinctive words of the question the passage title has (up to inflection)."""
        tw = _content_words(doc["title"])
        return sum(any(_stem_match(q, t) for t in tw) for q in _content_words(fix_asr(query)))

    def open_match(self, query: str) -> Hit | None:
        """For a question that names no known service: the passage whose title shares a
        distinctive word with it ('പരസ്യം വെക്കാൻ അനുമതി' -> 'പരസ്യങ്ങൾ പ്രദർശിപ്പിക്കുന്നതിന്
        അനുമതി'), else None. Title overlap, not BM25 alone: BM25 found a pension passage for
        'കേരളത്തിന്റെ തലസ്ഥാനം' because of common suffixes."""
        qw = _content_words(fix_asr(query))
        if not qw:
            return None
        scores = self.bm25.get_scores(tokens(fix_asr(query)))
        best, best_key = None, (0, 0.0)
        for i, d in enumerate(self.docs):
            tw = _content_words(d["title"])
            matched = sum(any(_stem_match(q, t) for t in tw) for q in qw)
            if d.get("kind") == "prose" and matched < 2:   # a prose chunk's "title" is just its first words
                continue
            key = (matched, scores[i] * (TIER1_BOOST if d.get("tier", 1) == 1 else 1))
            if matched and key > best_key:
                best, best_key = i, key
        if best is None or (best_key[0] < 2 and best_key[1] < BM25_MIN):
            return None
        return Hit(self.docs[best], 0.0, float(best_key[1]), None)

    def missing_facet(self, services: list[str], facet: str | None) -> bool:
        """True when the question asks a strict facet (amount, time, validity, compulsory)
        that no entry of its service(s) answers: decline instead of guessing."""
        if not services or facet not in STRICT_FACETS:
            return False
        return not any(facet in d.get("facets", []) for d in self.docs if d.get("service") in services)

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
