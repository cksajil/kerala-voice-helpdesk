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
# 1.5 was enough with 2 charters; with 22, a service has dozens of charter rows and one of them
# outranked the curated "₹120 a month" entry for "തൊഴിലില്ലായ്മ വേതനം എത്ര".
TIER1_BOOST = float(os.getenv("HELPDESK_TIER1_BOOST", "2.0"))
# Entries built from the charters (ingest.kb_auto, HELPDESK_KB_AUTO=1) are Tier 1 in schema but not
# hand-checked: at the curated boost they outranked curated entries in the conversation tests
# (widow pension amount, property tax rate). 1.5 = the highest boost with every suite passing
# (13 Nov sweep on Kaggle: synth dev strict 245 -> 267 of 1,109; 1.75 breaks a 12 Nov test).
AUTO_BOOST = float(os.getenv("HELPDESK_AUTO_BOOST", "1.5"))


def tier_boost(d: dict) -> float:
    if d.get("kind") == "kb_auto":
        return AUTO_BOOST
    return TIER1_BOOST if d.get("tier", 1) == 1 else 1.0
# Charter prose (scheme descriptions, notes) is long and mentions many services in passing: a
# pension paragraph that says "വിവാഹം" outranked every marriage entry for "വിവാഹം കഴിഞ്ഞു രണ്ടു
# മാസമായി". It still answers when nothing better matches.
PROSE_WEIGHT = float(os.getenv("HELPDESK_PROSE_WEIGHT", "0.6"))
# Canonical services (HELPDESK_CANON=1) carry 4 indexed questions and up to 6 synonyms each, so
# BM25 favours them on short generic questions ("തുകയെത്രയാണ്?" went to disability pension over the
# Tier-1 amount entry). Tuned on the synth dev split and the conversation tests, not by hand.
CANON_WEIGHT = float(os.getenv("HELPDESK_CANON_WEIGHT", "1.0"))
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


CANON_PATH = Path(__file__).resolve().parent.parent / "data" / "canon.jsonl"
# Tier-1-schema entries built from the charters by ingest/kb_auto.py. Off until the held-out eval
# (eval/heldout_v1.jsonl) shows they help: HELPDESK_KB_AUTO=1.
KB_AUTO_PATH = Path(__file__).resolve().parent.parent / "data" / "kb_auto.jsonl"


def load_all() -> list[dict]:
    """Tier 1 (kb.jsonl) then Tier 2 (docs.jsonl, if built). Set HELPDESK_TIER2=0 to leave it out.
    HELPDESK_CANON=1 adds the canonical services (ingest.canon) and drops the charter rows they
    replace; it stays off until eval/run_synth.py shows it helps."""
    docs = [{**d, "tier": 1} for d in load_kb()]
    if os.getenv("HELPDESK_TIER2", "1") == "1" and DOCS_PATH.exists():
        tier2 = [{**d, "tier": 2} for d in load_kb(DOCS_PATH)]
        if os.getenv("HELPDESK_CANON", "0") == "1" and CANON_PATH.exists():
            canon = [{**d, "tier": 2} for d in load_kb(CANON_PATH)]
            replaced = {m for c in canon for m in c.get("members", [])}
            rows = {d["id"]: d for d in tier2}
            for c in canon:
                # The rows a canonical passage replaces still define what its service talks about:
                # kept for the follow-up vocabulary only, never indexed ("വിവാഹ മോചിതർക്ക്…" stopped
                # being a widow-pension follow-up once the pension rows left the index, 10 Nov).
                c["vocab"] = " ".join(rows[m]["title"] + " " + rows[m]["text"]
                                      for m in c.get("members", []) if m in rows)
            tier2 = canon + [d for d in tier2 if d["id"] not in replaced]
        if os.getenv("HELPDESK_KB_AUTO", "0") == "1" and KB_AUTO_PATH.exists():
            # A promoted service replaces its canonical passage and its charter rows; the rows'
            # words stay as follow-up vocabulary, as for canon.
            auto = [{**d, "tier": 1} for d in load_kb(KB_AUTO_PATH)]
            gone = {m for a in auto for m in a.get("members", [])} | {"canon-" + a["id"].removeprefix("auto-") for a in auto}
            rows = {d["id"]: d for d in load_kb(DOCS_PATH)}
            for a in auto:
                a["vocab"] = " ".join(rows[m]["title"] + " " + rows[m]["text"] for m in a.get("members", []) if m in rows)
            docs += auto
            tier2 = [d for d in tier2 if d["id"] not in gone]
        docs += tier2
    return docs


# Words too common to identify a service on their own (open retrieval, see Retriever.open_match).
_GENERIC = ["സർട്ടിഫിക്കറ്റ", "സർട്ടിഫിക്ക", "അപേക്ഷ", "ലൈസൻസ", "രജിസ്ട്രേഷ", "രജിസ്റ്റ", "പഞ്ചായത്ത",
            "നഗരസഭ", "എത്ര", "എന്ത", "എങ്ങനെ", "എവിടെ", "ലഭിക്ക", "കിട്ട", "വേണ്ട", "വേണം", "വേണോ",
            "ആവശ്യ", "ചെയ്യ", "നൽക", "കേരള", "ദിവസ", "ഫീസ", "രൂപ", "രേഖ", "സേവന", "ഓഫീസ", "എടുക്ക",
            "ഉള്ള", "ഇപ്പോ", "എനിക്ക", "എന്റെ", "നിങ്ങ", "കുറിച്ച", "പറ്റി", "സംബന്ധ", "വിവര", "ഏതൊക്ക",
            "എന്തൊക്ക", "സർവീസ", "പേര", "നിന്റെ", "നീ", "എന്നിവ", "ഉണ്ട", "കഴിയ", "പറയ", "അറിയ", "സമയ", "തുക", "പണം", "കാര്യ", "പുതിയ",
            "തുടങ്ങ", "ആരംഭ", "പോക", "വര", "ഇന്ന", "നാളെ", "എപ്പോ", "ആരാ", "ഏതാ", "പദ്ധതി", "ജോലി",
            # pointing words: "അതിന് എത്ര ദിവസം?" names nothing new
            "അതിന", "ഇതിന", "അതിൽ", "ഇതിൽ", "അത്", "ഇത്", "അവിടെ", "ഇവിടെ", "അങ്ങനെ", "ഇങ്ങനെ", "അപ്പോൾ",
            "എല്ലാ", "മറ്റ", "കൂടി", "മാത്ര", "ആണോ", "ആണ്", "ഉണ്ടോ", "എടുത്ത", "കൊടുക്ക", "പറ്റുമോ",
            # place and action words: "കടയുടെ മുന്നിൽ" matched "ഡോക്ടറുടെ മുന്നിൽ ഹാജരാകണം" (9 Nov log)
            "മുന്നിൽ", "പിന്നിൽ", "അടുത്ത", "കൊണ്ട", "വെക്ക", "വയ്ക്ക", "ഇടാൻ", "പോകണ", "വരണ", "സാധിക്ക"]


# Spelling variants of the same word in charters and ASR output.
_CANON = [("ഇൻഷ്വ", "ഇൻഷു"), ("ഇൻഷൂ", "ഇൻഷു"), ("കുത്തിവയ്", "കുത്തിവെയ്"), ("കുത്തിവെപ്പ", "കുത്തിവെയ്പ്പ"),
          ("കുത്തിവെയ്പ്പ", "കുത്തിവെയ്പ"), ("അനുവാദ", "അനുമതി"), ("പരസ്യബോർഡ", "പരസ്യ ബോർഡ")]


def _place_stems() -> list[str]:
    from .tools import DISTRICTS, PLACES
    return [s for _, stems, *_ in [*DISTRICTS.values(), *PLACES.values()] for s in stems]


_GENERIC += ["നിന്ന", "കേരള", "ഇന്ത്യ", *_place_stems()]   # places say where, not which service


def _content_words(text: str) -> list[str]:
    t = normalise(text, strip_punct=True)
    for a, b in _CANON:
        t = t.replace(a, b)
    return [w for w in t.split()
            if len(w) >= 3 and not any(w.startswith(normalise(g)) for g in _GENERIC)]


_CHILLU_BASE = {"ൽ": "ല", "ർ": "ര", "ൻ": "ന", "ൾ": "ള", "ൺ": "ണ", "ൿ": "ക"}


def _bare(w: str) -> str:
    """Without a final virama or anusvara, and a final chillu as its consonant: 'കാർഡ്' -> 'കാർഡ',
    'പരസ്യം' -> 'പരസ്യ', 'തൊഴിൽ' -> 'തൊഴില', so inflected forms ('കാർഡിനു', 'പരസ്യങ്ങൾ',
    'തൊഴിലുറപ്പ്') start with the whole word."""
    if len(w) > 3 and w[-1] in "്ം":
        w = w[:-1]
    if w and w[-1] in _CHILLU_BASE:
        w = w[:-1] + _CHILLU_BASE[w[-1]]
    return w


def _match_score(a: str, b: str) -> float:
    """1 for the same word, 0.8 for the same word plus a suffix, 0.6 for a shared stem:
    'തൊഴിൽ' matches the row 'തൊഴിൽ കാർഡ്' better than 'കർഷക തൊഴിലാളി പെൻഷൻ' (9 Nov log)."""
    if a == b:
        return 1.0
    if not _stem_match(a, b):
        return 0.0
    a, b = _bare(a), _bare(b)
    n = len(os.path.commonprefix([a, b]))
    return 0.8 if n == min(len(a), len(b)) else 0.6


def _stem_match(a: str, b: str) -> bool:
    """Same word up to inflection: 'പരസ്യം' / 'പരസ്യങ്ങൾ', 'ചികിത്സ' / 'ചികിത്സയ്ക്ക്'.
    A partial match needs 5 shared letters: 'സ്ഥിതി' and 'സ്ഥിര' share only 'സ്ഥി'."""
    if a == b:
        return True
    a, b = _bare(a), _bare(b)
    n = len(os.path.commonprefix([a, b]))
    shorter = min(len(a), len(b))
    return (n >= 3 and n == shorter) or (n >= 5 and n >= 0.65 * shorter)


class Retriever:
    def __init__(self, docs: list[dict] | None = None, dense: bool | None = None):
        self.docs = docs or load_all()
        self._vocab: dict = {}
        self.last_open = (0.0, 0)
        self._title_words = [_content_words(d["title"]) for d in self.docs]
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
              * tier_boost(self.docs[i])
              * (PROSE_WEIGHT if self.docs[i].get("kind") == "prose" else 1)
              * (CANON_WEIGHT if self.docs[i].get("kind") == "canon" else 1) for i in pool}
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

    @staticmethod
    def title_score(query: str, doc: dict) -> float:
        """title_overlap weighted by match quality (exact 1, word+suffix 0.8, shared stem 0.6)."""
        return Retriever._overlap_score(_content_words(fix_asr(query)), doc["title"])

    def _score(self, qw: list[str], i: int) -> tuple[float, int]:
        tw = self._title_words[i]
        best = [max((_match_score(q, t) for t in tw), default=0.0) for q in qw]
        return sum(best), sum(b > 0 for b in best)

    def _coverage(self, qw: list[str], i: int) -> float:
        tw = self._title_words[i]
        return sum(any(_stem_match(q, t) for q in qw) for t in tw) / max(len(tw), 1)

    @staticmethod
    def _overlap_score(qw: list[str], title: str) -> float:
        tw = _content_words(title)
        return sum(max((_match_score(q, t) for t in tw), default=0.0) for q in qw)

    def open_match(self, query: str) -> Hit | None:
        """For a question that names no known service: the passage whose title shares a
        distinctive word with it ('പരസ്യം വെക്കാൻ അനുമതി' -> 'പരസ്യങ്ങൾ പ്രദർശിപ്പിക്കുന്നതിന്
        അനുമതി'), else None. Title overlap, not BM25 alone: BM25 found a pension passage for
        'കേരളത്തിന്റെ തലസ്ഥാനം' because of common suffixes."""
        qw = _content_words(fix_asr(query))
        if not qw:
            return None
        scores = self.bm25.get_scores(tokens(fix_asr(query)))
        best, best_key = None, (0.0, 0.0, False, 0.0, 0)
        for i, d in enumerate(self.docs):
            score, matched = self._score(qw, i)
            if d.get("kind") == "prose" and matched < 2:   # a prose chunk's "title" is just its first words
                continue
            # Ties go to table rows and Tier-1 entries: a prose chunk that mentions the words
            # ("തൊഴിൽ കാർഡിലും മസ്റ്റർ റോളിലും...") is not the service itself.
            # Then: how much of the title the question covers (a 3-word service title over a
            # 15-word OCR line that shares one word), rows over prose, BM25.
            key = (round(score, 2), round(self._coverage(qw, i), 2), d.get("kind") != "prose",
                   scores[i] * tier_boost(d), matched)
            if matched and key > best_key:
                best, best_key = i, key
        if best is None or (best_key[4] < 2 and best_key[3] < BM25_MIN):
            return None
        self.last_open = (best_key[0], best_key[4])          # score, words matched (for the router)
        return Hit(self.docs[best], 0.0, float(best_key[3]), None)

    @staticmethod
    def has_content(query: str) -> bool:
        return bool(_content_words(fix_asr(query)))

    def service_vocab(self, services: list[str]) -> list[str]:
        """Distinctive words of every passage of these services (for the follow-up check)."""
        key = tuple(sorted(services))
        if key not in self._vocab:
            words = set()
            for d in self.docs:
                if d.get("service") in services:
                    words.update(_content_words(" ".join([d["title"], d["text"], d.get("vocab", ""),
                                                          *d.get("questions", [])])))
            self._vocab[key] = sorted(words)
        return self._vocab[key]

    def fits_services(self, query: str, services: list[str]) -> bool:
        """Can this short question continue a conversation about these services? Each of its
        distinctive words must occur in their passages: "ബിരിയാണി എങ്ങനെ ഉണ്ടാക്കാം" after a
        marriage question is not a follow-up (9 Nov log), "പുരുഷന്മാർക്ക് ബാധകമാണോ" is."""
        vocab = self.service_vocab(services)
        return all(any(_stem_match(q, v) for v in vocab) for q in _content_words(fix_asr(query)))

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
