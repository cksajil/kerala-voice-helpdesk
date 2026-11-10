"""Snap misheard words to the knowledge base's own vocabulary, by sound.

ASR_FIXES in intent.py lists mishearings seen in the logs one by one ("കാലാപതി" -> "കാലാവധി").
The next mishearing is always a new one ("കാലാവരി", "പീസ്", "ഭീസ്"...). This layer handles the
general case: a word the knowledge base has never seen is replaced by the knowledge-base word
that sounds most like it, if one is close enough.

"Sounds like" compares phonetic skeletons: consonants collapsed to their place of articulation
(ക/ഖ/ഗ/ഘ, ത/ഥ/ദ/ധ, പ/ഫ/ബ/ഭ...), chillu = consonant, doubled consonants = single, vowel signs
reduced to five vowels. On the skeletons, a vowel slip costs 0.5, a retroflex/dental or ര/ല slip
0.5, anything else 1.

A different first sound costs 0.5 more: Whisper rarely gets a word's onset wrong, and without this
"അനുകുതി" (അനുമതി, permission) became "നികുതി" (tax) in the 9 Nov log.

Guards (a correction must never break a question Whisper heard right):
  - a word found anywhere in the documents (inflected forms included) is never changed;
  - short words (< 4 letters), numbers and Latin words are never changed, and words of under five
    sounds only when they sound identical (പീസ് = ഫീസ്);
  - two words heard as one are split only into a service word and a known word;
  - the best candidate must be clearly better than the second-best different one.

HELPDESK_ASR_SNAP=0 turns it off.
"""
from __future__ import annotations

import ast
import json
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from .normalise import normalise

DATA = Path(__file__).resolve().parent.parent / "data"
MAX_COST = float(os.getenv("HELPDESK_ASR_SNAP_MAX", "0.25"))  # edit cost / skeleton length
TARGETS = os.getenv("HELPDESK_ASR_SNAP_TARGETS", "domain")    # "domain" or "all" (every KB word)

_CLASS = {}
for letters, c in [("കഖഗഘ", "K"), ("ചഛജഝ", "C"), ("ടഠഡഢ", "T"), ("തഥദധ", "D"), ("പഫബഭ", "P"),
                   ("ങ", "G"), ("ഞ", "J"), ("നണ", "N"), ("മ", "M"), ("യ", "Y"), ("രറ", "R"), ("ലളഴ", "L"),
                   ("വ", "V"), ("ശഷസ", "S"), ("ഹ", "H"),
                   ("അആ", "a"), ("ഇഈ", "i"), ("ഉഊഋ", "u"), ("എഏഐ", "e"), ("ഒഓഔ", "o"),
                   ("ാ", "a"), ("ിീ", "i"), ("ുൂൃ", "u"), ("െേൈ", "e"), ("ൊോൌ", "o"), ("ം", "M")]:
    for ch in letters:
        _CLASS[ch] = c
_CHILLU = {"ൻ": "ന", "ർ": "ര", "ൽ": "ല", "ൾ": "ള", "ൺ": "ണ", "ൿ": "ക"}
_VOWELS = set("aiueo")
_NEAR = {frozenset("TD"), frozenset("RL"), frozenset("NM")}
_WORD = re.compile(r"[ഀ-ൿ]+")


def skeleton(word: str) -> str:
    w = word.replace("റ്റ", "ട്ട").replace("ന്റ", "ണ്ട")   # pronounced tta / nda
    w = "".join(_CHILLU.get(ch, ch) for ch in w)
    out = []
    for ch in w:
        c = _CLASS.get(ch)
        if c and not (out and out[-1] == c and c not in _VOWELS):   # ത്ത -> D, not DD
            out.append(c)
    return "".join(out)


def _sub(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if (a in _VOWELS) == (b in _VOWELS) and (a in _VOWELS or frozenset((a, b)) in _NEAR):
        return 0.5
    return 1.0


def _indel(a: str) -> float:
    return 0.5 if a in _VOWELS else 1.0


def distance(a: str, b: str, cap: float = 99.0) -> float:
    prev = [0.0]
    for ch in b:
        prev.append(prev[-1] + _indel(ch))
    for ca in a:
        cur = [prev[0] + _indel(ca)]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + _indel(ca), cur[j - 1] + _indel(cb), prev[j - 1] + _sub(ca, cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def _words(text: str) -> list[str]:
    return _WORD.findall(normalise(text, strip_punct=True))


@lru_cache(maxsize=1)
def _vocab():
    """(known words, {skeleton: most frequent clean word}, skeletons by length)."""
    known, clean = Counter(), Counter()
    for name in ("kb.jsonl", "canon.jsonl", "docs.jsonl"):
        p = DATA / name
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            d = json.loads(line)
            text = " ".join([d.get("title", ""), d.get("text", "")] + list(d.get("questions") or []))
            ws = _words(text)
            known.update(ws)
            if name != "docs.jsonl":          # snap targets: hand-written and canonical text only,
                clean.update(ws)              # not OCR'd charter text (OCR has its own errors)
    for name in ("graph.py", "tools.py", "fees.py"):  # the helpdesk's own replies (ഇന്നത്തെ, കാലാവസ്ഥ...):
        tree = ast.parse((Path(__file__).parent / name).read_text(encoding="utf-8"))   # strings, not comments
        known.update(w for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                     for w in _words(n.value))
    if TARGETS == "domain":         # snap only to words that decide an answer: service and facet words,
        from .intent import FACETS, SERVICES   # and words of knowledge-base titles
        stems = [normalise(st) for v in SERVICES.values() for st, _ in v if " " not in st]
        stems += [normalise(st) for _, v in FACETS for st in v if " " not in st]
        titles = set()
        for name in ("kb.jsonl", "canon.jsonl"):
            for line in (DATA / name).open(encoding="utf-8"):
                titles.update(_words(json.loads(line).get("title", "")))
        clean = Counter({w: n for w, n in clean.items()
                         if w in titles or any(w.startswith(st) for st in stems if len(st) >= 3)})
    best: dict[str, tuple[int, str]] = {}
    for w, n in clean.items():
        if len(w) >= 3:
            s = skeleton(w)
            if s and (s not in best or n > best[s][0]):
                best[s] = (n, w)
    by_len: dict[int, list[str]] = {}
    for s in best:
        by_len.setdefault(len(s), []).append(s)
    return set(known), {s: w for s, (n, w) in best.items()}, by_len, {w for w in clean if len(w) >= 3}


@lru_cache(maxsize=20000)
def snap_word(word: str, split: bool = True) -> str:
    known, target, by_len, clean = _vocab()
    if word in known or len(word) < 4:
        return word
    if split:                       # two words heard as one: "ജനനരജിസ്ട്രേഷനായി" = ജനന + രജിസ്ട്രേഷനായി
        for i in range(len(word) - 3, 2, -1):
            left, right = word[:i], word[i:]
            if left in clean and len(right) >= 5 and right[0] not in "്ാിീുൂൃെേൈൊോൌംഃ":
                if len(right) >= 8:           # long enough to correct safely
                    right = snap_word(right, split=False)
                if right in known:
                    return f"{left} {right}"
    s = skeleton(word)
    if len(s) < 3:
        return word
    cap = MAX_COST * len(s) if len(s) >= 5 else 0.0   # short words: only an identical sound (പീസ് = ഫീസ്)
    scored = []
    for L in range(len(s) - 2, len(s) + 3):
        for t in by_len.get(L, ()):
            d = distance(s, t, cap) + (0.5 if s[0] != t[0] else 0.0)   # first sounds are heard best
            if d <= cap:
                scored.append((d, t))
    if not scored:
        return word
    scored.sort()
    d0, t0 = scored[0]
    if len(scored) > 1 and scored[1][0] - d0 < 0.5:     # two candidates about as close: don't guess
        return word
    return target[t0]


def snap(text: str) -> str:
    """Replace unknown Malayalam words with their closest-sounding knowledge-base word."""
    if os.getenv("HELPDESK_ASR_SNAP", "1") == "0":
        return text
    return _WORD.sub(lambda m: snap_word(m.group(0)), text)
