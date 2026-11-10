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
  - two words heard as one are split only into a service name and a known word;
  - another ending of the same word is not a correction (സാക്ഷ്യപത്രത്തിന് stays): the index matches stems;
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
MAX_COST = float(os.getenv("HELPDESK_ASR_SNAP_MAX", "0.15"))  # edit cost / skeleton length
TARGETS = os.getenv("HELPDESK_ASR_SNAP_TARGETS", "domain")    # "domain" or "all" (every KB word)
CONTEXT = os.getenv("HELPDESK_ASR_SNAP_CONTEXT", "1") == "1"  # also fix real words in the wrong place
CONTEXT_MAX = float(os.getenv("HELPDESK_ASR_SNAP_CONTEXT_MAX", "0.3"))

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
_SIGNS = set("ാിീുൂൃെേൈൊോൌ")
VIRAMA = "്"
_NEAR = {frozenset("TD"), frozenset("RL"), frozenset("NM")}
_WORD = re.compile(r"[ഀ-ൿ]+")


def skeleton(word: str) -> str:
    """ജനന -> "CaNaNa", കാലാവധി -> "KaLaVaDi", പെർമിറ്റ് -> "PeRMiT", ഫീസ് -> "PiS"."""
    w = word.replace("റ്റ", "ട്ട").replace("ന്റ", "ണ്ട")   # pronounced tta / nda
    out = []
    for i, ch in enumerate(w):
        if ch in _CHILLU:
            out.append(_CLASS[_CHILLU[ch]])                  # chillu: the consonant, no vowel
            continue
        c = _CLASS.get(ch)
        if not c:
            continue
        nxt = w[i + 1] if i + 1 < len(w) else ""
        if c in _VOWELS or ch == "ം" or ch in _SIGNS:
            out.append(c)
        elif nxt == VIRAMA:
            after = w[i + 2] if i + 2 < len(w) else ""
            if _CLASS.get(after) != c:                       # ത്ത -> one D (doubling is not heard reliably)
                out.append(c)
        else:
            out.append(c)
            if nxt not in _SIGNS:
                out.append("a")                              # inherent vowel
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
            if (left in clean and _names_service(left) and len(right) >= 5
                    and right[0] not in "്ാിീുൂൃെേൈൊോൌംഃ" and right[1] != VIRAMA):   # not ...ത്തിന്റെ
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
    if any(d - d0 < 0.5 and not _family(target[t], target[t0]) for d, t in scored[1:]):
        return word                                   # two different words about as close: don't guess
    if _family(word, target[t0]):                     # another ending of the same word: the index matches
        return word                                   # stems anyway, and the swap only moves the ranking
    return target[t0]


def _family(a: str, b: str) -> bool:
    """Forms of one word: ജനന / ജനനം, പെർമിറ്റ് / പെർമിറ്റിന്റെ."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n >= max(3, min(len(a), len(b)) - 1)


@lru_cache(maxsize=1)
def _context() -> dict[str, Counter]:
    """Next word's first 5 letters -> Counter of the service words that come before it in the
    knowledge base ("രജിസ്" <- ജനന, മരണ, വിവാഹ...)."""
    _, _, _, clean = _vocab()
    before: dict[str, Counter] = {}
    for name in ("kb.jsonl", "canon.jsonl"):
        for line in (DATA / name).open(encoding="utf-8"):
            d = json.loads(line)
            for text in [d.get("title", ""), d.get("text", "")] + list(d.get("questions") or []):
                ws = _words(text)
                for a, b in zip(ws, ws[1:]):
                    if len(b) >= 5:
                        before.setdefault(b[:5], Counter())[a] += 1
    return {k: Counter({w: n for w, n in c.items() if _names_service(w) and n >= 3})
            for k, c in before.items() if sum(c.values()) >= 10}         # only service names: ജനന, മരണ...


def snap_in_context(prev: str, nxt: str) -> str:
    """A real word in the wrong place: "ജനറൽ രജിസ്ട്രേഷൻ" (no such service) -> "ജനന രജിസ്ട്രേഷൻ".
    prev is replaced only if the knowledge base never puts it before nxt, and a service word that does
    sounds almost the same."""
    if not CONTEXT or len(nxt) < 5 or len(prev) < 3 or _names_service(prev):
        return prev                     # പൂച്ചയെ (cat) must not become പട്ടിയെ (dog)
    seen = _context().get(nxt[:5])
    full = _CONTEXT_ALL().get(nxt[:5], ())
    if not seen or prev in full:
        return prev
    s = skeleton(prev)
    cap = CONTEXT_MAX * len(s)
    scored = sorted((distance(s, skeleton(t), cap) + (0.5 if s[:1] != skeleton(t)[:1] else 0.0), t)
                    for t in seen)
    scored = [x for x in scored if x[0] <= cap]
    if not scored:
        return prev
    rivals = [t for d, t in scored[1:] if d - scored[0][0] < 0.5 and not _family(t, scored[0][1])]
    if rivals:
        return prev
    family = [t for d, t in scored if d - scored[0][0] < 0.5]
    return max(family, key=lambda t: (seen[t], -len(t)))     # ജനന (48) over ജനനം (13)


def _asks_service_thing(text: str) -> bool:
    """Fee, documents, time, validity... or a word of a knowledge-base title other than a service name."""
    from .intent import FACETS
    if any(normalise(st) in text for _, v in FACETS for st in v):
        return True
    return False


@lru_cache(maxsize=4096)
def _names_service(word: str) -> bool:
    from .intent import SERVICES
    return any(word.startswith(normalise(st)) for v in SERVICES.values() for st, _ in v if " " not in st)


@lru_cache(maxsize=1)
def _CONTEXT_ALL() -> dict[str, set]:
    """Every word (documents included) seen before a next word's first 5 letters."""
    allb: dict[str, set] = {}
    for name in ("kb.jsonl", "canon.jsonl", "docs.jsonl"):
        for line in (DATA / name).open(encoding="utf-8"):
            d = json.loads(line)
            for text in [d.get("title", ""), d.get("text", "")] + list(d.get("questions") or []):
                ws = _words(text)
                for a, b in zip(ws, ws[1:]):
                    allb.setdefault(b[:5], set()).add(a)
    return allb


def snap(text: str) -> str:
    """Replace unknown Malayalam words with their closest-sounding knowledge-base word, and
    service words heard as another real word (by the word that follows)."""
    if os.getenv("HELPDESK_ASR_SNAP", "1") == "0":
        return text
    words = [m.group(0) for m in _WORD.finditer(text)]
    fixed = {w: snap_word(w) for w in words}
    if any(_names_service(n) and not _names_service(w) for w, n in fixed.items() if n != w):
        cue = any(_names_service(w) for w in words) or _asks_service_thing(" ".join(fixed.values()))
        if not cue:                    # "കഴിക്കാൻ പറ്റിയ കഞ്ഞി" must not become a dog-licence question
            fixed = {w: (w if _names_service(n) and not _names_service(w) else n) for w, n in fixed.items()}
    text = _WORD.sub(lambda m: fixed[m.group(0)], text)
    if not CONTEXT:
        return text
    ms = list(_WORD.finditer(text))
    out, last = [], 0
    for a, b in zip(ms, ms[1:]):
        if text[a.end():b.start()].strip():             # punctuation between: not a phrase
            continue
        new = snap_in_context(a.group(0), b.group(0))
        if new != a.group(0):
            out.append(text[last:a.start()] + new)
            last = a.end()
    return "".join(out) + text[last:]
