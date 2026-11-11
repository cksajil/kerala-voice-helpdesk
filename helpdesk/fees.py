"""Deterministic fee calculator for questions that need band arithmetic.

A 2B model reads fee tables unreliably ("2 months" -> ₹2 instead of ₹5;
"₹10 lakh service business" -> ₹10,000 instead of ₹500). These questions are
answered here, in code, before retrieval: no LLM call, no prefill, and the
same answer every time. Anything this module cannot parse with confidence
returns None and goes to the normal RAG path.

Sources: data/build_kb.py entries cr-fee-01 (birth/death fees),
svc-charge-01 (service charge), trade-03-svc / trade-03-mfg (licence bands).
"""
from __future__ import annotations

import re

from .normalise import normalise

# Malayalam number words as stems (inflected forms: രണ്ട് / രണ്ടു / രണ്ടാം).
# Longest first, so ഇരുപത്തിയഞ്ച wins over ഇരുപത and ഒന്നര over ഒന്ന.
_WORDS = sorted({
    "ഒന്നര": 1.5, "രണ്ടര": 2.5, "മൂന്നര": 3.5,
    "ഒരു": 1, "ഒന്ന": 1, "രണ്ട": 2, "മൂന്ന": 3, "നാല": 4, "അഞ്ച": 5, "ആറ": 6, "ഏഴ": 7,
    "എട്ട": 8, "ഒൻപത": 9, "ഒമ്പത": 9, "പത്ത": 10, "പതിനൊന്ന": 11, "പന്ത്രണ്ട": 12,
    "പതിനഞ്ച": 15, "ഇരുപത്തിയൊന്ന": 21, "ഇരുപത്തൊന്ന": 21, "ഇരുപത്തിയഞ്ച": 25,
    "ഇരുപത്തഞ്ച": 25, "ഇരുപത": 20, "മുപ്പത": 30, "നാൽപ്പത": 40, "നാല്പത": 40,
    "അമ്പത": 50, "അൻപത": 50, "നൂറ": 100,
}.items(), key=lambda kv: -len(kv[0]))

_TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[ഀ-ൿa-z]+")

DAY_UNITS = [("ദിവസ", 1), ("ആഴ്ച", 7), ("മാസ", 30), ("വർഷ", 365), ("കൊല്ല", 365),
             ("day", 1), ("week", 7), ("month", 30), ("year", 365)]
MONEY_UNITS = [("ലക്ഷ", 1e5), ("lakh", 1e5), ("lac", 1e5), ("കോടി", 1e7), ("crore", 1e7)]

MARRIAGE = ["വിവാഹ", "കല്യാണ", "മാര്യേജ", "marriage"]
POOR = ["ബി.പി.എൽ", "ബിപിഎൽ", "bpl", "ദാരിദ്ര്യ", "പട്ടികജാതി", "പട്ടികവർഗ", "sc/st", "എസ്.സി", "എസ്.ടി"]
BIRTH = ["ജനന", "ജനിച്ച", "പ്രസവ", "birth"]
DEATH = ["മരണ", "മരിച്ച", "death"]
REGISTER = ["രജിസ്റ്റ", "രജിസ്ട്രേഷ", "register"]
NOT_REGISTRATION = ["കോപ്പി", "പകർപ്പ", "തിരുത്ത", "പേര", "എക്സ്ട്രാക്റ്റ", "copy", "correction"]
LICENCE = ["ലൈസൻസ", "license", "licence"]
FEE_INTENT = ["ഫീസ", "രൂപ", "എത്ര", "fee", "ചാർജ"]
MFG = ["നിർമ്മാണ", "ഫാക്ടറി", "മാനുഫാക്ചറിംഗ", "manufactur", "factory"]
SVC = ["സേവന", "സർവീസ", "service", "കട", "ഷോപ്പ", "shop"]


def _has(q: str, stems: list[str]) -> bool:
    return any(s in q for s in stems)


_TENS = {"ഇരുപത": 20, "മുപ്പത": 30, "നാൽപ്പത": 40, "നാല്പത": 40, "അമ്പത": 50, "അൻപത": 50}
_UNITS = {"ഒന്ന": 1, "രണ്ട": 2, "മൂന്ന": 3, "നാല": 4, "അഞ്ച": 5, "ആറ": 6, "ഏഴ": 7, "എട്ട": 8,
          "ഒൻപത": 9, "ഒമ്പത": 9}
_SIGN_TO_VOWEL = {"ൊ": "ഒ", "േ": "ഏ", "െ": "എ", "ാ": "ആ", "ൂ": "ഊ"}


def _tens(tok: str) -> float | None:
    """'ഇരുപത്', 'ഇരുപത്തിമൂന്ന്', 'ഇരുപത്തേഴ്', 'മുപ്പത്തൊന്ന്' -> 20, 23, 27, 31. Anything else after
    the tens word is not read as the bare tens: 'ഇരുപത്തെന്ന' (an ASR slip for 21) was read as 20
    and answered 'within 21 days, no fee' (10 Nov Space log). Unknown -> None, so nothing is computed."""
    for stem, ten in _TENS.items():
        if not tok.startswith(stem):
            continue
        rest = tok[len(stem):]
        if rest in ("", "്", "ു"):
            return float(ten)
        if not rest.startswith("്ത"):
            return None
        rest = rest[2:]                                   # "ത്ത" joins the units word
        if rest.startswith("ി"):                          # ഇരുപത്തി-മൂന്ന്, ഇരുപത്തി-യൊന്ന്
            rest = rest[1:]
            if rest.startswith("യ"):                      # "യ" carries the units word's first vowel
                rest = rest[1:]
                rest = (_SIGN_TO_VOWEL[rest[0]] + rest[1:]) if rest[:1] in _SIGN_TO_VOWEL else "അ" + rest
        elif rest[:1] in _SIGN_TO_VOWEL:                  # ഇരുപത്തൊന്ന്, ഇരുപത്തേഴ്, ഇരുപത്തെട്ട്
            rest = _SIGN_TO_VOWEL[rest[0]] + rest[1:]
        else:                                             # ഇരുപത്തഞ്ച്: inherent "അ"
            rest = "അ" + rest
        for w, u in _UNITS.items():
            if rest.startswith(w):
                return float(ten + u)
        return None
    return None


def _num(tok: str) -> float | None:
    if tok[0].isdigit():
        t = tok.replace(",", "")
        try:
            return float(t)
        except ValueError:
            return None
    for stem, val in _WORDS:
        if stem in _TENS:                                 # compound tens: read the units too, or nothing
            if tok.startswith(stem):
                return _tens(tok)
            continue
        if tok.startswith(stem):
            return float(val)
    return None


def _quantity(q: str, units: list[tuple[str, float]]) -> float | None:
    """First '<number> <unit>' pair, e.g. 'രണ്ട് മാസം' -> 60 (days)."""
    toks = _TOKEN.findall(q)
    for i in range(len(toks) - 1):
        n = _num(toks[i])
        if n is None:
            continue
        for stem, mult in units:
            if toks[i + 1].startswith(stem):
                return n * mult
    return None


# "<N> ദിവസത്തിനു ശേഷം", "<N> ദിവസം കഴിഞ്ഞാൽ": past N, so the band above N. A bare "21 ദിവസം"
# reads as within 21 days. Read "ഇരുപത്തൊന്നു ദിവസത്തിനു ശേഷം" as 21 and answered "no fee, within 21
# days" (10 Nov Space log). Only words right after the unit count: "വിവാഹം കഴിഞ്ഞു രണ്ട് മാസമായി" has
# "കഴിഞ്ഞു" before the quantity, about the event.
AFTER = ["ശേഷം", "ശേഷമ", "കഴിഞ്ഞ", "കഴിഞ്ഞിട്ട", "പിന്നിട്ട", "വൈകി"]
_AFTER_GLUED = re.compile(r"(?:ത്തിന്|ത്തിനു|ത്തിൻ)?\s*ശേഷ")


def _past(q: str, units: list[tuple[str, float]]) -> bool:
    """True when the first '<number> <unit>' is followed by an 'after' word."""
    toks = _TOKEN.findall(q)
    for i in range(len(toks) - 1):
        if _num(toks[i]) is None or not any(toks[i + 1].startswith(st) for st, _ in units):
            continue
        if _AFTER_GLUED.search(toks[i + 1][2:]):                     # "ദിവസത്തിനുശേഷം"
            return True
        nxt = toks[i + 2:i + 4]                                      # "ദിവസത്തിനു ശേഷം", "ദിവസം കഴിഞ്ഞാൽ"
        return any(t.startswith(a) for t in nxt for a in AFTER)
    return False


MONEY_INTENT = ["ഫീസ", "രൂപ", "ചാർജ", "fee", "പിഴ"]
_TUKA = re.compile(r"(?<![ഀ-ൿ])തുക")      # "തുക", "തുകയെത്ര"; not "പുതുക്കാൻ"


def _money(q: str) -> bool:
    return _has(q, MONEY_INTENT) or bool(_TUKA.search(q))


# "എത്ര" alone asks how much ("ജനിച്ചിട്ട് 25 ദിവസം, രജിസ്ട്രേഷന് എത്ര?") unless the question asks for a time,
# a person or a procedure: "എത്ര ദിവസമെടുക്കും" (eval hd028), "ആരുടെ അനുമതി വേണം" (eval hd021).
NOT_MONEY = ["എത്ര ദിവസ", "എത്ര സമയ", "എത്ര നാൾ", "എത്ര ആഴ്ച", "ആരുടെ", "ആരാണ", "അനുമതി", "എങ്ങനെ",
             "എവിടെ", "രേഖ"]


def _asks_amount(q: str) -> bool:
    return _money(q) or (_has(q, ["എത്ര"]) and not _has(q, NOT_MONEY))
MARRIAGE_SCHEDULE = ("വിവാഹം കഴിഞ്ഞ് 45 ദിവസത്തിനകം രജിസ്റ്റർ ചെയ്താൽ രജിസ്ട്രേഷൻ ഫീ 100 രൂപയും "
                     "സർട്ടിഫിക്കറ്റ് ഫീ 20 രൂപയും, ആകെ 120 രൂപ. ദമ്പതികൾ ദാരിദ്ര്യരേഖയ്ക്ക് താഴെയുള്ളവരോ "
                     "പട്ടികജാതി-പട്ടികവർഗക്കാരോ ആണെങ്കിൽ ഫീസ് 10 രൂപ. 45 ദിവസം കഴിഞ്ഞാൽ 100 രൂപയും "
                     "5 വർഷം കഴിഞ്ഞാൽ 250 രൂപയും പിഴ കൂടി അടയ്ക്കണം.")


def marriage_fee(days: float) -> tuple[int, str]:
    """Kerala Registration of Marriages (Common) Rules 2008, as both charters state it:
    registration fee 100 + certificate fee 20; +100 fine after 45 days; +250 fine after 5 years."""
    if days <= 45:
        return 120, "വിവാഹം കഴിഞ്ഞ് 45 ദിവസത്തിനകമായതിനാൽ രജിസ്ട്രേഷൻ ഫീ 100 രൂപയും സർട്ടിഫിക്കറ്റ് ഫീ 20 രൂപയും, ആകെ 120 രൂപ."
    if days <= 5 * 365:
        return 220, ("45 ദിവസം കഴിഞ്ഞതിനാൽ 120 രൂപ ഫീസിനൊപ്പം 100 രൂപ പിഴ കൂടി, ആകെ 220 രൂപ.")
    return 370, ("5 വർഷം കഴിഞ്ഞതിനാൽ 120 രൂപ ഫീസിനൊപ്പം 250 രൂപ പിഴ, ആകെ 370 രൂപ. "
                 "രജിസ്ട്രാർ ജനറലിന്റെ അനുമതിയും വേണം.")


def registration_fee(days: float) -> tuple[int, str]:
    if days <= 21:
        return 0, "21 ദിവസത്തിനുള്ളിൽ ആയതിനാൽ രജിസ്ട്രേഷൻ ഫീസ് ഇല്ല."
    if days <= 30:
        return 2, "21 ദിവസത്തിനു ശേഷം 30 ദിവസത്തിനുള്ളിൽ ആയതിനാൽ രജിസ്ട്രേഷൻ ഫീസ് 2 രൂപ."
    if days <= 365:
        return 5, "30 ദിവസത്തിനു ശേഷം ഒരു വർഷത്തിനുള്ളിൽ ആയതിനാൽ രജിസ്ട്രേഷൻ ഫീസ് 5 രൂപ."
    return 10, "ഒരു വർഷം കഴിഞ്ഞതിനാൽ രജിസ്ട്രേഷൻ ഫീസ് 10 രൂപ."


# (upper limit in rupees, inclusive; fee). Bands from trade-03-svc / trade-03-mfg.
SVC_BANDS = [(10e5, 500), (25e5, 1000), (2e7, 5000), (5e7, 10000), (float("inf"), 15000)]
MFG_BANDS = [(25e5, 500), (1e7, 1000), (5e7, 5000), (10e7, 10000), (float("inf"), 15000)]


def licence_fee(amount: float, sector: str) -> int:
    for limit, fee in (SVC_BANDS if sector == "svc" else MFG_BANDS):
        if amount <= limit:
            return fee
    raise AssertionError("unreachable")


def _rupees(amount: float) -> str:
    if amount >= 1e7:
        return f"{amount / 1e7:g} കോടി രൂപ"
    if amount >= 1e5:
        return f"{amount / 1e5:g} ലക്ഷം രൂപ"
    return f"{amount:,.0f} രൂപ"


LICENCE_CAVEAT = "ഈ തുകകൾ 2018-ലെ നിർദ്ദേശപ്രകാരമുള്ളതാണ്; ഇപ്പോഴത്തെ തുക ഓഫീസിൽ ഉറപ്പാക്കുക."


def answer(query: str) -> dict | None:
    """Return {'answer', 'source_ids'} if this is a fee question we can compute, else None."""
    q = normalise(query)
    q = re.sub(r"(\d)([^\d\s.,])", r"\1 \2", q)  # '2മാസം' -> '2 മാസം'

    # Marriage registration fee: marriage + a delay + a how-much question. Registration words alone are
    # not enough: "ഒരു വർഷത്തിന് ശേഷം അപേക്ഷിച്ചാൽ എത്ര ദിവസമെടുക്കും" asks for time, not ₹220 (eval hd028).
    if _has(q, MARRIAGE) and _asks_amount(q) and not _has(q, NOT_REGISTRATION):
        days = _quantity(q, DAY_UNITS)
        if days is not None and _past(q, DAY_UNITS):
            days += 0.5
        if days is None and _money(q) and not _has(q, ["അക്ഷയ", "സർവീസ് ചാർജ", "ഓൺലൈൻ ചാർജ"]):
            # No delay given: read out the whole schedule. Gemma paraphrased "BPL: ₹10" as
            # "₹10 less than ₹120" in the Space log (8 Nov).
            return {"answer": MARRIAGE_SCHEDULE, "source_ids": ["ch-marr-fee-01"]}
        if days is not None:
            _, rule = marriage_fee(days)
            if _has(q, POOR):
                rule += " ദമ്പതികൾ ദാരിദ്ര്യരേഖയ്ക്ക് താഴെയോ പട്ടികജാതി-പട്ടികവർഗക്കാരോ ആണെങ്കിൽ ഫീസ് 10 രൂപ മാത്രം."
            return {"answer": rule + " ഓൺലൈനായോ അക്ഷയ വഴിയോ അപേക്ഷിക്കുമ്പോൾ സർവീസ് ചാർജ് വേറെ.",
                    "source_ids": ["ch-marr-fee-01", "svc-charge-01"]}

    # Birth/death late-registration fee: needs the event, a delay and a how-much question
    # ("ഒരു വർഷം കഴിഞ്ഞ ജനനം രജിസ്റ്റർ ചെയ്യാൻ ആരുടെ അനുമതി വേണം" asks who approves, not ₹10: eval hd021).
    is_birth, is_death = _has(q, BIRTH), _has(q, DEATH)
    if (is_birth or is_death) and _asks_amount(q) and not _has(q, NOT_REGISTRATION):
        days = _quantity(q, DAY_UNITS)
        if days is not None and _past(q, DAY_UNITS):
            days += 0.5                                              # past the boundary day
        if days is not None:
            _, rule = registration_fee(days)
            return {"answer": rule + " ഓൺലൈനായോ അക്ഷയ വഴിയോ അപേക്ഷിക്കുമ്പോൾ 40 രൂപ സർവീസ് ചാർജ് കൂടി ഉണ്ട്.",
                    "source_ids": ["cr-fee-01", "svc-charge-01"]}

    # Trade licence fee: needs licence + fee intent + an investment amount.
    if _has(q, LICENCE) and _has(q, FEE_INTENT):
        amount = _quantity(q, MONEY_UNITS)
        if amount is None:
            m = re.search(r"\b(\d{5,})\b", q.replace(",", ""))   # plain rupees, e.g. 500000
            amount = float(m.group(1)) if m else None
        if amount is not None:
            amt = _rupees(amount)
            if _has(q, MFG):
                text = f"നിർമ്മാണ മേഖലയിൽ {amt} മുതൽമുടക്കിന് പരമാവധി വാർഷിക ലൈസൻസ് ഫീസ് {licence_fee(amount, 'mfg'):,} രൂപ."
                ids = ["trade-03-mfg"]
            elif _has(q, SVC):
                text = f"സേവന മേഖലയിൽ {amt} മുതൽമുടക്കിന് പരമാവധി വാർഷിക ലൈസൻസ് ഫീസ് {licence_fee(amount, 'svc'):,} രൂപ."
                ids = ["trade-03-svc"]
            else:
                text = (f"{amt} മുതൽമുടക്കിന് പരമാവധി വാർഷിക ലൈസൻസ് ഫീസ് സേവന മേഖലയിൽ "
                        f"{licence_fee(amount, 'svc'):,} രൂപ, നിർമ്മാണ മേഖലയിൽ {licence_fee(amount, 'mfg'):,} രൂപ.")
                ids = ["trade-03-svc", "trade-03-mfg"]
            return {"answer": text + " " + LICENCE_CAVEAT, "source_ids": ids}
    return None
