"""Text normaliser for Malayalam TTS: digits and English terms -> Malayalam words.

The VITS voice was trained on Malayalam script only, and the Space drops every
character outside its vocabulary, so "26 ഡിഗ്രി" was spoken as "ഡിഗ്രി" and
"K-SMART" vanished. Most open Malayalam TTS models (MMS/VITS, Indic-Parler,
IndicF5) are trained on normalised text too, so the fix belongs in front of the
TTS, not in a different TTS. The chat bubble keeps the digits; only the text
sent to the voice goes through to_speech().

Number words follow common written Malayalam (Indian grouping: ആയിരം, ലക്ഷം,
കോടി). Inside 1-99 the joining forms are applied (ഇരുപത്തിയഞ്ച്, തൊണ്ണൂറ്റിയൊന്ന്);
larger parts are joined with a space ("ആയിരത്തി അഞ്ഞൂറ്"), which reads naturally.
indic-num2words was tried and rejected: it gives "ഒന്ന് നുറ്" for 100.
"""
from __future__ import annotations

import re

VIRAMA = "്"

UNITS = ["പൂജ്യം", "ഒന്ന്", "രണ്ട്", "മൂന്ന്", "നാല്", "അഞ്ച്", "ആറ്", "ഏഴ്", "എട്ട്", "ഒൻപത്",
         "പത്ത്", "പതിനൊന്ന്", "പന്ത്രണ്ട്", "പതിമൂന്ന്", "പതിനാല്", "പതിനഞ്ച്", "പതിനാറ്",
         "പതിനേഴ്", "പതിനെട്ട്", "പത്തൊൻപത്"]
TENS = {2: "ഇരുപത്", 3: "മുപ്പത്", 4: "നാൽപ്പത്", 5: "അമ്പത്", 6: "അറുപത്", 7: "എഴുപത്",
        8: "എൺപത്", 9: "തൊണ്ണൂറ്"}
HUNDREDS = {1: "നൂറ്", 2: "ഇരുനൂറ്", 3: "മുന്നൂറ്", 4: "നാനൂറ്", 5: "അഞ്ഞൂറ്", 6: "അറുനൂറ്",
            7: "എഴുനൂറ്", 8: "എണ്ണൂറ്", 9: "തൊള്ളായിരം"}

# Independent vowel -> vowel sign, for the "യ" glide: ഇരുപത്തി + അഞ്ച് -> ഇരുപത്തിയഞ്ച്
_GLIDE = {"അ": "", "ആ": "ാ", "ഇ": "ി", "ഈ": "ീ", "ഉ": "ു", "എ": "െ", "ഏ": "േ", "ഒ": "ൊ", "ഓ": "ോ"}


def _joining(word: str) -> str:
    """Form used before another number: പത്ത് -> പത്തി, നൂറ് -> നൂറ്റി, ആയിരം -> ആയിരത്തി."""
    if word.endswith("ം"):
        return word[:-1] + "ത്തി"
    if word.endswith("റ്"):
        return word[:-1] + "്റി"        # നൂറ് -> നൂറ്റി, തൊണ്ണൂറ് -> തൊണ്ണൂറ്റി
    if word.endswith("ത്") and not word.endswith("ത്ത്"):
        return word[:-2] + "ത്തി"       # ഇരുപത് -> ഇരുപത്തി
    if word.endswith(VIRAMA):
        return word[:-1] + "ി"
    return word + "ി"


def _join(left: str, right: str) -> str:
    """Glue a joining form to the next word, adding the യ glide before a vowel."""
    if right[0] in _GLIDE:
        return left + "യ" + _GLIDE[right[0]] + right[1:]
    return left + right


def _below_100(n: int) -> str:
    if n < 20:
        return UNITS[n]
    t, u = divmod(n, 10)
    return TENS[t] if u == 0 else _join(_joining(TENS[t]), UNITS[u])


def _below_1000(n: int) -> str:
    h, r = divmod(n, 100)
    if h == 0:
        return _below_100(r)
    if r == 0:
        return HUNDREDS[h]
    if r >= 10:                                       # നൂറ്റി ഇരുപത്തിയഞ്ച്: easier to read aloud
        return _joining(HUNDREDS[h]) + " " + _below_100(r)
    return _join(_joining(HUNDREDS[h]), _below_100(r))   # നൂറ്റിയഞ്ച്


def _scaled(n: int, word: str) -> str:
    """n thousand / lakh / crore. 1 thousand is just ആയിരം; 2 thousand is രണ്ടായിരം."""
    if word == "ആയിരം":
        if n == 1:
            return "ആയിരം"
        w = number_words(n)
        # irregular thousands: മൂവായിരം, അയ്യായിരം, എണ്ണായിരം, പതിനായിരം, ഇരുപതിനായിരം
        for end, new in (("മൂന്ന്", "മൂവായിരം"), ("ിയഞ്ച്", "യ്യായിരം"), ("ഞ്ച്", "യ്യായിരം"), ("െട്ട്", "െണ്ണായിരം"),
                         ("എട്ട്", "എണ്ണായിരം"), ("ത്ത്", "തിനായിരം"), ("ത്", "തിനായിരം")):
            if w.endswith(end):
                return w[: -len(end)] + new
        return w[:-1] + "ായിരം" if w.endswith(VIRAMA) else w + " ആയിരം"
    return ("ഒരു" if n == 1 else number_words(n)) + " " + word


def number_words(n: int) -> str:
    if n < 0:
        return "മൈനസ് " + number_words(-n)
    if n < 1000:
        return _below_1000(n)
    parts = []
    for size, word in ((10**7, "കോടി"), (10**5, "ലക്ഷം"), (1000, "ആയിരം")):
        q, n = divmod(n, size)
        if q:
            parts.append(_scaled(q, word))
    if n:
        parts.append(_below_1000(n))
    # every part but the last takes its joining form: ആയിരത്തി അഞ്ഞൂറ്, ലക്ഷത്തി ...
    head = [_joining(p) if not p.endswith("കോടി") else p for p in parts[:-1]]
    return " ".join(head + parts[-1:])


def _decimal(int_part: str, frac: str) -> str:
    whole = number_words(int(int_part))
    if frac == "5":                                   # 1.5 -> ഒന്നര, 2.5 -> രണ്ടര
        return (whole[:-1] if whole.endswith(VIRAMA) else whole) + "ര"
    return whole + " ദശാംശം " + " ".join(UNITS[int(d)] for d in frac)


# English terms the knowledge base and tools produce. Longest first.
LEXICON = sorted({
    "Application for New Enterprises License": "ആപ്ലിക്കേഷൻ ഫോർ ന്യൂ എന്റർപ്രൈസസ് ലൈസൻസ്",
    "IFTE and OS License": "ഐ എഫ് ടി ഇ ആൻഡ് ഒ എസ് ലൈസൻസ്",
    "Public Grievance Redressal": "പബ്ലിക് ഗ്രീവൻസ് റിഡ്രസൽ",
    "Civil Registration": "സിവിൽ രജിസ്ട്രേഷൻ", "Property Tax": "പ്രോപ്പർട്ടി ടാക്സ്",
    "Quick Pay": "ക്വിക്ക് പേ", "Google Play Store": "ഗൂഗിൾ പ്ലേ സ്റ്റോർ", "Play Store": "പ്ലേ സ്റ്റോർ",
    "App Store": "ആപ്പ് സ്റ്റോർ", "K-SMART": "കെ സ്മാർട്ട്", "KSMART": "കെ സ്മാർട്ട്",
    "WhatsApp": "വാട്സ്ആപ്പ്", "email": "ഇമെയിൽ", "e-mail": "ഇമെയിൽ", "online": "ഓൺലൈൻ",
    "download": "ഡൗൺലോഡ്", "login": "ലോഗിൻ", "website": "വെബ്സൈറ്റ്", "app": "ആപ്പ്",
    "eDCR": "ഇ ഡി സി ആർ", "Micro": "മൈക്രോ", "Mini": "മിനി", "Small": "സ്മോൾ",
    "Medium": "മീഡിയം", "Large": "ലാർജ്", "manufacturing": "മാനുഫാക്ചറിംഗ്", "service": "സർവീസ്",
}.items(), key=lambda kv: -len(kv[0]))

LETTERS = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                   ["എ", "ബി", "സി", "ഡി", "ഇ", "എഫ്", "ജി", "എച്ച്", "ഐ", "ജെ", "കെ", "എൽ", "എം",
                    "എൻ", "ഒ", "പി", "ക്യൂ", "ആർ", "എസ്", "ടി", "യു", "വി", "ഡബ്ല്യു", "എക്സ്", "വൈ", "സെഡ്"]))

_NUM = re.compile(r"(?<![\d.])(\d{1,3}(?:,\d{2,3})+|\d+)(?:\.(\d+))?(?![\d])")
_RANGE_UNIT = re.compile(r"(\d[\d,.]*)\s*[-–]\s*(\d[\d,.]*)\s+(ലക്ഷം|കോടി|രൂപ|ദിവസം|മാസം|വർഷം)")
_RANGE = re.compile(r"(\d[\d,.]*)\s*[-–]\s*(\d[\d,.]*)")
_ACRONYM = re.compile(r"\b[A-Z]{2,6}\b")


def to_speech(text: str) -> str:
    """Normalise one reply for the TTS. Leaves Malayalam text untouched."""
    t = text
    for en, ml in LEXICON:
        t = re.sub(re.escape(en), ml, t, flags=re.IGNORECASE)
    t = re.sub(r"₹\s*([\d,.]+)", r"\1 രൂപ", t)                    # ₹500 -> 500 രൂപ
    t = re.sub(r"\bRs\.?\s*([\d,.]+)", r"\1 രൂപ", t)
    t = re.sub(r"(\d)\s*%", r"\1 ശതമാനം", t)
    t = _RANGE_UNIT.sub(r"\1 \3 മുതൽ \2 \3 വരെ", t)                # 10-25 ലക്ഷം
    t = _RANGE.sub(r"\1 മുതൽ \2 വരെ", t)
    t = _NUM.sub(lambda m: _decimal(m.group(1).replace(",", ""), m.group(2)) if m.group(2)
                 else number_words(int(m.group(1).replace(",", ""))), t)
    t = _ACRONYM.sub(lambda m: " ".join(LETTERS[c] for c in m.group(0)), t)   # SC, BPL, NOC
    t = re.sub(r"(?<=[\u0D00-\u0D7F])-(?=[\u0D00-\u0D7F])", "", t)     # കെ സ്മാർട്ട്-ലെ -> ...ട്ട്ലെ
    t = t.replace("/", " ")
    return re.sub(r"\s+", " ", t).strip()
