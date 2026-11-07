"""Malayalam text normaliser shared by the index, the ASR output and the router.

Order matters: old-style chillu sequences (consonant + virama + ZWJ) must be
mapped to atomic chillu characters BEFORE stray ZWJ/ZWNJ are removed,
otherwise "ന്‍" collapses to "ന്" and stops matching "ൻ".
"""
import re
import unicodedata

VIRAMA = "്"
ZWJ, ZWNJ = "‍", "‌"

# consonant -> atomic chillu (Unicode 5.1)
_CHILLU = {
    "ണ": "ൺ",  # ണ -> ൺ
    "ന": "ൻ",  # ന -> ൻ
    "ര": "ർ",  # ര -> ർ
    "ല": "ൽ",  # ല -> ൽ
    "ള": "ൾ",  # ള -> ൾ
    "ക": "ൿ",  # ക -> ൿ
}
_OLD_CHILLU = re.compile("([" + "".join(_CHILLU) + "])" + VIRAMA + ZWJ)
_PUNCT = re.compile(r"[^\w\sഀ-ൿ₹%./-]")
_SPACE = re.compile(r"\s+")


def normalise(text: str, strip_punct: bool = False) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _OLD_CHILLU.sub(lambda m: _CHILLU[m.group(1)], text)
    text = text.replace(ZWJ, "").replace(ZWNJ, "")
    if strip_punct:
        text = _PUNCT.sub(" ", text)
    return _SPACE.sub(" ", text).strip().lower()


def tokens(text: str) -> list[str]:
    """Words plus character 3-grams, so BM25 still matches inflected forms
    (സർട്ടിഫിക്കറ്റ് / സർട്ടിഫിക്കറ്റിന് / സർട്ടിഫിക്കറ്റിനായി)."""
    words = normalise(text, strip_punct=True).split()
    grams = [w[i:i + 3] for w in words if len(w) > 3 for i in range(len(w) - 2)]
    return words + grams
