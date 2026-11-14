"""What is the caller asking about? Service, facet, and ASR clean-up.

One place for the keyword knowledge the router, the retriever and the fee
calculator share. Built from the 7-8 Nov Space logs, where most wrong answers
came from the wrong passage, not from the LLM:
  - "പട്ടിക്ക് ലൈസൻസ്" matched both pet ("പട്ടി") and trade ("ലൈസൻസ്"); trade won.
    Now each stem has a weight and only the most specific services count.
  - "പെർമിറ്റിന്റെ കാലാവധി" retrieved the trade-licence validity. Now the
    question's service filters the candidates, and a facet the knowledge base
    does not have for that service is declined in code instead of guessed.
  - ASR mishears domain words (വേതനം -> "വേദന", പെർമിറ്റ് -> "പർമിത്").
"""
from __future__ import annotations

import difflib
import re

from . import asr_snap
from .normalise import fold, normalise

# service -> [(stem, weight)]; weight 2 = names the service, 1 = generic word
SERVICES: dict[str, list[tuple[str, int]]] = {
    "birth_death": [("ജനന", 2), ("ജനിച്ച", 2), ("പ്രസവ", 2), ("മരണ", 2), ("മരിച്ച", 2), ("ദത്തെടു", 2),
                    ("birth", 2), ("death", 2)],
    "marriage": [("വിവാഹ", 2), ("കല്യാണ", 2), ("മാര്യേജ", 2), ("marriage", 2)],
    "building_permit": [("പെർമിറ്റ", 2), ("കെട്ടിട നിർമ്മാണ", 2), ("വീട് പണി", 2), ("വീട് വെക്ക", 2),
                        ("വീട് വയ്ക്ക", 2), ("പ്ലാൻ അംഗീകാര", 2), ("permit", 2),
                        ("വീടിന്റെ പ്ലാൻ", 2), ("വീട്ടിന്റെ പ്ലാൻ", 2), ("കെട്ടിടത്തിന്റെ പ്ലാൻ", 2),
                        ("കെട്ടിട പ്ലാൻ", 2), ("സൈറ്റ് പ്ലാൻ", 2), ("പ്ലാൻ", 1),
                        # inflected: "വീടിന്റെ പ്ലാനിൽ" has ന, not the chillu
                        ("വീടിന്റെ പ്ലാന", 2), ("വീട്ടിന്റെ പ്ലാന", 2), ("കെട്ടിടത്തിന്റെ പ്ലാന", 2),
                        ("കെട്ടിട പ്ലാന", 2), ("സൈറ്റ് പ്ലാന", 2), ("പ്ലാന", 1)],
    "trade_licence": [("ട്രേഡ", 2), ("കട തുടങ്ങ", 2), ("ഫാക്ടറി", 2), ("സംരംഭ", 2), ("ബിസിനസ", 2),
                      ("ലൈസൻസ", 1), ("license", 1), ("licence", 1)],
    "pension": [("പെൻഷ", 2), ("pension", 2)],
    "unemployment": [("തൊഴിലില്ലായ്മ", 2), ("തൊഴിൽരഹിത", 2), ("തൊഴിൽ രഹിത", 2), ("unemployment", 2)],
    "property_tax": [("നികുതി", 2), ("വീട്ടുകര", 2), ("ടാക്സ", 2), ("ഉടമസ്ഥ", 2), ("ഓണർഷിപ്പ", 2)],
    "pet": [("പട്ടി", 2), ("നായ", 2), ("പന്നി", 2), ("മൃഗ", 2), ("dog", 2),
            ("പൂച്ച", 2), ("പശു", 2), ("ആട", 2), ("cat", 2)],   # not കോഴി/എരുമ/പോത്ത: കോഴിക്കോട്, എരുമേലി, പോത്തൻകോട്
    "certificate": [("താമസ സർട്ടി", 2), ("റെസിഡൻസ", 2), ("സ്വഭാവ സർട്ടി", 2), ("കാരക്ടർ", 2),
                    ("തിരിച്ചറിയൽ സർട്ടി", 2), ("ഐഡന്റിറ്റി", 2), ("സർട്ടി", 1), ("സാക്ഷ്യപത്ര", 1),
                    ("certificate", 1)],
    "grievance": [("പരാതി", 2), ("complaint", 2)],
    "rti": [("വിവരാവകാശ", 2), ("ആർ.ടി.ഐ", 2), ("rti", 2)],
    "health": [("ഹെൽത്ത് കാർഡ", 2), ("മെഡിക്കൽ സർട്ടി", 2), ("ആരോഗ്യ കേന്ദ്ര", 2), ("പി.എച്ച്.സി", 2),
               ("വാക്സിൻ", 2), ("പോളിയോ", 2), ("vaccin", 2)],   # not "കുത്തിവ": title matching finds those rows
    "general": [("അക്ഷയ", 2), ("കുടുംബശ്രീ", 2), ("പ്രൊഫഷണൽ ടാക്സ", 3), ("പ്രൊഫഷൻ ടാക്സ", 3), ("തൊഴിൽ നികുതി", 3), ("professional tax", 3), ("കെ സ്മാർട്ട", 1), ("കെ-സ്മാർട്ട", 1), ("ksmart", 1),
                ("k-smart", 1), ("സർവീസ് ചാർജ", 2), ("ലഭിക്കുന്ന സേവന", 2), ("കിട്ടുന്ന സേവന", 2),
                ("സേവനങ്ങൾ എന്ത", 2), ("ഏതൊക്കെ സേവന", 2), ("എന്തൊക്കെ സേവന", 2)],
}

# Other departments' topics. Without a service that names itself (weight 2), these are out of
# scope and never treated as a follow-up: "പാസ്പോർട്ട് ലഭിക്കാൻ..." right after a marriage
# question must not get marriage documents; "ഡ്രൈവിംഗ് ലൈസൻസ്" is not a trade licence.
OUT_OF_SCOPE = ["പാസ്പോർട്ട", "passport", "ആധാർ", "aadhaar", "വോട്ടർ", "ഡ്രൈവിംഗ", "driving",
                # Revenue (village office): getting the certificate itself, not needing it as a document
                # ("വരുമാന സർട്ടിഫിക്കറ്റ് വെക്കണോ" for a pension stays in scope)
                "ജാതി സർട്ടിഫിക്കറ്റിന", "ജാതി സർട്ടിഫിക്കറ്റ് എടുക്ക", "ജാതി സർട്ടിഫിക്കറ്റ് ലഭി",
                "ജാതി സർട്ടിഫിക്കറ്റ് കിട്ട", "വരുമാന സർട്ടിഫിക്കറ്റിന", "വരുമാന സർട്ടിഫിക്കറ്റ് എടുക്ക",
                "വരുമാന സർട്ടിഫിക്കറ്റ് ലഭി", "വരുമാന സർട്ടിഫിക്കറ്റ് കിട്ട",
                # KSEB bills (connections are not listed: the panchayat ownership certificate is for one)
                "കറന്റ് ബില്ല", "കറന്റ് ബിൽ", "കറണ്ട് ബില്ല", "കറണ്ട് ബിൽ",
                # Water Authority, Supplyco, banks
                "വെള്ളക്കര", "വാട്ടർ ബിൽ", "വാട്ടർ അതോറിറ്റി", "നെല്ല് സംഭരണ", "കൃഷി വായ്പ", "ബാങ്ക് വായ്പ",
                "ലോൺ", "ഗ്യാസ് കണക്ഷ", "കെ.എസ്.ഇ.ബി", "വൈദ്യുതി ബിൽ", "ബസ്", "ട്രെയിൻ",
                "ദൂരം", "ദൂരമ", "തലസ്ഥാന", "സെക്രട്ടറിയറ്റ", "സെക്രട്ടേറിയറ്റ", "പി.എസ്.സി",
                "മുഖ്യമന്ത്രി", "പ്രധാനമന്ത്രി", "മന്ത്രി", "ഗവർണ", "കേരളത്തിൽ എത്ര", "ഇന്ത്യയിൽ",
                "ഇന്ത്യയുടെ"]
# General-knowledge counts: "കേരളത്തിൽ എത്ര പഞ്ചായത്തുകളുണ്ട്" is not a follow-up to property tax.
# "X-ൽ നിന്ന് Y-ലേക്ക്": a route or distance ("തിരുവനന്തപുരത്ത് നിന്നും വർക്കലയിലേക്ക്"), not a service.
_ROUTE_Q = re.compile(r"നിന്ന\S*\s+(?:\S+\s+)?\S+ലേക്ക")
_COUNT_Q = re.compile(r"എത്ര\s+\S*(?:കളു|ങ്ങളു|കൾ\s*ഉ|ങ്ങൾ\s*ഉ)ണ്ട")


# Medical advice: symptoms, or asking which medicine to take. A PHC certificate or health card is
# still in scope (no symptom words). "തലവേദനയും പനിയും, എന്ത് മരുന്ന്" got a palliative-care row (eval hd086).
# Word-start only: "കമ്പനി" (company) contains പനി; "ചുമതല" (duty) is why ചുമ is not listed.
_MEDICAL_Q = re.compile(r"(?<![ഀ-ൿ])(?:പനി|തലവേദന|ജലദോഷ|വയറുവേദന|ഛർദ്ദി|ഷുഗർ|പ്രഷർ)|"
                        r"(?:മരുന്ന|ഗുളിക)\S*\s+(?:\S+\s+)?(?:കഴിക്ക|കുടിക്ക|വാങ്ങ)")
# Services of another state: the help desk only knows Kerala's (eval hd089).
OTHER_STATES = ["തമിഴ്നാട", "കർണാടക", "ആന്ധ്ര", "തെലങ്കാന", "മഹാരാഷ്ട്ര", "ഗോവയിലെ", "ഡൽഹിയിലെ"]


def other_state(query: str) -> bool:
    q = fold(fix_asr(query))
    return any(fold(normalise(s)) in q for s in OTHER_STATES)


# Ration cards belong to Civil Supplies: a new card, or adding, correcting or removing a name on it.
# Not a mention of the card: it is a common document for panchayat services ("അവിവാഹിത പെൻഷന്
# റേഷൻ കാർഡ് വേണമോ"), and "…പേരില്ലെന്ന സർട്ടിഫിക്കറ്റ്" is the panchayat's own certificate.
_RATION_CHANGE = re.compile(r"(?:പുതിയ\s+റേഷ(?:ൻ|ന്)\s*കാർഡ|റേഷ(?:ൻ|ന്)\s*കാർഡിൽ\s+(?:\S+\s+){0,3}?പേര്\s+"
                            r"(?:ചേർ|തിരുത്ത|മാറ്റ|നീക്ക|കുറയ്ക്ക|ഒഴിവാക്ക))")


def ration_card_change(query: str) -> bool:
    q = normalise(fix_asr(query))
    return bool(_RATION_CHANGE.search(q)) and "സർട്ടിഫിക്ക" not in q


def out_of_scope(query: str) -> bool:
    q = fix_asr(query)
    return (any(fold(normalise(s)) in fold(q) for s in OUT_OF_SCOPE + OTHER_STATES) or ration_card_change(q)
            or (bool(_COUNT_Q.search(q)) and "രേഖ" not in q)
            or bool(_ROUTE_Q.search(q)) or bool(_MEDICAL_Q.search(q)))


# Genitive service names for the "I don't have that" reply.
SERVICE_GEN = {
    "birth_death": "ജനന-മരണ രജിസ്ട്രേഷന്റെ", "marriage": "വിവാഹ രജിസ്ട്രേഷന്റെ",
    "building_permit": "കെട്ടിട പെർമിറ്റിന്റെ", "trade_licence": "ട്രേഡ് ലൈസൻസിന്റെ",
    "pension": "ഈ പെൻഷന്റെ", "unemployment": "തൊഴിൽരഹിത വേതനത്തിന്റെ",
    "property_tax": "കെട്ടിട നികുതിയുടെ", "pet": "വളർത്തുമൃഗ ലൈസൻസിന്റെ",
    "certificate": "ഈ സർട്ടിഫിക്കറ്റിന്റെ", "grievance": "പരാതിയുടെ", "rti": "വിവരാവകാശ അപേക്ഷയുടെ",
    "health": "ഈ സേവനത്തിന്റെ", "general": "ഈ സേവനത്തിന്റെ",
}

# facet -> stems. Order is priority when several match (documents before compulsory:
# "എന്തൊക്കെ രേഖകൾ ആവശ്യമാണ്" asks for documents).
FACETS: list[tuple[str, list[str]]] = [
    ("validity", ["കാലാവധി", "വാലിഡിറ്റി", "validity", "പുതുക്ക", "റിന്യൂ", "സാധുവ", "എത്ര വർഷത്തേക്ക"]),
    ("time", ["എത്ര ദിവസ", "എത്ര സമയ", "എത്ര നാൾ", "സമയപരിധി", "എപ്പോൾ കിട്ട", "എപ്പോൾ ലഭി",
              "എത്ര കാത്തിരി", "how long"]),
    ("amount", ["തുക", "രൂപ", "ഫീസ", "ഫീ ", "ചാർജ", "പിഴ", "fee", "how much"]),
    # Naming a document makes "…നിർബന്ധമാണോ / വേണോ" a documents question: "സമ്മതപത്രം നിർബന്ധമാണോ"
    # was declined as an unknown "compulsory" fact although the documents entry lists it (eval hd006, hd033).
    ("documents", ["രേഖ", "ഡോക്യുമെന്റ", "document", "സമ്മതപത്ര", "കരാർ", "സാക്ഷ്യപത്ര", "സത്യവാങ്മൂല",
                   "എഞ്ചിനീയ", "എൻജിനീയ", "സാക്ഷി", "witness"]),
    ("compulsory", ["നിർബന്ധ", "ആവശ്യമാണോ", "ആവശ്യമാണ്", "വേണോ", "എടുക്കണോ", "എടുക്കണം",
                    "compulsory", "mandatory"]),
    ("eligibility", ["ആർക്ക", "അർഹത", "ബാധക", "അപേക്ഷിക്കാമോ", "അപേക്ഷിക്കാനോ", "eligib"]),
    ("amount", ["എത്രയാ", "എത്രെയാ", "എത്ര കിട്ടും"]),          # weak "how much", after the rest
    ("process", ["എങ്ങനെ", "എവിടെ", "എന്ത് ചെയ്യ", "എന്താണ് ചെയ്യ", "നടപടി", "how to"]),
]
FACET_ABOUT = {
    "validity": "കാലാവധിയെക്കുറിച്ച്", "time": "സമയപരിധിയെക്കുറിച്ച്", "amount": "തുകയെക്കുറിച്ച്",
    "documents": "ആവശ്യമായ രേഖകളെക്കുറിച്ച്", "compulsory": "നിർബന്ധമാണോ എന്നതിനെക്കുറിച്ച്",
    "eligibility": "അർഹതയെക്കുറിച്ച്", "process": "നടപടികളെക്കുറിച്ച്",
}
# Facets where a wrong neighbour passage is worse than "I don't have that":
# the 8 Nov log answered permit validity with the trade licence's "5 years".
STRICT_FACETS = {"validity", "time", "amount", "compulsory"}

# Frequent ASR mishearings of domain words: (heard, meant, required context or None).
ASR_FIXES = [
    ("വേദന", "വേതന", "തൊഴി"), ("ഭേദന", "വേതന", None), ("മുർഗ", "മൃഗ", None),
    ("പർമിത", "പെർമിറ്റ", None), ("പെര്മിറ്റ", "പെർമിറ്റ", None), ("കാലാപതി", "കാലാവധി", None),
    ("സക്കിറ്റ്", "സർട്ടിഫിക്കറ്റ്", None), ("കേസ് മാർട്ട", "കെ സ്മാർട്ട", None),
    ("ലൈക്സൻസ", "ലൈസൻസ", None), ("രേഖക്കൾ", "രേഖകൾ", None), (" മെത്ര", " എത്ര", None),
    ("കുറ്റികൾ", "കുട്ടികൾ", None), ("ചനന", "ജനന", None), ("ജനനെ ", "ജനന ", None), ("ജനന് ", "ജനന ", None),
    ("ഇൻഷുറൻസ", "ഇൻഷ്വറൻസ", None), ("കാമെടുക്ക", "കാർഡ് എടുക്ക", "തൊഴിൽ"), ("കാറിലെടുക്ക", "കാർഡ് എടുക്ക", "തൊഴിൽ"), ("കാളാവസ്ഥ", "കാലാവസ്ഥ", None),
    ("വികലാംഗ", "ഭിന്നശേഷി", None), ("വികലാങ്ക", "ഭിന്നശേഷി", None),
    # the charters say "പ്രതിരോധ കുത്തിവയ്പ്", callers say "വാക്സിനേഷൻ"
    ("വാക്സിനേഷൻ", "പ്രതിരോധ കുത്തിവയ്പ്", None), ("വാക്സിനേഷന", "പ്രതിരോധ കുത്തിവയ്പിന", None),
]
# Rewrites for detection only (not shown to the LLM):
_DETECT_ONLY = [
    (re.compile(r"വിവാഹം?\s*മോചി"), "മോചി"),         # divorced: not a marriage-registration question
    (re.compile(r"പട്ടികജാതി|പട്ടികവർഗ"), "എസ്സി"),     # Scheduled Caste/Tribe: not "dog" (പട്ടി)
    (re.compile(r"പട്ടിക(?!്ക)"), "ലിസ്റ്റ്"),            # a list ("വോട്ടർ പട്ടികയിൽ"), not "dog"
    # unmarried: the pension for unmarried women over 50, not marriage registration
    (re.compile(r"അവിവാഹിത\S*|വിവാഹിതരാ\S*ില്ലാത്ത\S*|വിവാഹിതരല്ലാത്ത\S*"), "അവിവ പെൻഷൻ"),  # (no "വിവാഹ" left in it)
]
_TOKEN = re.compile(r"[ഀ-ൿa-z.\-]+")


# Context-dependent fixes: "രണ്ടു നാസമായി" -> "രണ്ടു മാസമായി" (after a number only).
_ASR_REGEX = [
    (re.compile(r"((?:\d+|ഒന്ന|ഒരു|രണ്ട|മൂന്ന|നാല|അഞ്ച|ആറ|ഏഴ|എട്ട|ഒമ്പത|പത്ത|പന്ത്രണ്ട)\S*\s+)നാസ"), r"\1മാസ"),
]


def fix_asr(query: str, snap: bool = True) -> str:
    """Known mishearings (ASR_FIXES, regexes), then the sound-based snap layer. snap=False is the text
    the LLM reads: snap corrections steer routing and retrieval only."""
    q = normalise(query)
    for heard, meant, ctx in ASR_FIXES:
        if heard in q and (ctx is None or ctx in q):
            q = q.replace(heard, meant)
    for pat, rep in _ASR_REGEX:
        q = pat.sub(rep, q)
    return asr_snap.snap(q) if snap else q


def _detect_text(query: str) -> str:
    q = fix_asr(query)
    for pat, rep in _DETECT_ONLY:
        q = pat.sub(rep, q)
    return q


# Short service stems that occur inside unrelated words ("നായ" in "രജിസ്ട്രേഷനായിട്ടുള്ള"):
# word start only. Others stay substring matches so compounds work ("വളർത്തുമൃഗങ്ങൾ").
_SVC_WORD_START = {"നായ", "പന്നി", "ടാക്സ", "dog", "rti", "ആട", "cat", "പശു"}

# Animals the pet-licence entry does not cover (it names dogs and pigs only): say so in code
# instead of "ട്രേഡ് ലൈസൻസിന്റെ..." (9 Nov log, "പൂച്ചയെ വളർത്താൻ ലൈസൻസ്").
OTHER_ANIMALS = ["പൂച്ച", "പശു", "ആട", "cat"]
OTHER_ANIMAL_MSG = ("പട്ടി, പന്നി എന്നിവ വളർത്താനുള്ള ലൈസൻസിനെക്കുറിച്ചുള്ള വിവരം മാത്രമേ എന്റെ പക്കലുള്ളൂ. "
                    "മറ്റു മൃഗങ്ങൾക്ക് ലൈസൻസ് വേണോ എന്ന് പഞ്ചായത്ത് ഓഫീസിൽ ചോദിക്കൂ.")


def other_animal(query: str) -> bool:
    toks = [fold(t) for t in _TOKEN.findall(_detect_text(query))]
    return any(t.startswith(fold(normalise(a))) for a in OTHER_ANIMALS for t in toks)


def _fuzzy_in(stem: str, q: str, toks: list[str]) -> bool:
    """Exact substring, or (long stems only) a near match at a word start, for ASR slips
    the ASR_FIXES list does not cover. Short stems fuzzy-match far too much
    ("പട്ടി" matched inside "രജിസ്ട്രേഷനായിട്ടുള്ള"). q and toks are folded (normalise.fold)."""
    n = len(stem)
    word_start, stem = stem in _SVC_WORD_START, fold(stem)
    if word_start:
        return any(t.startswith(stem) for t in toks)
    if stem in q:
        return True
    if n < 7 or " " in stem:
        return False
    return any(difflib.SequenceMatcher(None, t[: len(stem)], stem).ratio() >= 0.85 for t in toks)


def detect_services(query: str) -> list[str]:
    """Most specific services named in the question (all ties), or []."""
    q = _detect_text(query)
    toks = [fold(t) for t in _TOKEN.findall(q)]
    q = fold(q)
    scores = {}
    for svc, stems in SERVICES.items():
        w = max((wt for st, wt in stems if _fuzzy_in(normalise(st), q, toks)), default=0)
        if w:
            scores[svc] = w
    if not scores:
        return []
    best = max(scores.values())
    if best < 2 and out_of_scope(query):
        return []
    return sorted(s for s, w in scores.items() if w == best)


# Short stems that also occur inside other words ("തുക" in "പുതുക്കാൻ"): match at word start only.
_WORD_START = {"തുക", "രൂപ", "പിഴ", "ഫീസ", "രേഖ", "ചാർജ"}
# Facets that make sense to ask "which service?" about when no service is named.
CLARIFY_FACETS = {"amount", "time", "validity", "documents", "compulsory"}


def detect_facet(query: str) -> str | None:
    q = _detect_text(query)
    toks = [fold(t) for t in _TOKEN.findall(q)]
    q = fold(q)
    for facet, stems in FACETS:
        for s in stems:
            s = normalise(s)
            ws, s = s in _WORD_START, fold(s)
            if (any(t.startswith(s) for t in toks) if ws else s in q):
                return facet
    return None


def missing_message(services: list[str], facet: str) -> str:
    svc = SERVICE_GEN.get(services[0], "ഈ സേവനത്തിന്റെ") if len(services) == 1 else "ഈ സേവനത്തിന്റെ"
    return (f"ക്ഷമിക്കണം, {svc} {FACET_ABOUT[facet]} എന്റെ പക്കൽ ഉറപ്പുള്ള വിവരം ഇല്ല. "
            "പഞ്ചായത്ത് ഓഫീസിലോ അക്ഷയ കേന്ദ്രത്തിലോ ചോദിക്കൂ.")
