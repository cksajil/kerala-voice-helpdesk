"""Offline tests: no LLM, no network (tools and LLM are monkeypatched)."""
import json
import os

os.environ["HELPDESK_DENSE"] = "0"
os.environ["HELPDESK_LLM"] = "none"

import pytest

from helpdesk import graph as G
from helpdesk import llm, tools
from helpdesk.normalise import normalise, tokens
from helpdesk.retriever import Retriever


@pytest.fixture(scope="module")
def app():
    return G.build_graph(Retriever(dense=False))


# --- normaliser -------------------------------------------------------------
def test_old_chillu_becomes_atomic_before_zwj_strip():
    old = "ന്‍"          # ന + ് + ZWJ (old-style ൻ)
    assert normalise(old) == "ൻ"   # ൻ, not ന്
    assert normalise("ലൈസ" + old + "സ്") == normalise("ലൈസൻസ്")


def test_stray_zwnj_removed_and_nfc():
    assert "‌" not in normalise("ക‌ക")


def test_tokens_include_char_trigrams():
    t = tokens("സർട്ടിഫിക്കറ്റിന്")
    assert "സർട്ടിഫിക്കറ്റിന്" in t and len(t) > 1


# --- router -----------------------------------------------------------------
@pytest.mark.parametrize("q,route", [
    ("കൊച്ചിയിൽ ഇന്ന് മഴയുണ്ടോ?", "weather"),
    ("ഇന്നത്തെ വാർത്തകൾ", "news"),
    ("ട്രേഡ് ലൈസൻസ് പുതുക്കാൻ എന്ത് ചെയ്യണം?", "service_faq"),
    ("ജനന സർട്ടിഫിക്കറ്റ് എങ്ങനെ കിട്ടും", "service_faq"),
])
def test_rule_router(q, route):
    assert G.rule_route(q) == route


@pytest.mark.parametrize("q,d", [
    ("തിരുവനന്തപുരത്ത് ചൂട് എത്ര?", "thiruvananthapuram"),
    ("കോഴിക്കോട്ടെ കാലാവസ്ഥ", "kozhikode"),
    ("weather in Kochi", "ernakulam"),
    ("ഇന്ന് മഴയുണ്ടോ?", None),
])
def test_find_district(q, d):
    assert tools.find_district(q) == d


# --- graph ------------------------------------------------------------------
def test_service_question_extractive_without_llm(app):
    s = app.invoke({"query": "ട്രേഡ് ലൈസൻസ് പുതുക്കാൻ എന്ത് ചെയ്യണം?"})
    assert s["route"] == "service_faq" and not s["refused"]
    assert s["sources"][0]["id"] == "trade-04"
    assert "30 ദിവസം" in s["answer"]


def test_weather_uses_tool_text(app, monkeypatch):
    monkeypatch.setattr(tools, "weather", lambda q: {"text": "എറണാകുളം: ഇപ്പോൾ 29 ഡിഗ്രി", "source": "Open-Meteo"})
    s = app.invoke({"query": "കൊച്ചിയിൽ മഴയുണ്ടോ?"})
    assert s["route"] == "weather" and "29" in s["answer"]


def test_tool_failure_is_graceful(app, monkeypatch):
    def boom(q):
        raise RuntimeError("network down")
    monkeypatch.setattr(tools, "news", boom)
    s = app.invoke({"query": "ഇന്നത്തെ വാർത്തകൾ"})
    assert s["answer"] == G.TOOL_FAIL and s["refused"]


def test_llm_unsure_becomes_refusal(app, monkeypatch):
    monkeypatch.setattr(llm, "enabled", lambda: True)
    monkeypatch.setattr(llm, "chat", lambda *a, **k: "UNSURE")
    s = app.invoke({"query": "കെട്ടിട നികുതി അടയ്ക്കേണ്ട അവസാന ദിവസം ഏതാണ്?"})
    assert s["route"] == "service_faq" and s["refused"] and s["answer"] == G.REFUSAL


def test_no_service_goes_to_scope_without_an_llm_call(app, monkeypatch):
    # 8 Nov: LLM classification sent "കേരളത്തിന്റെ തലസ്ഥാനം" to retrieval -> "തി ആണ്". Removed.
    def no_llm(*a, **k):
        raise AssertionError("no LLM call expected")
    monkeypatch.setattr(llm, "enabled", lambda: True)
    monkeypatch.setattr(llm, "chat", no_llm)
    s = app.invoke({"query": "നിന്റെ പേരെന്താ?"})
    assert s["route"] == "other" and s["answer"] == G.SCOPE


@pytest.mark.parametrize("q", ["ഡ്രൈവിംഗ് ലൈസൻസ് എങ്ങനെ എടുക്കാം?", "പാസ്പോർട്ട് പുതുക്കാൻ എത്ര ഫീസ് ആകും?",
                               "ആധാർ കാർഡിൽ പേര് മാറ്റാൻ"])
def test_other_departments_are_out_of_scope(app, q):
    assert G.rule_route(q) is None
    assert app.invoke({"query": q})["answer"] == G.SCOPE          # not "which service?"


# --- news parsing -----------------------------------------------------------
def _get_without_district_feeds(response):
    """Mock requests.get where every district search feed is down, so the general
    feeds and the district filter are exercised."""
    district_urls = {tools._district_feed(k)[1] for k in tools.DISTRICTS}

    def get(url, **k):
        if url in district_urls:
            raise tools.requests.RequestException("district feed down")
        return response
    return get


RSS = b"""<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel><title>t</title>
<item><title>\xe0\xb4\x95\xe0\xb5\x8a\xe0\xb4\x9a\xe0\xb5\x8d\xe0\xb4\x9a\xe0\xb4\xbf news one</title><link>https://x/1</link>
<description>&lt;p&gt;first &amp;amp; summary&lt;/p&gt;</description></item>
<item><title>Second headline</title><link>https://x/2</link><description>two</description></item>
</channel></rss>"""


def test_news_parses_and_filters_by_district(monkeypatch):
    class R:
        content = RSS
        def raise_for_status(self): pass
    tools._CACHE.clear()
    monkeypatch.setattr(tools.requests, "get", _get_without_district_feeds(R()))
    out = tools.news("കൊച്ചിയിലെ വാർത്തകൾ")
    assert out["items"][0]["title"].startswith("കൊച്ചി")
    assert len(out["items"]) == 1                   # filtered to the district
    assert "<p>" not in out["items"][0]["summary"]  # HTML stripped


# --- gate and in-process LLM ------------------------------------------------
def test_gate_uses_cosine_when_dense_present():
    from helpdesk.retriever import Hit
    high_bm25_low_cos = [Hit(doc={}, rrf=0.1, bm25=50.0, cos=0.20)]
    assert not Retriever.confident(high_bm25_low_cos)
    assert Retriever.confident([Hit(doc={}, rrf=0.1, bm25=0.5, cos=0.90)])
    assert Retriever.confident([Hit(doc={}, rrf=0.1, bm25=50.0, cos=None)])  # BM25 fallback


def test_in_process_llama_backend():
    class FakeLlama:
        def create_chat_completion(self, messages, **kw):
            return {"choices": [{"message": {"content": " ശരി "}}]}
    llm.use_llama_cpp(FakeLlama())
    try:
        assert llm.chat("s", "u") == "ശരി"
    finally:
        llm.use_llama_cpp(None)


# --- fee calculator ---------------------------------------------------------
from helpdesk import fees


@pytest.mark.parametrize("q,expected", [
    ("കുഞ്ഞ് ജനിച്ച് പത്ത് ദിവസമായി, ജനനം രജിസ്റ്റർ ചെയ്യാൻ ഫീസ് എത്രയാണ്?", "ഫീസ് ഇല്ല"),
    ("മരണം നടന്ന് രണ്ട് മാസം കഴിഞ്ഞു, ഇപ്പോൾ രജിസ്റ്റർ ചെയ്യാൻ എത്ര രൂപ ആകും?", "ഫീസ് 5 രൂപ"),
    ("ജനിച്ചിട്ട് 25 ദിവസം, രജിസ്ട്രേഷന് എത്ര?", "ഫീസ് 2 രൂപ"),
    ("മൂന്ന് ആഴ്ച മുൻപ് ജനിച്ച കുട്ടിയുടെ ജനന രജിസ്ട്രേഷൻ ഫീസ്", "ഫീസ് ഇല്ല"),        # 21 days
    ("ഒന്നര വർഷം മുൻപ് മരിച്ചു, മരണം രജിസ്റ്റർ ചെയ്യാൻ ഫീസ്", "ഫീസ് 10 രൂപ"),
    ("birth registration fee after 2months", "ഫീസ് 5 രൂപ"),
])
def test_registration_fee(q, expected):
    out = fees.answer(q)
    assert out and expected in out["answer"] and "cr-fee-01" in out["source_ids"]


@pytest.mark.parametrize("q,expected", [
    ("പത്ത് ലക്ഷം രൂപ മുതൽമുടക്കുള്ള സർവീസ് സ്ഥാപനത്തിന് ലൈസൻസ് ഫീസ് എത്ര?", "500 രൂപ"),
    ("30 ലക്ഷം മുതൽമുടക്കുള്ള ഫാക്ടറിക്ക് ലൈസൻസ് ഫീസ്", "1,000 രൂപ"),
    ("ഒന്നര കോടി രൂപയുടെ സേവന സ്ഥാപനത്തിന് ലൈസൻസിന് എത്ര രൂപ", "5,000 രൂപ"),
    ("12 ലക്ഷം രൂപ മുതൽമുടക്ക്, ലൈസൻസ് ഫീസ് എത്ര?", "സേവന മേഖലയിൽ 1,000 രൂപ, നിർമ്മാണ മേഖലയിൽ 500 രൂപ"),
])
def test_licence_fee(q, expected):
    out = fees.answer(q)
    assert out and expected in out["answer"]


@pytest.mark.parametrize("q", [
    "ജനന സർട്ടിഫിക്കറ്റിൽ കുട്ടിയുടെ പേര് ചേർക്കാൻ എത്ര രൂപ?",      # name inclusion, not registration
    "ഒരു വർഷം മുൻപ് മരിച്ച ആളുടെ സർട്ടിഫിക്കറ്റ് കോപ്പി എടുക്കാൻ ഫീസ്",  # copy, not registration
    "ജനന സർട്ടിഫിക്കറ്റ് കിട്ടാൻ എത്ര ദിവസം എടുക്കും?",               # no number
    "ട്രേഡ് ലൈസൻസ് കിട്ടാൻ എത്ര ദിവസം എടുക്കും?",                   # licence, but no amount
    "അക്ഷയ വഴി ജനന രജിസ്ട്രേഷന് സർവീസ് ചാർജ് എത്ര?",
    "പാസ്പോർട്ട് പുതുക്കാൻ എത്ര ഫീസ് ആകും?",
])
def test_fee_calculator_does_not_fire(q):
    assert fees.answer(q) is None


def test_band_edges():
    assert [fees.registration_fee(d)[0] for d in (21, 22, 30, 31, 365, 366)] == [0, 2, 2, 5, 5, 10]
    assert fees.licence_fee(10e5, "svc") == 500 and fees.licence_fee(10e5 + 1, "svc") == 1000
    assert fees.licence_fee(25e5, "mfg") == 500 and fees.licence_fee(25e5 + 1, "mfg") == 1000


def test_fee_question_skips_llm(app, monkeypatch):
    def no_llm(*a, **k):
        raise AssertionError("LLM must not be called for computed fees")
    monkeypatch.setattr(llm, "enabled", lambda: True)
    monkeypatch.setattr(llm, "chat", no_llm)
    s = app.invoke({"query": "മരണം നടന്ന് രണ്ട് മാസം കഴിഞ്ഞു, ഇപ്പോൾ രജിസ്റ്റർ ചെയ്യാൻ എത്ര രൂപ ആകും?"})
    assert s["tool"] == "fee_calc" and "5 രൂപ" in s["answer"]
    assert [x["id"] for x in s["sources"]] == ["cr-fee-01", "svc-charge-01"]


def test_warmup_prompt_matches_answer_prompt_prefix():
    """The cache only helps if warm-up and real answers share the same prefix."""
    from helpdesk.retriever import load_kb
    d = load_kb()[0]
    warm = G.answer_message(G.passage(d), "x")
    real = G.answer_message(G.passage(d), "മറ്റൊരു ചോദ്യം")
    shared = warm[: warm.index("ചോദ്യം:")]
    assert real.startswith(shared) and G.passage(d) in shared


# --- regressions from the Space log (7 Nov) ----------------------------------
@pytest.mark.parametrize("out", [
    "പാസ്പോർട്ട് പുതുക്കുന്നതിനെക്കുറിച്ച് വിവരങ്ങൾ നൽകിയിട്ടില്ല.",                 # turn 12
    "കെട്ടിട നികുതി അടയ്ക്കുന്നതിനെക്കുറിച്ച് പ്രത്യേക തീയതി വിവരങ്ങൾ നൽകിയിട്ടില്ല.",  # turn 9
    "UNSURE", "", "  unsure  ",
])
def test_meta_refusals_become_refusals(out):
    assert G.clean_answer(out) == (G.REFUSAL, True)


def test_cut_off_sentence_is_dropped():
    # turn 8 ended mid-sentence at the token limit
    out = ("K-SMART തദ്ദേശ സ്ഥാപനങ്ങളുടെ സേവനങ്ങൾക്കായുള്ള പ്ലാറ്റ്‌ഫോമാണ്. "
           "ഇതിൽ ജനന-മരണ-വിവാഹ രജിസ്ട്രേഷൻ, വ്യാപാര ലൈസൻസ് തുടങ്ങിയ സേവനങ്ങൾ")
    ans, refused = G.clean_answer(out)
    assert not refused and ans.endswith("പ്ലാറ്റ്‌ഫോമാണ്.")


def test_complete_answer_is_kept():
    out = "കാലാവധി തീരുന്നതിന് 30 ദിവസം മുൻപ് ഫീസ് അടച്ചാൽ ലൈസൻസ് സ്വയം പുതുക്കപ്പെടും."
    assert G.clean_answer(out) == (out, False)


def test_asr_garbled_certificate_still_routes_to_service():
    assert G.rule_route("ജനങ്ങൾ സർട്ടിപ്പിക്ക്ഷ്.") == "service_faq"     # turn 7


@pytest.mark.parametrize("heard, meant", [          # 10 Nov logged clips, none in ASR_FIXES
    ("കാലാവരി", "കാലാവധി"), ("പീസ്", "ഫീസ്"), ("പർമിത്തീന്റെ ഫീസ്", "പെർമിറ്റിന്റെ ഫീസ്"),
    ("ജനനരജിസ്ട്രേഷനായി", "ജനന രജിസ്ട്രേഷനായി"),       # two words heard as one
])
def test_asr_snap_fixes_unseen_mishearings(heard, meant):
    from helpdesk.asr_snap import snap
    assert snap(normalise(heard)) == normalise(meant)


@pytest.mark.parametrize("word", ["ഇന്നത്തെ", "എത്ര", "അനുകുതി", "കോഴിക്കോട്", "കാലാവസ്ഥ"])
def test_asr_snap_leaves_known_and_ambiguous_words(word):
    from helpdesk.asr_snap import snap      # അനുകുതി (അനുമതി) must not become നികുതി (tax)
    assert snap(normalise(word)) == normalise(word)


@pytest.mark.parametrize("heard, meant", [
    ("ജനറൽ രജിസ്ട്രേഷൻ ഫീസ്", "ജനന രജിസ്ട്രേഷൻ ഫീസ്"),     # real word, wrong place (3 clips, 10 Nov)
    ("പൂച്ചയെ വളർത്താൻ ലൈസൻസ്", "പൂച്ചയെ വളർത്താൻ ലൈസൻസ്"),   # cat stays cat, not dog
    ("ഇന്ന് എന്ത് ഭക്ഷണം കഴിക്കണം", "ഇന്ന് എന്ത് ഭക്ഷണം കഴിക്കണം"),  # not കല്യാണം (marriage)
    ("ബിരിയാണി എങ്ങനെ ഉണ്ടാക്കാം", "ബിരിയാണി എങ്ങനെ ഉണ്ടാക്കാം"),
    ("കഴിക്കാൻ പറ്റിയ നല്ല കഞ്ഞി ഏതാ", "കഴിക്കാൻ പറ്റിയ നല്ല കഞ്ഞി ഏതാ"),   # not പട്ടിയെ (dog)
])
def test_asr_snap_in_context(heard, meant):
    from helpdesk.asr_snap import snap
    assert snap(normalise(heard)) == normalise(meant)


def test_asr_snap_can_be_turned_off(monkeypatch):
    from helpdesk.asr_snap import snap
    monkeypatch.setenv("HELPDESK_ASR_SNAP", "0")
    assert snap("പീസ്") == "പീസ്"


@pytest.mark.parametrize("clean", [          # correctly heard eval questions the first version rewrote (11 Nov)
    "രോഗികളുടെ വീട്ടിൽ ചെന്നുള്ള പരിചരണം എങ്ങനെയാണ് ക്രമീകരിച്ചിരിക്കുന്നത്?",   # not രേഖകളുടെ
    "കെട്ടിടത്തിന്റെ ഉടമസ്ഥൻ മാറിയാൽ പേര് മാറ്റാൻ വൈകിയാൽ പിഴയുണ്ടോ?",           # not മരിച്ചാൽ / മരണ
    "ചെറിയ വീടുകളുടെ നികുതി ഒഴിവാക്കാൻ എത്ര സമയമെടുക്കും?",                        # not വിത്തുകളുടെ
    "അപേക്ഷയോടൊപ്പം എന്ത് കോർട്ട് ഫീ സ്റ്റാമ്പാണ് പതിക്കേണ്ടത്?",                    # not പുതുക്കേണ്ടത്
    "ഒ.പിയിൽ കാണിക്കാനുള്ള ഫീസ് എത്രയാണ്?",                                      # not കണക്ഷനുള്ള
    "പഠനത്തിനായി റഫറൻസ് ബുക്കുകൾ ലഭ്യമാണോ?",                                     # not പുതുക്കൽ
])
def test_asr_snap_leaves_inflected_words(clean):
    from helpdesk.asr_snap import snap
    assert snap(normalise(clean)) == normalise(clean)


def test_llm_reads_the_question_without_snap():
    from helpdesk.intent import fix_asr
    assert fix_asr("ഒരു പീസ്") != fix_asr("ഒരു പീസ്", snap=False) == normalise("ഒരു പീസ്")


def test_asr_snap_rewrites_almost_no_clean_question():
    """Clean eval questions (Gemini-written, never misheard) must come through unchanged: 36 of 1,694 were
    rewritten before the inflection guard, 2 after (വ്യാവസായിക -> വ്യവസായ, പഴമാ -> ഫലം)."""
    from pathlib import Path
    from helpdesk.asr_snap import snap
    root = Path(__file__).resolve().parent.parent / "eval"
    qs = [json.loads(l)["q"] for l in (root / "synth_questions.jsonl").open(encoding="utf-8")]
    qs += [json.loads(l)["question"] for l in (root / "questions.jsonl").open(encoding="utf-8")]
    changed = [q for q in qs if snap(normalise(q)) != normalise(q)]
    assert len(changed) <= 3, changed[:5]


def test_news_three_headlines_without_publisher(monkeypatch):
    items = "".join(f"<item><title>വാർത്ത {i} - Publisher {i}</title><link>https://x/{i}</link></item>"
                    for i in range(5)).encode()
    rss = b'<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel><title>t</title>' + items + b"</channel></rss>"

    class R:
        content = rss
        def raise_for_status(self): pass
    tools._CACHE.clear()
    monkeypatch.setattr(tools.requests, "get", _get_without_district_feeds(R()))
    out = tools.news("ഇന്നത്തെ വാർത്തകൾ")
    assert len(out["items"]) == 3 and "Publisher" not in out["text"]
    assert out["items"][0]["title"].endswith("Publisher 0")       # kept for the UI
    out = tools.news("തിരുവനന്തപുരത്തെ വാർത്തകൾ")                 # turn 5: no local headline
    assert out["text"].startswith("തിരുവനന്തപുരം ജില്ലയിലെ പ്രത്യേക വാർത്തകൾ കിട്ടിയില്ല")


# --- TTS normaliser ---------------------------------------------------------
from helpdesk.speech import number_words, to_speech


@pytest.mark.parametrize("n,words", [
    (0, "പൂജ്യം"), (10, "പത്ത്"), (21, "ഇരുപത്തിയൊന്ന്"), (22, "ഇരുപത്തിരണ്ട്"), (25, "ഇരുപത്തിയഞ്ച്"),
    (91, "തൊണ്ണൂറ്റിയൊന്ന്"), (100, "നൂറ്"), (105, "നൂറ്റിയഞ്ച്"), (125, "നൂറ്റി ഇരുപത്തിയഞ്ച്"),
    (500, "അഞ്ഞൂറ്"), (900, "തൊള്ളായിരം"), (1000, "ആയിരം"), (1500, "ആയിരത്തി അഞ്ഞൂറ്"),
    (2018, "രണ്ടായിരത്തി പതിനെട്ട്"), (3000, "മൂവായിരം"), (5000, "അയ്യായിരം"), (8000, "എണ്ണായിരം"),
    (10000, "പതിനായിരം"), (15000, "പതിനയ്യായിരം"), (25000, "ഇരുപത്തയ്യായിരം"),
    (100000, "ഒരു ലക്ഷം"), (1000000, "പത്ത് ലക്ഷം"), (15000000, "ഒരു കോടി അമ്പത് ലക്ഷം"),
])
def test_number_words(n, words):
    assert number_words(n) == words


def test_to_speech_weather_and_fees():
    assert to_speech("ഇപ്പോൾ 26 ഡിഗ്രി സെൽഷ്യസ്. മഴ സാധ്യത 100 ശതമാനം.") == \
        "ഇപ്പോൾ ഇരുപത്തിയാറ് ഡിഗ്രി സെൽഷ്യസ്. മഴ സാധ്യത നൂറ് ശതമാനം."
    assert to_speech("ഫീസ് 1,000 രൂപ. ₹500 വരെ. 98% സാധ്യത. 1.5 കോടി.") == \
        "ഫീസ് ആയിരം രൂപ. അഞ്ഞൂറ് രൂപ വരെ. തൊണ്ണൂറ്റിയെട്ട് ശതമാനം സാധ്യത. ഒന്നര കോടി."
    assert to_speech("10-25 ലക്ഷം") == "പത്ത് ലക്ഷം മുതൽ ഇരുപത്തിയഞ്ച് ലക്ഷം വരെ"


def test_to_speech_english_terms_have_no_latin_left():
    out = to_speech("K-SMART-ലെ Quick Pay വഴി; WhatsApp, email; SC/ST, BPL; eDCR, GIS.")
    assert not any("A" <= c <= "z" for c in out), out


# --- marriage fee (both charters agree) --------------------------------------
@pytest.mark.parametrize("q,expected", [
    ("കല്യാണം കഴിഞ്ഞ് പത്ത് ദിവസമായി, രജിസ്റ്റർ ചെയ്യാൻ എത്ര രൂപ?", "ആകെ 120 രൂപ"),
    ("കല്യാണം കഴിഞ്ഞ് രണ്ട് മാസമായി, രജിസ്റ്റർ ചെയ്യാൻ എത്ര രൂപ?", "ആകെ 220 രൂപ"),
    ("വിവാഹം കഴിഞ്ഞ് ആറ് വർഷമായി, രജിസ്ട്രേഷന് എത്ര ഫീസ്?", "ആകെ 370 രൂപ"),
])
def test_marriage_fee(q, expected):
    out = fees.answer(q)
    assert out and expected in out["answer"] and out["source_ids"][0] == "ch-marr-fee-01"


def test_marriage_fee_poor_household_note():
    out = fees.answer("ബിപിഎൽ കുടുംബമാണ്, വിവാഹം കഴിഞ്ഞ് 20 ദിവസം, ഫീസ് എത്ര?")
    assert "10 രൂപ മാത്രം" in out["answer"]


def test_marriage_edges():
    assert [fees.marriage_fee(d)[0] for d in (45, 46, 5 * 365, 5 * 365 + 1)] == [120, 220, 220, 370]


def test_kb_csv_builds_and_ids_are_unique():
    from helpdesk.retriever import load_kb
    ids = [d["id"] for d in load_kb()]
    assert len(ids) == len(set(ids)) and all(d["source"] for d in load_kb())


@pytest.mark.parametrize("q", ["വിധവ പെൻഷന് എന്തൊക്കെ രേഖകൾ വേണം?", "പെൻഷൻ കിട്ടാൻ എന്ത് ചെയ്യണം",
                               "പട്ടിക്ക് ലൈസൻസ് എടുക്കണോ?", "വിവരാവകാശ അപേക്ഷയ്ക്ക് എത്ര ദിവസം?"])
def test_charter_topics_route_to_service(q):
    assert G.rule_route(q) == "service_faq"


# --- regressions from the Space log (8 Nov, evening) -------------------------
def test_marriage_fee_question_without_delay_reads_schedule():
    out = fees.answer("വിവാഹ രജിസ്ട്രേഷനായിട്ടുള്ള ഫീസ് എത്രയാണ്?")
    assert out and "ആകെ 120 രൂപ" in out["answer"] and "ഫീസ് 10 രൂപ" in out["answer"]
    assert "കുറയ" not in out["answer"]


def test_marriage_how_long_is_not_a_fee_question():
    assert fees.answer("വിവാഹ സർട്ടിഫിക്കറ്റ് കിട്ടാൻ എത്ര ദിവസം എടുക്കും?") is None


@pytest.mark.parametrize("q,top", [
    ("വിവാഹ രജിസ്ട്രേഷൻ നിർബന്ധമാ.", "ch-marr-compulsory-01"),
    ("കർഷക തൊഴിലാളി പെൻഷൻ എത്രെയാണ്?", "ch-pen-amount-01"),
    ("എത്രയാണ് പെൻഷൻ തുക?", "ch-pen-amount-01"),
    ("കർഷക തൊഴിലാളി പെൻഷന് ആർക്കൊക്കെ അപേക്ഷിക്കാം?", "ch-pen-agri-01"),
    ("വീടിന് പെർമിറ്റ് കിട്ടാൻ എത്ര ദിവസം?", "ch-bp-time-01"),
    ("അക്ഷയകേന്ദ്രത്തിൽ ലഭിക്കുന്ന സർവീസുകളൊന്നൊക്കെ", "access-01"),
])
def test_log_questions_retrieve_the_right_entry(q, top):
    assert Retriever(dense=False).search(q, k=1)[0].doc["id"] == top


def test_follow_up_keeps_the_previous_topic(app):
    s = app.invoke({"query": "അതിന് എന്തൊക്കെ രേഖകൾ വേണം?",
                    "previous": "കർഷക തൊഴിലാളി പെൻഷന് ആർക്കൊക്കെ അപേക്ഷിക്കാം?", "previous_route": "service_faq"})
    assert s["route_by"] == "follow-up" and s["sources"][0]["id"] == "ch-pen-agri-01"


def test_scope_message_is_short():
    assert len(G.SCOPE) < 150


def test_district_news_uses_district_search_feed(monkeypatch):
    seen = []
    rss = (b'<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>'
           b'<item><title>\xe0\xb4\xb6\xe0\xb5\x8d\xe0\xb4\xb0\xe0\xb5\x80\xe0\xb4\x95\xe0\xb4\xbe\xe0\xb4\xb0\xe0\xb5\x8d\xe0\xb4\xaf\xe0\xb4\x82 news - P</title></item></channel></rss>')

    class R:
        content = rss
        def raise_for_status(self): pass

    def get(url, **k):
        seen.append(url)
        return R()
    tools._CACHE.clear()
    monkeypatch.setattr(tools.requests, "get", get)
    out = tools.news("തിരുവനന്തപുരത്തെ വാർത്തകൾ")
    assert "search?q=" in seen[0] and out["text"].startswith("തിരുവനന്തപുരം വാർത്തകൾ:")


# --- conversation tests (eval/conversations.jsonl, from the Space logs) -------
import json as _json
from pathlib import Path as _Path

_CONVS = [_json.loads(l) for l in (_Path(__file__).parent.parent / "eval" / "conversations.jsonl").open(encoding="utf-8") if l.strip()]


def _spell(text: str) -> str:
    from helpdesk.normalise import normalise
    from helpdesk.retriever import _CANON
    t = normalise(text, strip_punct=True)
    for a, b in _CANON:
        t = t.replace(a, b)
    return t


@pytest.mark.parametrize("conv", _CONVS, ids=[c["name"] for c in _CONVS])
def test_conversation(app, conv):
    prev = {}
    for t in conv["turns"]:
        s = app.invoke({"query": t["q"], **prev})
        where = f"turn {t['q']!r}: route={s['route']}/{s.get('route_by')} tool={s.get('tool')} " \
                f"services={s.get('services')} sources={[x.get('id') for x in s.get('sources', [])]}"
        if "route" in t:
            assert s["route"] == t["route"], where
        if "route_by" in t:
            assert s.get("route_by") == t["route_by"], where
        if "tool" in t:
            assert s.get("tool") == t["tool"], where
        if "entry" in t:
            assert s.get("sources") and s["sources"][0]["id"] == t["entry"], where
        if "title" in t:     # Tier 2: any charter's row of that service will do; spelling variants
            # (ഇൻഷ്വറൻസ്/ഇൻഷുറൻസ്, കുത്തിവെപ്പ്/കുത്തിവയ്പ്പ്) count as the same title
            assert s.get("sources") and _spell(t["title"]) in _spell(s["sources"][0]["title"]), where
        if "service" in t:
            assert t["service"] in (s.get("services") or []), where
        if "contains" in t:
            assert t["contains"] in s["answer"], where
        prev = {"previous": t["q"], "previous_route": s["route"], "previous_services": s.get("services") or [],
                "previous_tool": s.get("tool") or "", "previous_topic": s.get("topic") or ""}  # as the Space app passes it


@pytest.mark.parametrize("q,services", [
    ("വിവാഹ രജിസ്ട്രേഷനായിട്ടുള്ള ഫീസ് എത്രയാണ്?", ["marriage"]),        # not "നായ" (dog)
    ("ബിപിഎൽ, പട്ടികജാതി ആണ്, വിവാഹ ഫീസ് എത്ര?", ["marriage"]),          # not "പട്ടി" (dog)
    ("വളർത്തുമൃഗങ്ങൾക്ക് ലൈസൻസ് വേണോ", ["pet"]),
    ("നായയെ വളർത്താൻ ലൈസൻസ്", ["pet"]),
    ("ജനന സർട്ടിഫിക്കറ്റിന്റെ പകർപ്പ്", ["birth_death"]),
])
def test_service_detection_specificity(q, services):
    from helpdesk.intent import detect_services
    assert detect_services(q) == services


@pytest.mark.parametrize("q,name", [
    ("വാഗമണിലെ ഇപ്പോഴത്തെ കാലാവസ്ഥ എന്താണ്?", "വാഗമൺ"),
    ("മൂന്നാറിൽ മഴയുണ്ടോ", "മൂന്നാർ"),
    ("കോവളത്തെ കാലാവസ്ഥ", "കോവളം"),
])
def test_weather_tourist_places(q, name):
    assert tools.find_place(q)[0] == name


@pytest.mark.parametrize("q", ["കോഴിക്കോട് ജനന സർട്ടിഫിക്കറ്റ്", "എരുമേലിയിൽ വിവാഹ രജിസ്ട്രേഷൻ"])
def test_place_names_are_not_animals(q):
    from helpdesk.intent import detect_services
    assert "pet" not in detect_services(q)


def test_unknown_place_is_said_not_silently_defaulted(monkeypatch):
    class R:
        def raise_for_status(self): pass
        def json(self):
            return {"current": {"temperature_2m": 29, "weather_code": 3},
                    "daily": {"temperature_2m_max": [29], "temperature_2m_min": [24],
                              "precipitation_probability_max": [80], "precipitation_sum": [3]}}
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: R())
    assert tools.weather("പാലായിലെ കാലാവസ്ഥ എന്താണ്?")["text"].startswith("ആ സ്ഥലം")
    assert tools.weather("ഇപ്പോഴത്തെ കാലാവസ്ഥ എന്താണ്?")["text"].startswith("തിരുവനന്തപുരം")
    assert tools.weather("വാഗമണിലെ കാലാവസ്ഥ")["text"].startswith("വാഗമൺ (ഇടുക്കി ജില്ല)")


def test_video_headline_suffix_is_dropped():
    t = "പെട്രോളില്‍ എഥനോള്‍ എത്രയുണ്ട്? | ഹലോ കേരളം | Hello Keralam | Latest News | 08 - 11 - 2026 - Asianet"
    assert tools._headline(t) == "പെട്രോളില്‍ എഥനോള്‍ എത്രയുണ്ട്?"


# --- Tier 2: passages built from charters (ingest/) ---------------------------
_T2 = [
    {"id": "x-p11-r3", "tier": 2, "kind": "row", "service": "lsgd_other", "lsgi": "കടനാട്",
     "title": "പരസ്യങ്ങൾ പ്രദർശിപ്പിക്കുന്നതിന് അനുമതി",
     "text": "പരസ്യങ്ങൾ പ്രദർശിപ്പിക്കുന്നതിന് അനുമതി. അപേക്ഷിക്കേണ്ട വിധം: K-SMART അപേക്ഷ. ലഭിക്കുന്ന സമയം: ഫീസ് അടച്ച് 7 ദിവസത്തിനകം.",
     "facets": ["process", "time"], "source": "x", "questions": []},
    {"id": "y-p15-r6", "tier": 2, "kind": "row", "service": "pet", "lsgi": "കടനാട്",
     "title": "പശു, എരുമ എന്നിവയുടെ കൃത്രിമ ബീജാധാനം", "text": "പശു, എരുമ എന്നിവയുടെ കൃത്രിമ ബീജാധാനം.",
     "facets": [], "source": "y", "questions": []},
    {"id": "z-p9-c2", "tier": 2, "kind": "prose", "service": "lsgd_other", "lsgi": "തച്ചനാട്ടുകര",
     "title": "15 143 കാരയിൽപ്പുറം 3 സിനി ചന്ദ്രൻ", "text": "15 143 കാരയിൽപ്പുറം 3 സിനി ചന്ദ്രൻ പി",
     "facets": [], "source": "z", "questions": []},
]


@pytest.fixture(scope="module")
def app2():
    from helpdesk.retriever import load_kb
    return G.build_graph(Retriever(docs=[{**d, "tier": 1} for d in load_kb()] + _T2, dense=False))


def test_open_match_reaches_a_charter_row_and_names_the_panchayat(app2):
    s = app2.invoke({"query": "പരസ്യം വെക്കാൻ അനുമതി വേണോ?"})
    assert s["route_by"] == "open" and s["sources"][0]["id"] == "x-p11-r3"
    assert "കടനാട് പഞ്ചായത്തിന്റെ പൗരാവകാശ രേഖ പ്രകാരം" in s["answer"]


@pytest.mark.parametrize("q", ["സിനിമ കാണാൻ എവിടെ പോകണം", "കേരളത്തിന്റെ തലസ്ഥാനം എവിടെയാണ്?",
                               "ഇന്ന് എന്ത് ഭക്ഷണം കഴിക്കണം"])
def test_open_match_does_not_catch_general_questions(app2, q):
    assert app2.invoke({"query": q})["answer"] == G.SCOPE


def test_cow_service_is_not_declined_as_a_licence_question(app2):
    s = app2.invoke({"query": "പശുവിന് കൃത്രിമ ബീജാധാനം എവിടെ കിട്ടും?"})
    assert s.get("tool") != "missing_facet" and s["sources"][0]["id"] == "y-p15-r6"
    assert app2.invoke({"query": "പശുവിനെ വളർത്താൻ ലൈസൻസ് വേണോ?"})["tool"] == "other_animal"


def test_tier1_answers_have_no_attribution(app2):
    s = app2.invoke({"query": "വളർത്തുമൃഗങ്ങൾക്ക് ലൈസൻസ് വേണോ"})
    assert s["sources"][0].get("tier") == 1 and "പൗരാവകാശ" not in s["answer"]


def test_dense_vectors_are_cached(tmp_path, monkeypatch):
    import sys
    import types

    import numpy as np
    from helpdesk import retriever as R
    calls = []

    class Fake:
        def __init__(self, *a, **k): pass

        def encode(self, texts, **k):
            calls.append(len(texts))
            return np.ones((len(texts), 4)) / 2

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=Fake))
    monkeypatch.setattr(R, "KB_PATH", tmp_path / "kb.jsonl")
    R.Retriever(docs=_T2, dense=True)
    R.Retriever(docs=_T2, dense=True)
    assert calls == [len(_T2)] and len(list(tmp_path.glob("emb-*.npy"))) == 1


# --- ingest (no OCR needed) ----------------------------------------------------
def test_clean_fixes_ocr_and_ksmart():
    from ingest.build_docs import clean
    assert clean("Ksmart അപേക്ഷ ; ഫീസ്: ജല") == "K-SMART അപേക്ഷ; ഫീസ്: ഇല്ല"
    assert clean("ലൈസന്‍സ്‌") == "ലൈസൻസ്"


def test_row_doc_text_and_facets():
    from ingest.build_docs import row_doc
    d = row_doc({"slug": "k", "lsgi_ml": "കടനാട്", "lsgi_en": "Kadanad", "url": "u", "year": "2026"}, 11, 2,
                {"service": "പന്നി, പട്ടി എന്നിവ വളർത്തുന്നതിനുള്ള ലൈസൻസ്", "how": "Ksmart അപേക്ഷ",
                 "fee": "10 രൂപ", "time": "15 ദിവസം", "section": ""})
    assert d["service"] == "pet" and d["facets"] == ["amount", "process", "time"]
    assert "ഫീസ്: 10 രൂപ." in d["text"] and d["id"] == "k-p11-r2" and "PDF p.11" in d["source"]


def test_interleaved_table_text_is_not_a_prose_chunk():
    from ingest.build_docs import prose_docs
    src = {"slug": "t", "lsgi_ml": "ത", "lsgi_en": "T"}
    garbled = "കുടുംബാംഗങ്ങളുടെ ഫോട്ടോകൾ 1. നിശ്ചിത ഫോറത്തിലുള്ള അപേക്ഷ 1. അപേക്ഷിച്ച് 14 ദിവസം 2. തൊഴിൽ കാർഡ് നമ്പർ രേഖപ്പെടുത്തിയിരിക്കണം മഹാത്മാഗാന്ധി ദേശീയ"
    good = "മാസചന്തകൾ, സരസ് മേളകൾ, ഭക്ഷ്യ മേളകൾ, സംയോജിത വിപണന കേന്ദ്രങ്ങൾ, നാനോ മാർക്കറ്റ് എന്നിവയിലൂടെ കുടുംബശ്രീ ഉൽപന്നങ്ങൾ വിതരണം ചെയ്യുന്നു."
    assert prose_docs(src, 1, garbled) == [] and len(prose_docs(src, 1, good)) == 1


def test_pdf_links_prefers_the_charter():
    from ingest.fetch import pdf_links
    html = ('<a href="/files/budget.pdf">x</a><a href="https://cdn.web.lsgkerala.gov.in/G1/public-files/'
            '2026-07/Pouravakasha%20rekha%202026.pdf">y</a>')
    assert "Pouravakasha" in pdf_links(html, "https://x.lsgkerala.gov.in/en/node/1")[0]


def test_header_detection_and_reading_order():
    pytest.importorskip("pdfplumber")
    from ingest.ocr import _is_header, reading_order
    assert _is_header(["ക്രമ നമ്പർ", "ലഭ്യമാകുന്ന സേവനം", "അപേക്ഷിക്കുന്ന വിധം", "മറ്റ് നിബന്ധനകൾ", "ഫീസ്", "ദിവസം"])
    assert not _is_header(["5", "ജനന സർട്ടിഫിക്കറ്റ്", "K-SMART അപേക്ഷ, ഫീസ് അടച്ച രസീത്, 21 ദിവസം കഴിഞ്ഞാൽ പിഴ സഹിതം അപേക്ഷിക്കണം", "", "", ""])
    w = lambda t, x, y: {"t": t, "x": x, "y": y, "h": 20}
    assert reading_order([w("ദിവസം", 60, 11), w("15", 10, 10), w("വരെ", 10, 40)]) == "15 ദിവസം വരെ"


def test_kb_check_flags_plain_paper_against_ksmart(monkeypatch):
    from ingest import check_kb
    monkeypatch.setattr(check_kb, "page_text", lambda slug, page: "ksmartഅപേക്ഷ10രൂപ15ദിവസം")
    assert check_kb.check({"text": "വെള്ളക്കടലാസിൽ അപേക്ഷ. ഫീസ് 10 രൂപ."}, "k", 11) == \
        ["says plain-paper application; page says k-smart"]
    assert check_kb.check({"text": "K-SMART അപേക്ഷ. ഫീസ് 10 രൂപ, 15 ദിവസം."}, "k", 11) == []


def test_fetch_respects_robots_txt(monkeypatch):
    from ingest import fetch

    class R:
        status_code = 200
        text = "User-agent: *\nDisallow: /private/\n"
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: R())
    fetch._ROBOTS.clear()
    assert fetch.allowed("https://x.lsgkerala.gov.in/en/node/1")
    assert not fetch.allowed("https://x.lsgkerala.gov.in/private/a.pdf")
    assert fetch.fetch_one({"slug": "zz-test", "url": "https://x.lsgkerala.gov.in/private/a.pdf"}, {}).startswith("SKIPPED")


# --- website crawl (ingest.web) -------------------------------------------------
_HTML = {
    "https://lsg.example/ml/services": '<html><head><title>സേവനങ്ങൾ | LSGD</title></head><body><nav>മെനു</nav>'
        '<article><p>' + "കെട്ടിട നിർമ്മാണ പെർമിറ്റിന് K-SMART വഴി അപേക്ഷിക്കാം. " * 6 + '</p></article>'
        '<a href="/ml/services/permit">പെർമിറ്റ്</a><a href="/ml/news/1">x</a>'
        '<a href="/search/node?k=1">s</a><a href="/files/go.pdf">GO</a></body></html>',
    "https://lsg.example/ml/services/permit": '<html><head><title>പെർമിറ്റ്</title></head><body><article><p>'
        + "പെർമിറ്റ് ഫീസ് 100 ചതുരശ്ര മീറ്റർ വരെ 150 രൂപ. " * 6 + '</p></article></body></html>',
}


class _Resp:
    def __init__(self, url):
        self.url, self.text, self.headers = url, _HTML[url], {"Content-Type": "text/html"}

    def raise_for_status(self): pass


def test_crawl_stays_in_section_obeys_robots_and_lists_pdfs(monkeypatch):
    pytest.importorskip("trafilatura")
    from ingest import web
    monkeypatch.setattr(web, "allowed", lambda u: "/search/" not in u)
    got = []
    pages, pdfs = web.crawl({"url": "https://lsg.example/ml/services", "path_prefix": "/ml/services"}, 10,
                            get=lambda u, **k: got.append(u) or _Resp(u), sleep=lambda s: None)
    assert got == ["https://lsg.example/ml/services", "https://lsg.example/ml/services/permit"]
    assert [p["title"] for p in pages] == ["സേവനങ്ങൾ", "പെർമിറ്റ്"] and "മെനു" not in pages[0]["text"]
    assert pdfs == ["https://lsg.example/files/go.pdf"]


def test_web_diff_reports_changed_numbers():
    from ingest.web import diff
    old = {"u": {"sha256": "a", "text": "ഫീസ് 150 രൂപ"}}
    lines = diff(old, [{"url": "u", "title": "പെർമിറ്റ്", "sha256": "b", "text": "ഫീസ് 200 രൂപ"},
                       {"url": "v", "title": "പുതിയത്", "sha256": "c", "text": "x"}])
    assert lines[0].startswith("- CHANGED u") and "removed: 150" in lines[0] and "added: 200" in lines[0]
    assert lines[1].startswith("- NEW v")


def test_web_docs_carry_page_title_and_site_attribution():
    from ingest.build_docs import web_docs
    pg = {"page": 1, "url": "https://lsg.example/ml/permit", "title": "കെട്ടിട പെർമിറ്റ്",
          "text": "കെട്ടിട നിർമ്മാണ പെർമിറ്റിന് K-SMART വഴി അപേക്ഷിക്കണം. " * 4 + "\nThis paragraph is in English only. " * 4}
    docs, english = web_docs("egov", pg)
    assert docs[0]["kind"] == "web" and docs[0]["title"] == "കെട്ടിട പെർമിറ്റ്" and docs[0]["site"] == "LSGD"
    assert english == 4                      # four English paragraphs
    assert G.attribution([{"tier": 2, "site": "LSGD"}]) == " (തദ്ദേശ സ്വയംഭരണ വകുപ്പിന്റെ വെബ്സൈറ്റ് പ്രകാരം.)"


def test_refetched_changed_charter_is_marked_for_reextraction(tmp_path, monkeypatch):
    from ingest import fetch

    class R:
        status_code, content, text, url = 200, b"%PDF-new", "", "u.pdf"
        def raise_for_status(self): pass
    monkeypatch.setattr(fetch, "RAW", tmp_path)
    monkeypatch.setattr(fetch, "HERE", tmp_path)
    monkeypatch.setattr(fetch, "allowed", lambda u: True)
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: R())
    (tmp_path / "pages").mkdir()
    (tmp_path / "pages" / "k.jsonl").write_text("{}")
    manifest = {"k": {"sha256": "old"}}
    assert fetch.fetch_one({"slug": "k", "url": "https://x/a.pdf"}, manifest, force=True).startswith("CHANGED")
    assert not (tmp_path / "pages" / "k.jsonl").exists()
    assert fetch.fetch_one({"slug": "k", "url": "https://x/a.pdf"}, manifest, force=True) == "unchanged"


# --- ingest.canon (task B) with a fake model ---------------------------------------
def _fake_model(prompt: str) -> str:
    import re as _re
    if "<group " in prompt:
        out = []
        for gid, body in _re.findall(r'<group id="(\w+)"[^>]*>(.*?)</group>', prompt, _re.S):
            rows = _re.findall(r'<row id="([^"]+)"[^>]*>(.*?)</row>', body, _re.S)
            title = rows[0][1].split(".")[0]
            junk = "ഡോക്ടറുടെ" in title
            out.append({"group_id": gid, "is_service": not junk, "same_service": True, "service": "pet",
                        "title_ml": title, "synonyms_ml": ["നായ ലൈസൻസ്"],
                        "summary_ml": "K-SMART വഴി അപേക്ഷിക്കണം. ഫീസ് 999 രൂപ.",       # 999: not in the rows
                        "variants": [{"row_id": rid, "fee": "10 രൂപ", "time": "15 ദിവസം"} for rid, _ in rows],
                        "facets": ["amount"], "questions_ml": [f"{title} ചോദ്യം {i}" for i in range(8)]})
        return json.dumps(out, ensure_ascii=False)
    if "<entry " in prompt:
        ids = _re.findall(r'<entry id="([^"]+)"', prompt)
        return "```json\n" + json.dumps([{"id": i, "questions_ml": [f"{i} q{k}" for k in range(6)]} for i in ids]) + "\n```"
    return json.dumps(["കേരളത്തിന്റെ തലസ്ഥാനം ഏതാണ്?", "മീൻ കറി എങ്ങനെ വെക്കാം?", "നിന്റെ പേരെന്താ?"],
                      ensure_ascii=False)


def test_canon_pipeline_with_fake_model(tmp_path, monkeypatch):
    from ingest import canon
    rows = [{"id": f"p{i}-r0", "kind": "row", "service": "pet", "lsgi": lsgi, "source": "s",
             "title": "പട്ടി വളർത്തുന്നതിനുള്ള ലൈസൻസ്", "facets": ["amount"],
             "text": "പട്ടി വളർത്തുന്നതിനുള്ള ലൈസൻസ്. ഫീസ്: 10 രൂപ. ലഭിക്കുന്ന സമയം: 15 ദിവസം."}
            for i, lsgi in enumerate(["കടനാട്", "കൊപ്പം"])]
    rows.append({"id": "x-r9", "kind": "row", "service": "health", "lsgi": "കടനാട്", "source": "s",
                 "title": "ഡോക്ടറുടെ മുന്നിൽ ഹാജരാകണം", "text": "ഡോക്ടറുടെ മുന്നിൽ ഹാജരാകണം.", "facets": []})
    (tmp_path / "docs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    kb = [{"id": "ch-pet-01", "service": "pet", "title": "വളർത്തുമൃഗ ലൈസൻസ്", "text": "ഫീസ് 10 രൂപ."}]
    (tmp_path / "kb.jsonl").write_text(json.dumps(kb[0], ensure_ascii=False) + "\n")
    for name, f in [("DOCS", "docs.jsonl"), ("KB", "kb.jsonl"), ("CANON", "canon.jsonl"),
                    ("SYNTH", "synth.jsonl"), ("CACHE", "cache.jsonl")]:
        monkeypatch.setattr(canon, name, tmp_path / f)
    calls = []
    stats = canon.run(lambda p: calls.append(p) or _fake_model(p), batch=8)
    assert len(calls) == 3                                   # groups, tier-1, out-of-scope
    docs = [json.loads(l) for l in (tmp_path / "canon.jsonl").open()]
    assert len(docs) == 1 and stats["not_a_service"] == 1    # the doctor row is not a service
    d = docs[0]
    assert d["members"] == ["p0-r0", "p1-r0"] and d["lsgis"] == ["കടനാട്", "കൊപ്പം"]
    assert "999" not in d["text"] and "കടനാട്: ഫീസ്: 10 രൂപ" in d["text"]   # unsupported number dropped
    synth = [json.loads(l) for l in (tmp_path / "synth.jsonl").open()]
    held = {s["q"] for s in synth if s["kind"] == "canon"}
    assert held and not held & set(d["questions"])           # eval questions are never indexed
    assert {s["kind"] for s in synth} == {"canon", "tier1", "out_of_scope"}
    assert len(canon.run(lambda p: calls.append(p) or _fake_model(p))) and len(calls) == 3   # all cached


def test_canon_check_group_odd_shapes():
    """JSON mode guarantees valid JSON, not field types: a list summary (seen on Kaggle, 10 Nov),
    None questions or a string variant must not crash write_outputs."""
    from ingest import canon
    g = {"id": "g1", "service": "pet", "rows": [{"id": "r1", "lsgi": "കടനാട്",
         "text": "പട്ടി ലൈസൻസ്. ഫീസ്: 10 രൂപ. സമയം: 15 ദിവസം."}]}
    c = canon.check_group({"is_service": True, "title_ml": ["പട്ടി ലൈസൻസ്"],
                           "summary_ml": ["K-SMART വഴി അപേക്ഷിക്കണം.", "ഫീസ് 10 രൂപ."],
                           "synonyms_ml": "നായ ലൈസൻസ്", "questions_ml": None, "facets": "amount",
                           "variants": ["bad", {"row_id": "r1", "fee": ["10 രൂപ"], "time": None}]}, g)
    assert c["title"] == "പട്ടി ലൈസൻസ്" and c["summary"].startswith("K-SMART") and "10" in c["summary"]
    assert c["synonyms"] == ["നായ ലൈസൻസ്"] and c["questions"] == [] and c["facets"] == ["amount"]
    assert c["variants"] == [{"row_id": "r1", "lsgi": "കടനാട്", "fee": "10 രൂപ", "time": ""}]
    assert canon.check_group({"is_service": True, "title_ml": "x", "summary_ml": ["ഫീസ് 999"]}, g)["summary"] == ""
    assert canon.check_group({"is_service": False, "title_ml": "x"}, g) is None
    assert not canon.numbers_ok(["99"], "10") and canon.numbers_ok({"a": "10"}, "10")


def test_canon_question_filters():
    """Other-script slips are dropped; questions that lean on an earlier turn are not held out."""
    from ingest import canon
    assert canon._OTHER_SCRIPT.search("എത്ര ദിവസமாகும்?") and not canon._OTHER_SCRIPT.search("എത്ര ദിവസം?")
    assert canon._DEICTIC.search("ഈ സർട്ടിഫിക്കറ്റിന് ഏതൊക്കെ രേഖകൾ?") and canon._DEICTIC.search("ഇതിന് എത്ര ഫീസ്?")
    assert not canon._DEICTIC.search("അത്യാവശ്യമായി മരണ സർട്ടിഫിക്കറ്റ് വേണം")
    assert not canon._DEICTIC.search("മരണ സർട്ടിഫിക്കറ്റിന് എത്ര രൂപ?")


def test_canon_keeps_replaced_rows_vocabulary(tmp_path, monkeypatch):
    """A canonical passage replaces its rows in the index, not in the follow-up vocabulary."""
    from helpdesk import retriever as R
    rows = [{"id": "r1", "kind": "row", "service": "pension", "title": "വിധവ പെൻഷൻ", "source": "s",
             "text": "വിവാഹ മോചിതർക്കും അപേക്ഷിക്കാം."}]
    canon = [{"id": "canon-g1", "kind": "canon", "service": "pension", "title": "വിധവ പെൻഷൻ", "source": "s",
              "text": "വിധവ പെൻഷൻ.", "members": ["r1"]}]
    for name, f, data in [("DOCS_PATH", "docs.jsonl", rows), ("CANON_PATH", "canon.jsonl", canon)]:
        (tmp_path / f).write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in data))
        monkeypatch.setattr(R, name, tmp_path / f)
    monkeypatch.setenv("HELPDESK_CANON", "1")
    docs = R.load_all()
    assert "r1" not in {d["id"] for d in docs}
    assert R.Retriever(docs=docs, dense=False).fits_services("വിവാഹ മോചിതർക്ക് അപേക്ഷിക്കാനോ?", ["pension"])


@pytest.mark.parametrize("q,fee", [
    ("ജനനരജിസ്ട്രേഷനായി ഉള്ള ഫീസ് എത്രയാണ് ഇരുപത്തൊന്നു ദിവസത്തിനു ശേഷം.", 2),   # 10 Nov Space log
    ("ജനന രജിസ്ട്രേഷൻ 21 ദിവസത്തിനുശേഷം ഫീസ്", 2),
    ("ജനന രജിസ്ട്രേഷൻ 21 ദിവസം കഴിഞ്ഞാൽ ഫീസ് എത്ര", 2),
    ("ജനന രജിസ്ട്രേഷൻ 21 ദിവസത്തിനുള്ളിൽ ഫീസ്", 0),
    ("ജനന രജിസ്ട്രേഷൻ 10 ദിവസം കഴിഞ്ഞു ഫീസ്", 0),
    ("മരണ രജിസ്ട്രേഷൻ 30 ദിവസത്തിനു ശേഷം ഫീസ് എത്ര", 5),
])
def test_fee_after_n_days_is_past_the_boundary(q, fee):
    from helpdesk import fees
    a = fees.answer(q)["answer"]
    assert ("ഫീസ് ഇല്ല" in a) if fee == 0 else (f"ഫീസ് {fee} രൂപ" in a), a


@pytest.mark.parametrize("word,n", [
    ("ഇരുപത്", 20), ("ഇരുപത്തൊന്ന്", 21), ("ഇരുപത്തിയൊന്ന്", 21), ("ഇരുപത്തിമൂന്ന്", 23),
    ("ഇരുപത്തഞ്ച്", 25), ("ഇരുപത്തേഴ്", 27), ("ഇരുപത്തെട്ട്", 28), ("മുപ്പത്തിയൊന്ന്", 31),
    ("നാൽപ്പത്തിയഞ്ച്", 45), ("നാല്", 4),
    ("ഇരുപത്തെന്ന", None), ("ഇരുപത്തുനിവസം", None),           # ASR slips (10 Nov log): not 20
])
def test_fee_number_words(word, n):
    from helpdesk import fees
    from helpdesk.normalise import normalise
    assert fees._num(normalise(word)) == n


def test_fee_unreadable_delay_computes_nothing():
    """'ഇരുപത്തെന്ന ദിവസത്തിനു ശേഷം' was answered 'within 21 days, no fee' (10 Nov Space log)."""
    from helpdesk import fees
    assert fees.answer("ജനനര്ജിസ്ട്രേഷന് പീസ് ഇരുപത്തെന്ന ദിവസത്തിനു ശേഷം എത്രയാണ്?") is None


# --- 100-question answer eval (eval/answer_eval.jsonl, 11 Nov) --------------
@pytest.mark.parametrize("q", [
    "ജനിച്ചിട്ട് ഒരു വർഷം കഴിഞ്ഞ കുട്ടിയുടെ ജനനം രജിസ്റ്റർ ചെയ്യാൻ ആരുടെയൊക്കെ അനുമതി വേണം?",     # hd021: who approves
    "വിവാഹം കഴിഞ്ഞ് ഒരു വർഷത്തിന് ശേഷം അപേക്ഷിച്ചാൽ സർട്ടിഫിക്കറ്റ് കിട്ടാൻ എത്ര ദിവസമെടുക്കും?",  # hd028: how long
])
def test_fee_calculator_leaves_non_fee_questions(q):
    """A delay plus 'register' is not a fee question: these got ₹10 and ₹220."""
    assert fees.answer(q) is None


@pytest.mark.parametrize("q", [
    "വാടക കെട്ടിടത്തിൽ ഒരു കട തുടങ്ങാൻ ലൈസൻസിന് അപേക്ഷിക്കുമ്പോൾ കെട്ടിട ഉടമയുടെ സമ്മതപത്രം നിർബന്ധമാണോ?",  # hd006
    "വാടക വീട്ടിൽ താമസിക്കുന്നവർക്ക് റെസിഡൻസ് സർട്ടിഫിക്കറ്റ് കിട്ടാൻ വാടകക്കരാർ നിർബന്ധമാണോ?",  # hd033
])
def test_compulsory_document_is_a_documents_question(app, q):
    """Was declined as an unknown 'compulsory' fact; the documents entry lists the document."""
    from helpdesk.intent import detect_facet
    assert detect_facet(q) == "documents"
    s = app.invoke({"query": q})
    assert s.get("tool") != "missing_facet" and not s.get("refused"), s


@pytest.mark.parametrize("q", ["വിവാഹ രജിസ്ട്രേഷൻ നിർബന്ധമാണോ", "പട്ടിയെ വളർത്താൻ ലൈസൻസ് വേണോ"])
def test_service_itself_compulsory_keeps_the_facet(q):
    from helpdesk.intent import detect_facet
    assert detect_facet(q) == "compulsory"


@pytest.mark.parametrize("q,oos", [
    ("എനിക്ക് രണ്ട് ദിവസമായിട്ട് ഭയങ്കര തലവേദനയും പനിയും ആണ്, എന്ത് മരുന്നാണ് കഴിക്കേണ്ടത്?", True),  # hd086
    ("പനിക്ക് ഏത് ഗുളിക കഴിക്കണം", True),
    ("കമ്പനി തുടങ്ങാൻ ട്രേഡ് ലൈസൻസ് വേണോ", False),              # കമ്പനി contains പനി
    ("സെക്രട്ടറിയുടെ ചുമതല ആർക്കാണ്", False),                   # ചുമതല starts like ചുമ
    ("ഹോട്ടൽ ജോലിക്കാർക്ക് ഹെൽത്ത് കാർഡ് എവിടെ കിട്ടും", False),
    ("പാലിയേറ്റീവ് കെയർ രോഗികൾക്ക് മരുന്ന് വിതരണം എപ്പോൾ", False),
])
def test_medical_advice_is_out_of_scope(q, oos):
    from helpdesk.intent import out_of_scope
    assert out_of_scope(q) is oos


def test_other_state_service_is_out_of_scope(app):
    """A Tamil Nadu birth certificate got Kerala's K-SMART steps (hd089)."""
    s = app.invoke({"query": "തമിഴ്നാട്ടിലെ ഒരു പഞ്ചായത്തിൽ നിന്ന് ജനന സർട്ടിഫിക്കറ്റ് എടുക്കാൻ എന്താണ് വഴി?"})
    assert s["route"] == "other" and s.get("refused"), s
