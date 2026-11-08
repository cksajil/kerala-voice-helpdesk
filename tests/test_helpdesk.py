"""Offline tests: no LLM, no network (tools and LLM are monkeypatched)."""
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
    s = app.invoke({"query": "പാസ്പോർട്ട് പുതുക്കാൻ എത്ര ഫീസ് ആകും?"})
    assert s["refused"] and s["answer"] == G.REFUSAL


def test_llm_classifier_used_when_no_rule(app, monkeypatch):
    monkeypatch.setattr(llm, "enabled", lambda: True)
    monkeypatch.setattr(llm, "chat", lambda *a, **k: "other")
    s = app.invoke({"query": "നിന്റെ പേരെന്താ?"})
    assert s["route"] == "other" and s["route_by"] == "llm" and s["answer"] == G.SCOPE


# --- news parsing -----------------------------------------------------------
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
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: R())
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


def test_news_three_headlines_without_publisher(monkeypatch):
    items = "".join(f"<item><title>വാർത്ത {i} - Publisher {i}</title><link>https://x/{i}</link></item>"
                    for i in range(5)).encode()
    rss = b'<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel><title>t</title>' + items + b"</channel></rss>"

    class R:
        content = rss
        def raise_for_status(self): pass
    tools._CACHE.clear()
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: R())
    out = tools.news("ഇന്നത്തെ വാർത്തകൾ")
    assert len(out["items"]) == 3 and "Publisher" not in out["text"]
    assert out["items"][0]["title"].endswith("Publisher 0")       # kept for the UI
    out = tools.news("തിരുവനന്തപുരത്തെ വാർത്തകൾ")                 # turn 5: no local headline
    assert out["text"].startswith("തിരുവനന്തപുരം ജില്ലയിലെ പ്രത്യേക വാർത്തകൾ കിട്ടിയില്ല")
