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
