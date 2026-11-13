"""ingest/kb_auto.py: automatically built Tier-1-schema entries (no network, no LLM)."""
import json
import re

import pytest

from ingest import kb_auto as A

ENTRIES = [json.loads(l) for l in A.OUT.open(encoding="utf-8")] if A.OUT.exists() else []
ROWS = {d["id"]: d for d in A._load(A.DOCS)}


@pytest.mark.parametrize("cell,key", [
    ("ഇല്ല", "ഇല്ല"), ("ഫീസ് ഇല്ല", "ഇല്ല"), ("ഫീസില്ല", "ഇല്ല"), ("10 രൂപ", "rs:10"),
    ("തെരച്ചിൽ ഫീ 2/- രൂപാ, സർട്ടിഫിക്കറ്റ് ഫീ 5/-", "rs:2,5"), ("ഗ്രാമ പഞ്ചായത്ത്", None),
])
def test_fee_key(cell, key):
    assert A.fee_key(cell) == key


@pytest.mark.parametrize("cell,key", [
    ("അന്നേ ദിവസം", "അതേ ദിവസം"), ("അതേ ദിവസം", "അതേ ദിവസം"), ("7 ദിവസം", "7 ദിവസം"),
    ("3 പ്രവൃത്തി ദിവസം", "3 പ്രവൃത്തി ദിവസം"), ("1 മാസം", "1 മാസം"), ("ഉടൻ", "അതേ ദിവസം"), ("", None),
])
def test_time_key(cell, key):
    assert A.time_key(cell) == key


def test_consensus_needs_two_panchayats():
    assert A.consensus([("a", "7 ദിവസം"), ("b", "7 ദിവസം"), ("c", "3 ദിവസം")], A.time_key)[:2] == ("7 ദിവസം", 2)
    assert A.consensus([("a", "7 ദിവസം"), ("b", "3 ദിവസം")], A.time_key)[0] is None
    assert A.consensus([("a", "7 ദിവസം"), ("a", "7 ദിവസം")], A.time_key)[0] is None   # one vote per panchayat


def test_dedupe_is_asymmetric():
    curated = [{"id": "bp", "service": "building_permit", "title": "കെട്ടിട നിർമ്മാണ പെർമിറ്റിന് വേണ്ട രേഖകൾ",
                "questions": [], "text": "ആധാരം, സൈറ്റ് പ്ലാൻ"}]
    assert A.duplicate_of({"service": "building_permit", "title": "കിണർ നിർമ്മാണ പെർമിറ്റ്"}, curated) is None
    assert A.duplicate_of({"service": "building_permit", "title": "കെട്ടിട നിർമ്മാണ പെർമിറ്റ്"}, curated) == "bp"


def test_official_domains():
    assert A.official("https://cdn.web.lsgkerala.gov.in/x.pdf") and A.official("https://ksmart.lsgkerala.gov.in/")
    assert not A.official("https://www.example.com/lsgkerala.gov.in")


@pytest.mark.skipif(not ENTRIES, reason="data/kb_auto.jsonl not built")
def test_entries_have_the_kb_schema_and_provenance():
    curated = {json.loads(l)["id"] for l in A.KB.open(encoding="utf-8")}
    ids = [e["id"] for e in ENTRIES]
    assert len(ids) == len(set(ids)) and not set(ids) & curated
    assert len(ENTRIES) >= 50
    for e in ENTRIES:
        assert {"id", "service", "title", "text", "questions", "source", "verify", "checked", "facets"} <= set(e)
        assert isinstance(e["verify"], bool) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", e["checked"]), e["id"]
        assert re.search(r"https?://", e["source"]) and len(e["lsgis"]) >= A.MIN_LSGIS
        if not e["verify"]:
            assert all(A.official(u) for u in A._URL.findall(e["source"])), e["id"]


@pytest.mark.skipif(not ENTRIES, reason="data/kb_auto.jsonl not built")
def test_every_number_is_in_the_member_rows():
    """Grounding: no number in an auto entry that its charter rows do not contain (counts of
    panchayats, which the builder adds, are excluded)."""
    for e in ENTRIES:
        text = re.sub(r"\(\d+ പഞ്ചായത്തുകളുടെ പൗരാവകാശ രേഖ പ്രകാരം\)", "", e["text"])
        src = " ".join(ROWS[m]["text"] for m in e["members"] if m in ROWS).replace(",", "")
        for n in A._NUM.findall(text):
            assert n.replace(",", "") in src, (e["id"], n)


@pytest.mark.skipif(not ENTRIES, reason="data/kb_auto.jsonl not built")
def test_loader_replaces_member_rows(monkeypatch):
    from helpdesk import retriever as R
    monkeypatch.setenv("HELPDESK_KB_AUTO", "1")
    monkeypatch.setenv("HELPDESK_CANON", "1")
    docs = R.load_all()
    ids = {d["id"] for d in docs}
    e = ENTRIES[0]
    assert e["id"] in ids and not set(e["members"]) & ids and "canon-" + e["id"][5:] not in ids
    assert next(d for d in docs if d["id"] == e["id"])["tier"] == 1
    monkeypatch.setenv("HELPDESK_KB_AUTO", "0")
    assert e["id"] not in {d["id"] for d in R.load_all()}


def test_robots_rfc9309(monkeypatch):
    """4xx robots.txt allows; 5xx or a network error disallows everything."""
    import requests
    from ingest import fetch

    class R:
        def __init__(self, code, text=""):
            self.status_code, self.text = code, text

    for code, text, url, ok in [(404, "", "https://a.example/x.pdf", True),
                                (503, "", "https://b.example/x.pdf", False),
                                (200, "User-agent: *\nDisallow: /private/", "https://c.example/private/x", False),
                                (200, "User-agent: *\nDisallow: /private/", "https://c.example/public/x", True)]:
        monkeypatch.setattr(fetch.requests, "get", lambda *a, _r=R(code, text), **k: _r)
        fetch._ROBOTS.clear()
        assert fetch.allowed(url) is ok, (code, url)

    def boom(*a, **k):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(fetch.requests, "get", boom)
    fetch._ROBOTS.clear()
    assert fetch.allowed("https://d.example/x.pdf") is False
