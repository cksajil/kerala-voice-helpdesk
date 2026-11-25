"""FastAPI service (helpdesk/api.py), without a model: HELPDESK_LLM=none or a fake Llama."""
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from helpdesk import api, llm  # noqa: E402


def wait_ready(client, timeout=120):
    t = time.time()
    while time.time() - t < timeout:
        h = client.get("/health").json()
        if h["status"] in ("ok", "error"):
            return h
        time.sleep(0.2)
    raise TimeoutError(h)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("HELPDESK_LLM", "none")
    monkeypatch.setenv("HELPDESK_DENSE", "0")
    monkeypatch.setattr(api, "BACKEND", "none")
    monkeypatch.setattr(api, "service", api.Service())
    with TestClient(api.app) as c:
        yield c


def test_health_and_service_answer(client):
    h = wait_ready(client)
    assert h["status"] == "ok" and h["backend"] == "none" and h["passages"] > 50
    r = client.post("/ask", json={"question": "ജനന സർട്ടിഫിക്കറ്റ് എങ്ങനെ കിട്ടും?"})
    assert r.status_code == 200
    body = r.json()
    assert body["route"] == "service_faq" and not body["refused"] and body["sources"]
    assert body["timings"]["total"] > 0 and "router" in body["timings"]
    assert body["context"]["previous"] == "ജനന സർട്ടിഫിക്കറ്റ് എങ്ങനെ കിട്ടും?"


def test_context_carries_a_follow_up(client):
    wait_ready(client)
    first = client.post("/ask", json={"question": "വിവാഹ രജിസ്ട്രേഷന് എന്തൊക്കെ രേഖകൾ വേണം?"}).json()
    alone = client.post("/ask", json={"question": "എത്ര ദിവസം എടുക്കും?"}).json()
    follow = client.post("/ask", json={"question": "എത്ര ദിവസം എടുക്കും?", "context": first["context"]}).json()
    assert alone["route"] != "service_faq"                 # nothing to follow: "which service?"
    assert follow["route"] == "service_faq" and follow["route_by"] == "follow-up"


def test_out_of_scope_and_validation(client):
    wait_ready(client)
    r = client.post("/ask", json={"question": "ഒരു തമാശ പറയൂ"}).json()
    assert r["route"] == "other"
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x" * (api.MAX_QUESTION + 1)}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422


def test_warm_up_prefills_each_curated_passage(monkeypatch):
    calls = []

    class FakeLlama:
        def create_chat_completion(self, messages, max_tokens=0, **kw):
            calls.append((messages[1]["content"], max_tokens))
            return {"choices": [{"message": {"content": "ശരി"}}]}

    monkeypatch.setenv("HELPDESK_DENSE", "0")
    monkeypatch.setenv("HELPDESK_KB_AUTO", "1")
    monkeypatch.delenv("HELPDESK_LLM", raising=False)
    monkeypatch.setattr(api, "BACKEND", "http")             # build the graph, load no model
    monkeypatch.setattr(api, "WARMUP", "kb")
    s = api.Service()
    s.load()
    monkeypatch.setattr(api, "BACKEND", "inproc")
    llm.use_llama_cpp(FakeLlama())
    try:
        s.warm_up()
    finally:
        llm.use_llama_cpp(None)
    curated = [d for d in s.retriever.docs if d.get("tier") == 1 and d.get("kind") != "kb_auto"]
    prefills = [c for c in calls if c[1] == 1]
    assert s.status == "ok" and s.warm["done"] == s.warm["total"] == len(curated) == len(prefills)
    assert all(d["title"] in c[0] for d, c in zip(curated, prefills))
