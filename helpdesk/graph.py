"""LangGraph agent for the Malayalam voice helpdesk (reply stage).

    router ─► weather ───────────────────────► answer ─► END
           ├► news ──────────────────────────► answer
           ├► calc ─ fee computed in code ───────────────► END
           │    └─ not a fee question ─► retrieve ► answer
           └► other (scope message) ─────────────────────► END

Small models (Gemma 4 E2B) are unreliable at free-form tool calling, so the
graph makes the routing explicit: keyword rules first, LLM classification
only when no rule fires. Tool numbers go out through templates; the LLM only
writes service answers, and only from retrieved passages.
"""
from __future__ import annotations

import os
import re
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from . import fees, llm, tools
from .normalise import normalise
from .retriever import Retriever, mentions_service

Route = Literal["weather", "news", "service_faq", "other"]
CTX_K = int(os.getenv("HELPDESK_CTX_K", "2"))           # passages sent to the LLM
ANSWER_TOKENS = int(os.getenv("HELPDESK_ANSWER_TOKENS", "100"))   # ~0.34 s per token on a 2-core CPU

REFUSAL = ("ക്ഷമിക്കണം, ഇതിനെക്കുറിച്ച് എനിക്ക് ഉറപ്പുള്ള വിവരം ഇല്ല. "
           "നിങ്ങളുടെ പഞ്ചായത്ത് അല്ലെങ്കിൽ നഗരസഭ ഓഫീസിലോ അക്ഷയ കേന്ദ്രത്തിലോ ചോദിക്കൂ.")
SCOPE = ("ഞാൻ പഞ്ചായത്ത്, നഗരസഭ സേവനങ്ങൾ, കാലാവസ്ഥ, വാർത്ത എന്നിവയിൽ സഹായിക്കാം. "
         "ഉദാഹരണത്തിന്, വിവാഹ രജിസ്ട്രേഷന് എന്തൊക്കെ രേഖകൾ വേണം എന്ന് ചോദിക്കാം.")
TOOL_FAIL = "ക്ഷമിക്കണം, ഇപ്പോൾ ആ വിവരം ലഭിക്കുന്നില്ല. കുറച്ചു കഴിഞ്ഞ് വീണ്ടും ചോദിക്കൂ."

# Stems, not full words: Malayalam inflects heavily (മഴ / മഴയുണ്ടോ, വാർത്ത / വാർത്തകൾ).
RULES: list[tuple[Route, list[str]]] = [
    ("weather", ["കാലാവസ്ഥ", "മഴ", "ചൂട്", "താപനില", "ഡിഗ്രി", "weather", "rain", "temperature"]),
    ("news", ["വാർത്ത", "news", "headline", "സംഭവ"]),
    ("service_faq", ["സർട്ടി", "ലൈസൻസ", "ഫീസ", "രജിസ്ട്രേഷ", "രജിസ്റ്റ", "നികുതി", "ടാക്സ",
                     "പെർമിറ്റ", "അനുമതി", "പരാതി", "ജനന", "മരണ", "വിവാഹ", "കല്യാണ", "അക്ഷയ",
                     "ksmart", "കെ-സ്മാർട്ട", "കെ സ്മാർട്ട", "പഞ്ചായത്ത", "നഗരസഭ", "മുനിസിപ്പാലിറ്റി",
                     "certificate", "licence", "license", "permit", "tax", "fee", "trade", "ട്രേഡ",
                     # topics added from the panchayat citizen charters (8 Nov)
                     "പെൻഷ", "pension", "വേതനം", "വിവരാവകാശ", "ആർ.ടി.ഐ", "rti", "പട്ടി", "നായ",
                     "ഹെൽത്ത് കാർഡ", "മെഡിക്കൽ സർട്ടിഫിക്കറ്റ", "ദത്തെടു", "വീട്ടുകര", "ഉടമസ്ഥ"]),
]
CLASSIFY_PROMPT = (
    "Classify the user's Malayalam question into exactly one label: weather, news, service_faq, other. "
    "service_faq = Kerala local-body services (certificates, licences, property tax, building permits, "
    "complaints, K-SMART, Akshaya). Reply with the label only.")
ANSWER_PROMPT = (
    "നീ കേരളത്തിലെ തദ്ദേശ സ്ഥാപന സേവനങ്ങളെക്കുറിച്ചുള്ള ഒരു വോയ്സ് ഹെൽപ്ഡെസ്ക് ആണ്. "
    "താഴെ കൊടുത്ത വിവരങ്ങൾ മാത്രം ഉപയോഗിച്ച് ലളിതമായ സംസാര മലയാളത്തിൽ 1-2 ചെറിയ വാക്യങ്ങളിൽ ഉത്തരം പറയുക. "
    "തുകകളും ദിവസങ്ങളും പേരുകളും വിവരങ്ങളിൽ ഉള്ളതുപോലെ തന്നെ പറയുക. Markdown, ലിസ്റ്റ്, ലിങ്ക് എന്നിവ വേണ്ട. "
    "തുകകൾ കൂട്ടുകയോ കുറയ്ക്കുകയോ ചെയ്യരുത്. "
    "വിവരങ്ങൾ ചോദ്യത്തിലെ അതേ സേവനത്തെക്കുറിച്ചല്ലെങ്കിലോ, ചോദിച്ച കാര്യം (ഉദാഹരണത്തിന് തുക, തീയതി, "
    "നിർബന്ധമാണോ എന്ന്) വിവരങ്ങളിൽ നേരിട്ട് ഇല്ലെങ്കിലോ UNSURE എന്ന് മാത്രം എഴുതുക.")

# Gemma sometimes refuses in its own words ("...വിവരങ്ങൾ നൽകിയിട്ടില്ല") instead of writing UNSURE.
# Those leak the RAG plumbing to the caller and are not counted as refusals, so map them to REFUSAL.
META_REFUSAL = ["നൽകിയിട്ടില്ല", "വിവരങ്ങളിൽ ഇല്ല", "വിവരങ്ങളിൽ പറയുന്നില്ല", "പരാമർശിച്ചിട്ടില്ല",
                "വിവരം ലഭ്യമല്ല", "വിവരങ്ങൾ ലഭ്യമല്ല", "not provided", "not mentioned"]
_SENT_END = re.compile(r"[.?!।](?=\s|$)")


def clean_answer(out: str) -> tuple[str, bool]:
    """Returns (answer, refused). Maps meta-refusals to REFUSAL and drops a sentence
    cut off by the token limit, so TTS never reads half a sentence."""
    out = out.strip()
    if not out or "UNSURE" in out.upper() or any(m in out for m in META_REFUSAL):
        return REFUSAL, True
    ends = [m.end() for m in _SENT_END.finditer(out)]
    if ends and ends[-1] < len(out):
        out = out[: ends[-1]]
    return out, False


class State(TypedDict, total=False):
    query: str
    route: Route
    route_by: str            # "rule" | "llm" | "default"
    tool: str                # set when a deterministic tool answered (e.g. "fee_calc")
    previous: str            # the caller's previous question in this conversation, if any
    previous_route: Route
    context: str
    sources: list[dict]
    answer: str
    refused: bool


def passage(doc: dict) -> str:
    """How one KB entry appears in the LLM prompt (without the BM25-only Keywords line)."""
    return f"[{doc['id']}] {doc['title']}\n{doc['text'].split(' Keywords:')[0]}"


def answer_message(ctx: str, query: str) -> str:
    """User message for the answer step. Passages come first so that a prompt
    cache can reuse 'system prompt + passage' across different questions."""
    return f"വിവരങ്ങൾ:\n{ctx}\n\nചോദ്യം: {query}"


def rule_route(query: str) -> Route | None:
    q = normalise(query)
    for route, stems in RULES:
        if any(normalise(s) in q for s in stems):
            return route
    return None


def build_graph(retriever: Retriever | None = None):
    retriever = retriever or Retriever()

    def router(s: State) -> State:
        r = rule_route(s["query"])
        if r:
            return {"route": r, "route_by": "rule"}
        # Short follow-up after a service question ("അതിന് എത്ര ദിവസം എടുക്കും?"): stay on service.
        if s.get("previous_route") == "service_faq" and len(s["query"].split()) <= 6:
            return {"route": "service_faq", "route_by": "follow-up"}
        if llm.enabled():
            try:
                label = llm.chat(CLASSIFY_PROMPT, s["query"], max_tokens=5, temperature=0).lower()
                m = re.search(r"weather|news|service_faq|other", label)
                if m:
                    return {"route": m.group(0), "route_by": "llm"}
            except Exception:
                pass
        # Unknown: try retrieval; the confidence check refuses if nothing matches.
        return {"route": "service_faq", "route_by": "default"}

    def weather(s: State) -> State:
        try:
            w = tools.weather(s["query"])
            return {"context": w["text"], "sources": [{"title": w["source"]}]}
        except Exception:
            return {"context": ""}

    def news(s: State) -> State:
        try:
            n = tools.news(s["query"])
            return {"context": n["text"],
                    "sources": [{"title": i["title"], "url": i["link"]} for i in n["items"]]}
        except Exception:
            return {"context": ""}

    by_id = {d["id"]: d for d in retriever.docs}

    def calc(s: State) -> State:
        """Fee questions that need band arithmetic are answered in code, not by the LLM."""
        out = fees.answer(s["query"])
        if not out:
            return {}
        return {"answer": out["answer"], "refused": False, "tool": "fee_calc",
                "sources": [{"id": i, "title": by_id[i]["title"], "url": by_id[i]["source"],
                             "verify": by_id[i].get("verify", False)} for i in out["source_ids"] if i in by_id]}

    def retrieve(s: State) -> State:
        query = s["query"]
        # A follow-up that names no service borrows the previous question for retrieval only.
        if s.get("previous") and not mentions_service(query):
            query = s["previous"] + " " + query
        hits = retriever.search(query, k=3)
        if not retriever.confident(hits):
            return {"context": "", "sources": []}
        # Only the top CTX_K passages go to the LLM (prefill dominates CPU latency),
        # and without the English 'Keywords:' line, which exists only for BM25.
        ctx = "\n\n".join(passage(h.doc) for h in hits[:CTX_K])
        return {"context": ctx, "sources": [
            {"id": h.doc["id"], "title": h.doc["title"], "url": h.doc["source"],
             "verify": h.doc.get("verify", False), "bm25": round(h.bm25, 2),
             "cos": None if h.cos is None else round(h.cos, 3)} for h in hits[:CTX_K]]}

    def other(s: State) -> State:
        return {"answer": SCOPE, "refused": False}

    def answer(s: State) -> State:
        route, ctx = s["route"], s.get("context", "")
        if route in ("weather", "news"):
            return {"answer": ctx or TOOL_FAIL, "refused": not ctx}
        if not ctx:
            return {"answer": REFUSAL, "refused": True}
        if not llm.enabled():
            # Extractive fallback: read out the best passage.
            best = ctx.split("\n\n")[0].split("\n", 1)[1]
            return {"answer": best.split(" Keywords:")[0], "refused": False}
        try:
            out = llm.chat(ANSWER_PROMPT, answer_message(ctx, s["query"]),
                           max_tokens=ANSWER_TOKENS, temperature=0)
        except Exception:
            return {"answer": TOOL_FAIL, "refused": True}
        answer, refused = clean_answer(out)
        return {"answer": answer, "refused": refused}

    g = StateGraph(State)
    for name, fn in [("router", router), ("weather", weather), ("news", news),
                     ("calc", calc), ("retrieve", retrieve), ("other", other), ("answer", answer)]:
        g.add_node(name, fn)
    g.set_entry_point("router")
    g.add_conditional_edges("router", lambda s: s["route"], {
        "weather": "weather", "news": "news", "service_faq": "calc", "other": "other"})
    g.add_conditional_edges("calc", lambda s: "done" if s.get("tool") else "retrieve",
                            {"done": END, "retrieve": "retrieve"})
    for n in ("weather", "news", "retrieve"):
        g.add_edge(n, "answer")
    g.add_edge("other", END)
    g.add_edge("answer", END)
    return g.compile()


def ask(query: str, graph=None) -> State:
    graph = graph or build_graph()
    return graph.invoke({"query": query})
