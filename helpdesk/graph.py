"""LangGraph agent for the Malayalam voice helpdesk (reply stage).

    router ─► weather ───────────────────────► answer ─► END
           ├► news ──────────────────────────► answer
           ├► calc ─ fee computed in code ───────────────► END
           │    └─ not a fee question ─► retrieve ► answer   (or "I don't have that" in code)
           └► other (scope message, or "which service?") ────► END

Small models (Gemma 4 E2B) are unreliable at free-form tool calling, so the
graph makes the routing explicit with the keyword knowledge in intent.py.
A question must name a service (or follow one) to be answered from the
knowledge base; the service filters the passages; a strict facet the
knowledge base lacks is declined in code. Tool numbers go out through
templates; the LLM only writes service answers, only from one passage.
"""
from __future__ import annotations

import os
import re
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from . import fees, llm, tools
from .normalise import normalise
from .intent import (CLARIFY_FACETS, STRICT_FACETS, OTHER_ANIMAL_MSG, detect_facet, detect_services, fix_asr, missing_message,
                     other_animal, out_of_scope)
from .retriever import BM25_MIN, Hit, Retriever

Route = Literal["weather", "news", "service_faq", "other"]
CTX_K = int(os.getenv("HELPDESK_CTX_K", "2"))           # passages sent to the LLM
ANSWER_TOKENS = int(os.getenv("HELPDESK_ANSWER_TOKENS", "100"))   # ~0.34 s per token on a 2-core CPU

REFUSAL = ("ക്ഷമിക്കണം, ഇതിനെക്കുറിച്ച് എനിക്ക് ഉറപ്പുള്ള വിവരം ഇല്ല. "
           "നിങ്ങളുടെ പഞ്ചായത്ത് അല്ലെങ്കിൽ നഗരസഭ ഓഫീസിലോ അക്ഷയ കേന്ദ്രത്തിലോ ചോദിക്കൂ.")
SCOPE = ("ഞാൻ പഞ്ചായത്ത്, നഗരസഭ സേവനങ്ങൾ, കാലാവസ്ഥ, വാർത്ത എന്നിവയിൽ സഹായിക്കാം. "
         "ഉദാഹരണത്തിന്, വിവാഹ രജിസ്ട്രേഷന് എന്തൊക്കെ രേഖകൾ വേണം എന്ന് ചോദിക്കാം.")
TOOL_FAIL = "ക്ഷമിക്കണം, ഇപ്പോൾ ആ വിവരം ലഭിക്കുന്നില്ല. കുറച്ചു കഴിഞ്ഞ് വീണ്ടും ചോദിക്കൂ."
CLARIFY = "ഏത് സേവനത്തെക്കുറിച്ചാണ് ചോദിക്കുന്നത്? ഉദാഹരണത്തിന് വിവാഹ രജിസ്ട്രേഷൻ, പെൻഷൻ, കെട്ടിട പെർമിറ്റ്."
FOLLOW_UP_MAX_WORDS = 8
ACK = re.compile(r"^(ഇല്ല|അതെ|ശരി|ശരിയാണ്|ഓക്കെ|ഓക്കേ|ok|okay|നന്ദി|താങ്ക്സ്|thanks|thank you|വേണ്ട|മതി)$")
ACK_REPLY = "ശരി. മറ്റെന്തെങ്കിലും അറിയണമെങ്കിൽ ചോദിക്കൂ."

# Stems, not full words: Malayalam inflects heavily (മഴ / മഴയുണ്ടോ, വാർത്ത / വാർത്തകൾ).
RULES: list[tuple[Route, list[str]]] = [
    ("weather", ["കാലാവസ്ഥ", "മഴ", "ചൂട്", "താപനില", "ഡിഗ്രി", "weather", "rain", "temperature"]),
    ("news", ["വാർത്ത", "news", "headline", "സംഭവ"]),
]
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
    route_by: str            # "rule" | "follow-up" | "open" | "none"
    open_id: str             # route_by "open": the passage whose title matched
    tool: str                # set when a deterministic tool answered (e.g. "fee_calc")
    previous: str            # the caller's previous question in this conversation, if any
    previous_route: Route
    previous_services: list[str]
    previous_tool: str        # e.g. "other_animal": the last answer was a decline, carry nothing
    previous_topic: str       # the caller passes back the last turn's "topic"
    topic: str               # title of the passage that started this thread of follow-ups
    services: list[str]      # services the question is about (detected, or carried from a follow-up)
    facet: str | None        # what it asks: amount, documents, time, validity, compulsory, ...
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


def attribution(sources: list[dict]) -> str:
    """Spoken source for Tier-2 answers: one panchayat's charter may differ from yours."""
    top = sources[0] if sources else {}
    if top.get("tier") != 2:
        return ""
    if top.get("site"):
        return " (തദ്ദേശ സ്വയംഭരണ വകുപ്പിന്റെ വെബ്സൈറ്റ് പ്രകാരം.)"
    if top.get("kind") == "canon" and len(top.get("lsgis") or []) >= 2:
        return " (പഞ്ചായത്തുകളുടെ പൗരാവകാശ രേഖകൾ പ്രകാരം. നിങ്ങളുടെ പഞ്ചായത്തിൽ വ്യത്യാസമുണ്ടാകാം.)"
    if top.get("agreed_by") and len(top["agreed_by"]) >= 2:
        return " (പഞ്ചായത്തുകളുടെ പൗരാവകാശ രേഖകൾ പ്രകാരം.)"
    if top.get("lsgi"):
        return (f" ({top['lsgi']} പഞ്ചായത്തിന്റെ പൗരാവകാശ രേഖ പ്രകാരം. "
                "നിങ്ങളുടെ പഞ്ചായത്തിൽ വ്യത്യാസമുണ്ടാകാം.)")
    return ""


def rule_route(query: str) -> Route | None:
    """weather / news by keyword, service_faq if a service is named, else None."""
    q = fix_asr(query)
    for route, stems in RULES:
        if any(normalise(s) in q for s in stems):
            return route
    return "service_faq" if detect_services(query) else None


def build_graph(retriever: Retriever | None = None):
    retriever = retriever or Retriever()

    def router(s: State) -> State:
        q = s["query"]
        facet = detect_facet(q)
        r = rule_route(q)
        if r == "service_faq":
            return {"route": r, "route_by": "rule", "services": detect_services(q), "facet": facet}
        if r:
            return {"route": r, "route_by": "rule"}
        # A service no keyword names, found by title overlap in the Tier-2 documents
        # ("പരസ്യം വെക്കാൻ അനുമതി വേണോ" -> the advertisement-permission row of a charter).
        # Checked before follow-ups: a new topic in a few words is not a follow-up.
        # Follow-up that names no service ("അതിന് എന്തൊക്കെ രേഖകൾ വേണം?"): carry the previous service.
        prev = s.get("previous_services") or detect_services(s.get("previous") or "")
        if ACK.match(normalise(q, strip_punct=True)):
            # "ഇല്ല", "ശരി", "നന്ദി": a reply to us, not a question ("ഇല്ല" got a vaccination passage).
            return {"route": "other", "route_by": "ack", "facet": None}
        follow = (bool(prev) and s.get("previous_route") == "service_faq"
                  and len(q.split()) <= FOLLOW_UP_MAX_WORDS and not out_of_scope(q)
                  and "lsgd_other" not in prev and s.get("previous_tool") != "other_animal"
                  and retriever.fits_services(q, prev))
        if not out_of_scope(q):
            h = retriever.open_match(q)
            # A question that fits the last service stays with it, unless two of its words name
            # another service's passage ("തൊഴിലുറപ്പ് കാർഡിന് എങ്ങനെ അപേക്ഷിക്കാം" after a marriage
            # question: "കാർഡ്" occurs in marriage passages too).
            if h is not None and (not follow or (retriever.last_open[1] >= 2 and retriever.last_open[0] >= 1.4
                                                 and h.doc.get("service") not in prev)):
                return {"route": "service_faq", "route_by": "open", "services": [h.doc.get("service", "")],
                        "facet": facet, "open_id": h.doc["id"]}
        if ("lsgd_other" in prev and s.get("previous_route") == "service_faq"
                and len(q.split()) <= FOLLOW_UP_MAX_WORDS and not retriever.has_content(q)):
            # The previous answer came from an untagged charter row: "lsgd_other" is a catch-all,
            # not a service to search ("രജിസ്ട്രേഷൻ തുക എത്ര?" after the advertisement question
            # found a paramedical-registration row). Re-find the previous passage instead.
            h = retriever.open_match((s.get("previous") or "") + " " + q)
            if h is not None:
                return {"route": "service_faq", "route_by": "follow-up", "services": [h.doc.get("service", "")],
                        "facet": facet, "open_id": h.doc["id"]}
        if follow:
            # Keep the passage the previous answer came from when the previous question still finds
            # it ("അതിന് എത്ര ദിവസം?" after the advertisement row went to a factory licence row).
            h = retriever.open_match(s.get("previous") or "")
            pin = h.doc["id"] if h is not None and h.doc.get("service") in prev else None
            return {"route": "service_faq", "route_by": "follow-up", "services": prev, "facet": facet,
                    **({"open_id": pin} if pin else {})}
        # No service: answering from the knowledge base would only find a lookalike passage
        # ("കേരളത്തിന്റെ തലസ്ഥാനം" -> a pension passage -> "തി ആണ്").
        return {"route": "other", "route_by": "none", "facet": facet}

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
        q = fix_asr(s["query"])
        out = fees.answer(q)
        if not out and s.get("route_by") == "follow-up" and s.get("facet") == "amount" and s.get("previous"):
            # "കല്യാണം കഴിഞ്ഞു രണ്ട് മാസമായി." -> "രജിസ്ട്രേഷന് തുക എത്ര?": the delay is in the previous turn.
            out = fees.answer(fix_asr(s["previous"]) + " " + q)
        if not out:
            return {}
        return {"answer": out["answer"], "refused": False, "tool": "fee_calc",
                "sources": [{"id": i, "title": by_id[i]["title"], "url": by_id[i]["source"],
                             "verify": by_id[i].get("verify", False)} for i in out["source_ids"] if i in by_id]}

    def retrieve(s: State) -> State:
        services, facet = s.get("services") or [], s.get("facet")
        if ("pet" in services and other_animal(s["query"])
                and (facet == "compulsory" or "ലൈസൻസ" in fix_asr(s["query"]))):
            return {"answer": OTHER_ANIMAL_MSG, "refused": True, "tool": "other_animal", "sources": []}
        if not s.get("open_id") and retriever.missing_facet(services, facet):
            # e.g. building-permit validity: no entry has it, so say so instead of letting
            # Gemma borrow the trade licence's "5 years" (8 Nov log).
            return {"answer": missing_message(services, facet), "refused": True, "tool": "missing_facet",
                    "sources": []}
        query = s["query"]
        follow = s.get("route_by") == "follow-up"
        if follow and s.get("previous"):
            query = s["previous"] + " " + query          # keeps specifics like "വിധവ" for retrieval
        if follow and s.get("previous_topic"):
            # Two follow-ups in a row lose the topic: "വിധവ പെൻഷൻ…" -> "എന്തൊക്കെ രേഖകൾ?" ->
            # "തുകയെത്രയാണ്?" has no "വിധവ" in either question and found disability pension
            # (10 Nov, canon on). The title of the passage that started the thread carries it.
            query = s["previous_topic"] + " " + query
        hits = retriever.search(query, k=3, services=services, facet=facet)
        if follow and facet in STRICT_FACETS and hits and facet not in hits[0].doc.get("facets", []):
            # The topic words pull the topic's own passage to the top; when it does not answer the
            # asked amount/time/validity, the best-ranked passage that does goes first
            # ("തുകയെത്രയാണ്?" in a widow-pension thread: the pension-amount entry, not eligibility).
            j = next((j for j, h in enumerate(hits) if facet in h.doc.get("facets", [])), None)
            if j is not None:
                hits = [hits[j]] + hits[:j] + hits[j + 1:]
        open_id = s.get("open_id")
        if not open_id and hits and s.get("route_by") != "follow-up":   # a follow-up keeps its service
            # A passage whose title says more of the question than the top hit's title does
            # ("വാസയോഗ്യമായ വീടില്ലെന്ന സർട്ടിഫിക്കറ്റ്": a charter row, not the residence certificate).
            h = retriever.open_match(s["query"])
            if h is not None:
                cand, top = (retriever.title_score(s["query"], d) for d in (h.doc, hits[0].doc))
                # Clearly better, not a near tie, and not trading an entry that answers the asked
                # facet for one that does not ("കർഷക തൊഴിലാളി പെൻഷൻ എത്ര": the amount entry wins).
                # A curated entry that answers the asked amount/time stays: "തൊഴിലില്ലായ്മ വേതനം
                # എത്ര" is the Tier-1 ₹120 entry, not a charter row about distributing it.
                answers = (facet in STRICT_FACETS and facet in hits[0].doc.get("facets", [])
                           and (facet not in h.doc.get("facets", []) or hits[0].doc.get("tier", 1) == 1))
                # At least one more distinctive word, matched well (>= 0.8): "വിധവ പെൻഷൻ" beats a
                # "ഭിന്നശേഷി പെൻഷൻ" entry BM25 put first; "സർവീസുകൾ" (a 0.6 stem match) does not move
                # an Akshaya question to the service-charge entry.
                # And within the service the keywords named: "വിവാഹം കഴിഞ്ഞു രണ്ടു മാസമായി" must not
                # become the pension row for unmarried women over 50.
                same = not services or "lsgd_other" in services or h.doc.get("service") in services
                if cand > top and (top == 0 or cand - top >= 0.8) and not answers and same:
                    open_id = h.doc["id"]
        if open_id:
            # Title overlap vouches for this passage; put it first.
            pinned = next(i for i, d in enumerate(retriever.docs) if d["id"] == open_id)
            hits = [Hit(retriever.docs[pinned], 1.0, BM25_MIN, None)] + [h for h in hits if h.doc["id"] != open_id]
        elif not retriever.confident(hits):
            return {"context": "", "sources": []}
        # Only the top CTX_K passages go to the LLM (prefill dominates CPU latency).
        ctx = "\n\n".join(passage(h.doc) for h in hits[:CTX_K])
        topic = (s.get("previous_topic") if follow else "") or hits[0].doc["title"]
        return {"context": ctx, "topic": topic, "sources": [
            {"id": h.doc["id"], "title": h.doc["title"], "url": h.doc["source"],
             "verify": h.doc.get("verify", False), "bm25": round(h.bm25, 2),
             "tier": h.doc.get("tier", 1), "lsgi": h.doc.get("lsgi"), "agreed_by": h.doc.get("agreed_by"),
             "site": h.doc.get("site"), "kind": h.doc.get("kind"), "lsgis": h.doc.get("lsgis"),
             "cos": None if h.cos is None else round(h.cos, 3)} for h in hits[:CTX_K]]}

    def other(s: State) -> State:
        if s.get("route_by") == "ack":
            return {"answer": ACK_REPLY, "refused": True, "tool": "ack"}
        # "തുക എത്രയാണ്?" with no service and nothing to follow: ask which service.
        if s.get("facet") in CLARIFY_FACETS and not out_of_scope(s["query"]):
            return {"answer": CLARIFY, "refused": True, "tool": "clarify"}
        return {"answer": SCOPE, "refused": True}

    def answer(s: State) -> State:
        route, ctx = s["route"], s.get("context", "")
        if s.get("tool") in ("missing_facet", "other_animal"):     # answered in code already
            return {}
        if route in ("weather", "news"):
            return {"answer": ctx or TOOL_FAIL, "refused": not ctx}
        if not ctx:
            return {"answer": REFUSAL, "refused": True}
        if not llm.enabled():
            # Extractive fallback: read out the best passage.
            best = ctx.split("\n\n")[0].split("\n", 1)[1]
            return {"answer": best.split(" Keywords:")[0] + attribution(s.get("sources", [])), "refused": False}
        try:
            out = llm.chat(ANSWER_PROMPT, answer_message(ctx, fix_asr(s["query"], snap=False)),
                           max_tokens=ANSWER_TOKENS, temperature=0)
        except Exception:
            return {"answer": TOOL_FAIL, "refused": True}
        answer, refused = clean_answer(out)
        if not refused:
            answer += attribution(s.get("sources", []))
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
