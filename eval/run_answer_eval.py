"""End-to-end answer eval on eval/answer_eval.jsonl (100 questions, 11 Nov 2026).

    python -m eval.run_answer_eval                          # Gemma 4 E2B Q4_0 in-process (llama.cpp)
    python -m eval.run_answer_eval --http                   # use helpdesk.llm's HTTP endpoint instead
    python -m eval.run_answer_eval --rescore eval/results/answer_eval_rules_fix.json   # re-score saved answers

Unlike eval.run_eval and eval.run_synth, this runs the LLM, so it checks the answer itself, not only the
route and the top passage. Writes eval/results/<--out>.json (one row per question: verdict, path, sources,
tokens, seconds, answer) and prints a table by category.

The set: one question per KB entry (50), colloquial variants (15), fee arithmetic (8), government questions
the KB does not cover (10), off-topic (10), weather (4), news (3). Drafted with Gemini from data/kb.jsonl
and checked: every kb_id exists, every key fact is in the cited entry, fee answers match helpdesk/fees.py,
no question copies a KB example question.

Verdicts
  answer / fee_calc: correct = not refused, every key fact present (stem match, digits without commas),
                     no must_not; partial = some key facts; false_refusal = refused; wrong otherwise.
  decline / scope:   correct = refused (refusal, scope or clarify message); wrong = answered.
  weather / news:    correct = routed to that tool (the HTTP call is not scored).
Substring matching cannot see paraphrases; read the answers of non-correct cases before trusting a change.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
SET = HERE / "answer_eval.jsonl"
CHILLU = {"ണ്‍": "ൺ", "ന്‍": "ൻ", "ര്‍": "ർ", "ല്‍": "ൽ", "ള്‍": "ൾ"}
ENDINGS = re.compile(r"(ും|ുടെ|ിന്റെ|ിനെ|ിന്|ിൽ|ാണ്|ാണ|യും|യുടെ|ത്തിന്റെ)$")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    for k, v in CHILLU.items():
        s = s.replace(k, v)
    s = s.replace("‍", "").replace("‌", "")
    return re.sub(r"(?<=\d),(?=\d)", "", s)                 # 5,000 -> 5000


def _stem(w: str) -> str:
    w = ENDINGS.sub("", w)
    return w[:-1] if w.endswith("്") else w


def has_fact(answer: str, fact: str) -> bool:
    a, f = _norm(answer), _norm(fact)
    if re.fullmatch(r"[\d.]+", f):                           # a number: whole match, not inside 2026
        return re.search(rf"(?<![\d.]){re.escape(f)}(?!\d)", a) is not None
    return f in a or all(_stem(w)[:max(3, len(_stem(w)) - 1)] in a for w in f.split())


def verdict(case: dict, route: str | None, refused: bool, answer: str) -> str:
    exp = case["expected_route"]
    if exp in ("weather", "news"):
        return "correct" if route == exp else "wrong"
    if exp in ("decline", "scope"):
        return "correct" if refused else "wrong"
    if refused:
        return "false_refusal"
    hits = [has_fact(answer, f) for f in case["key_facts"]]
    bad = any(has_fact(answer, m) for m in case.get("must_not", []))
    if all(hits) and not bad:
        return "correct"
    return "partial" if any(hits) and not bad else "wrong"


def run(cases: list[dict], http: bool) -> list[dict]:
    os.environ.setdefault("HELPDESK_DENSE", "0")
    os.environ.setdefault("HELPDESK_CTX_K", "1")
    os.environ.setdefault("HELPDESK_CANON", "1")              # as on the Space
    from helpdesk import llm
    from helpdesk.graph import build_graph
    from helpdesk.retriever import Retriever

    usage = {}
    if not http:
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama
        llama = Llama(model_path=hf_hub_download("ggml-org/gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-Q4_0.gguf"),
                      n_ctx=2048, n_threads=int(os.getenv("N_THREADS", "4")), verbose=False)
        llm.use_llama_cpp(llama)

        def chat(system, user, max_tokens=200, temperature=0.2):
            out = llama.create_chat_completion(messages=[{"role": "system", "content": system},
                                                         {"role": "user", "content": user}],
                                               max_tokens=max_tokens, temperature=temperature)
            usage.update(tokens=out["usage"]["completion_tokens"], cap=max_tokens)
            return out["choices"][0]["message"]["content"].strip()
        llm.chat = chat                                       # graph.py calls llm.chat at call time

    graph = build_graph(Retriever(dense=os.environ["HELPDESK_DENSE"] == "1"))
    rows = []
    for c in cases:
        usage.clear()
        t = time.perf_counter()
        s = graph.invoke({"query": c["question"]})
        secs = time.perf_counter() - t
        answer, refused = s.get("answer", ""), bool(s.get("refused"))
        sources = [x.get("id") for x in s.get("sources", []) if x.get("id")]
        rows.append(dict(
            id=c["id"], category=c["category"], expected=c["expected_route"],
            verdict=verdict(c, s.get("route"), refused, answer),
            route=s.get("route"), route_by=s.get("route_by"), tool=s.get("tool"), refused=refused,
            sources=sources[:2], kb_hit=bool(set(c["kb_ids"]) & set(sources)) if c["kb_ids"] else None,
            llm=bool(usage), tokens=usage.get("tokens"), secs=round(secs, 2), answer=answer))
        r = rows[-1]
        print(f'{r["id"]} {r["verdict"]:13} {r["route"] or "-":12} {r["tool"] or "":12} {r["secs"]:5.1f}s', flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="answer_eval_run", help="name for eval/results/<out>.json")
    ap.add_argument("--http", action="store_true", help="use helpdesk.llm's HTTP endpoint (llama.cpp server)")
    ap.add_argument("--rescore", help="re-apply the verdicts to a saved results file, no LLM")
    ap.add_argument("--only", help="comma-separated ids, e.g. hd006,hd021")
    args = ap.parse_args()

    cases = [json.loads(l) for l in SET.open(encoding="utf-8") if l.strip()]
    if args.only:
        keep = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in keep]
    if args.rescore:
        saved = {r["id"]: r for r in json.load(open(args.rescore, encoding="utf-8"))}
        rows = []
        for c in cases:
            r = dict(saved[c["id"]])
            r["verdict"] = verdict(c, r["route"], r["refused"], r["answer"])
            rows.append(r)
    else:
        rows = run(cases, args.http)

    out = HERE / "results" / f"{args.out}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    cats = sorted({r["category"] for r in rows})
    kinds = ["correct", "partial", "wrong", "false_refusal"]
    print(f"\n{'category':15}" + "".join(f"{k:>14}" for k in kinds))
    for cat in cats + ["all"]:
        n = Counter(r["verdict"] for r in rows if cat == "all" or r["category"] == cat)
        print(f"{cat:15}" + "".join(f"{n[k]:>14}" for k in kinds))
    timed = sorted(r["secs"] for r in rows if r["llm"])
    if timed:
        print(f"\nLLM turns {len(timed)}: median {timed[len(timed) // 2]:.1f} s; wrote {out}")


if __name__ == "__main__":
    main()
