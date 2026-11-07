"""Evaluate the helpdesk on eval/questions.jsonl.

Offline checks (no LLM, no network):  python -m eval.run_eval
  - route accuracy (keyword router)
  - retrieval hit@3 for answerable service questions
  - refusal accuracy (confidence gate) on answerable + unanswerable questions
  - district detection for weather questions
  - BM25/cosine scores per question, to tune HELPDESK_BM25_MIN / HELPDESK_COS_MIN

Full run (LLM + live tools):  python -m eval.run_eval --full
  adds: answer contains the key fact; refused when it should; latency.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from helpdesk import tools
from helpdesk.graph import build_graph, rule_route
from helpdesk.retriever import Retriever

Q_PATH = Path(__file__).with_name("questions.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="run the whole graph (needs LLM + network)")
    ap.add_argument("--dense", action="store_true", help="use BGE-M3 dense retrieval too")
    args = ap.parse_args()

    qs = [json.loads(l) for l in Q_PATH.open(encoding="utf-8") if l.strip()]
    ret = Retriever(dense=args.dense)
    route_ok = hit = n_ans = refuse_ok = n_svc = dist_ok = n_w = 0

    print(f"{'id':4} {'route':12} {'top3':34} {'bm25':>6} {'cos':>6} conf  ok")
    for q in qs:
        r = rule_route(q["question"]) or "service_faq"
        route_ok += r == q["route"]
        line = f"{q['id']:4} {r:12}"
        if q["route"] == "service_faq":
            n_svc += 1
            hits = ret.search(q["question"], k=3)
            conf = ret.confident(hits)
            ids = [h.doc["id"] for h in hits]
            bm = max(h.bm25 for h in hits)
            cos = max((h.cos for h in hits if h.cos is not None), default=float("nan"))
            if q["expect"] == "refuse":
                ok = not conf
            else:
                n_ans += 1
                got = any(i in q["expect"] for i in ids)
                hit += got
                ok = got and conf
            refuse_ok += conf == (q["expect"] != "refuse")
            line += f" {','.join(ids):34} {bm:6.2f} {cos:6.3f} {'yes ' if conf else 'no  '}  {'✓' if ok else '✗'}"
        elif q["route"] == "weather":
            n_w += 1
            d = tools.find_district(q["question"])
            dist_ok += d == q["district"]
            line += f" district={d}  {'✓' if d == q['district'] else '✗'}"
        print(line)

    print(f"\nroute accuracy      {route_ok}/{len(qs)}")
    print(f"retrieval hit@3     {hit}/{n_ans}")
    print(f"answer/refuse gate  {refuse_ok}/{n_svc}")
    print(f"district detection  {dist_ok}/{n_w}")

    if args.full:
        graph = build_graph(ret)
        good = 0
        llm_times, rows = [], []
        print("\n--- full graph ---")
        for q in qs:
            t = time.perf_counter()
            s = graph.invoke({"query": q["question"]})
            dt = time.perf_counter() - t
            if q.get("expect") == "refuse":
                ok = s.get("refused", False)
            elif "must_contain" in q:
                ok = (not s.get("refused")) and q["must_contain"] in s["answer"]
            else:
                ok = not s.get("refused")
            good += ok
            if s["route"] == "service_faq":
                llm_times.append(dt)
            rows.append({"id": q["id"], "ok": bool(ok), "seconds": round(dt, 1), "route": s["route"],
                         "route_by": s.get("route_by"), "answer": s["answer"],
                         "sources": [x.get("id") for x in s.get("sources", [])]})
            print(f"{q['id']} {'✓' if ok else '✗'} {dt:5.1f}s [{s['route']}/{s.get('route_by')}] {s['answer'][:120]}")
        print(f"\nend-to-end correct  {good}/{len(qs)}")
        if llm_times:
            print(f"service answer latency  median {statistics.median(llm_times):.1f}s, "
                  f"max {max(llm_times):.1f}s (n={len(llm_times)})")
        out = Path(__file__).with_name("out.jsonl")
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"full answers written to {out} (open in VS Code to read the Malayalam)")


if __name__ == "__main__":
    main()
