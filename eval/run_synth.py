"""Held-out retrieval eval on the questions ingest.canon generated (task A).

    python -m eval.run_synth                 # baseline (rows) vs canonical services, test split
    python -m eval.run_synth --split all     # dev + test
    python -m eval.run_synth --show 15       # print some misses

No LLM, no network: it runs the graph with HELPDESK_LLM=none and checks routing and the top
passage, which is what decided most wrong answers in the Space logs.
A question is right when
  in scope:      route is service_faq and the top source is the expected entry, the canonical
                 service, or one of the charter rows it was built from (a fee-calculator answer
                 counts when it cites the expected Tier-1 entry)
  out of scope:  route is "other" (scope message, clarify or ack)
"strict" needs the exact id. "Same service" also accepts a top passage that names the same
service as the expected one (same service tag, title similarity >= 0.5, the grouping
threshold): the Tier-1 entry or another panchayat's row for "death certificate" answers a
death-certificate question as well as the canonical passage does, and strict counts it wrong.
The model that wrote these questions is not the one that answers on the Space, and none of
them are in the retrieval index (ingest.canon indexes the other half).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

SYNTH = Path(__file__).with_name("synth_questions.jsonl")


def evaluate(qs: list[dict], canon: bool, show: int = 0) -> dict:
    os.environ["HELPDESK_LLM"] = "none"
    os.environ["HELPDESK_CANON"] = "1" if canon else "0"
    from helpdesk.graph import build_graph
    from helpdesk.retriever import Retriever
    from ingest.canon import title_similarity
    retr = Retriever(dense=os.getenv("HELPDESK_DENSE", "0") == "1")
    graph = build_graph(retr)
    meta = {d["id"]: (d.get("service", ""), d.get("title", "")) for d in retr.docs}
    by_kind, misses, why, per_q = defaultdict(lambda: [0, 0, 0]), [], Counter(), []
    for q in qs:
        s = graph.invoke({"query": q["q"]})
        ids = [x.get("id") for x in s.get("sources") or []]
        if q["kind"] == "out_of_scope":
            ok = s["route"] == "other"
            reason = "" if ok else f"answered ({s['route']})"
        else:
            ok = s["route"] == "service_faq" and bool(ids) and ids[0] in q["expect"]
            reason = ("" if ok else "routed other" if s["route"] != "service_faq" else
                      f"declined ({s.get('tool')})" if not ids else "wrong passage")
        same = ok
        if not ok and q["kind"] != "out_of_scope" and s["route"] == "service_faq" and ids and q.get("title"):
            svc, title = meta.get(ids[0], ("", ""))
            same = svc == q["service"] and title_similarity(title, q["title"]) >= 0.5
            if same:
                reason = "same service, other passage"
        by_kind[q["kind"]][0] += ok
        by_kind[q["kind"]][1] += 1
        by_kind[q["kind"]][2] += same
        per_q.append((bool(ok), bool(same), ids[:1]))
        if not ok:
            why[reason] += 1
            misses.append((q["kind"], q["q"], reason, ids[:1]))
    total = [sum(v[0] for v in by_kind.values()), sum(v[1] for v in by_kind.values()),
             sum(v[2] for v in by_kind.values())]
    for m in misses[:show]:
        print("   miss", m)
    return {"total": total, "by_kind": {k: v for k, v in by_kind.items()}, "why": dict(why), "per_q": per_q}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "dev", "all"])
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()
    qs = [json.loads(l) for l in SYNTH.open(encoding="utf-8") if l.strip()]
    qs = [q for q in qs if args.split == "all" or q["split"] == args.split]
    print(f"{len(qs)} questions ({args.split}): {dict(Counter(q['kind'] for q in qs))}")
    report = {}
    for name, canon in (("baseline (charter rows)", False), ("canonical services", True)):
        r = evaluate(qs, canon, args.show)
        report[name] = r
        t = r["total"]
        kinds = "  ".join(f"{k} {v[0]}|{v[2]}/{v[1]}" for k, v in sorted(r["by_kind"].items()))
        print(f"{name:26} strict {t[0]}/{t[1]} = {100 * t[0] / max(t[1], 1):.1f}%   "
              f"same service {t[2]}/{t[1]} = {100 * t[2] / max(t[1], 1):.1f}%")
        print(f"{'':26} by kind (strict|same/n): {kinds}\n{'':26} misses: {r['why']}")
    base, can = (report[k].pop("per_q") for k in report)
    for j, metric in ((0, "strict"), (1, "same service")):
        gained = [i for i in range(len(qs)) if can[i][j] and not base[i][j]]
        lost = [i for i in range(len(qs)) if base[i][j] and not can[i][j]]
        print(f"{metric:13} canon fixes {len(gained)}, breaks {len(lost)}; exact McNemar p = {mcnemar(len(gained), len(lost)):.3g}")
        report[f"flips_{metric.replace(' ', '_')}"] = {"gained": len(gained), "lost": len(lost)}
        if j == 1:
            for i in lost[:args.show or 10]:
                print("   broken by canon:", qs[i]["kind"], qs[i]["q"], base[i][2], "->", can[i][2])
    Path(__file__).with_name("synth_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))


def mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar test on the discordant pairs (binomial, p = 0.5)."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


if __name__ == "__main__":
    main()
