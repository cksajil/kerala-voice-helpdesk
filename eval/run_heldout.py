"""Run the whole graph on the frozen held-out set (eval/heldout_v1.jsonl) and score it.

    python -m eval.run_heldout --kb old --tag bm25_old            # Gemma in-process (llama-cpp-python)
    python -m eval.run_heldout --kb expanded --dense --tag dense_exp
    python -m eval.run_heldout --llm http --tag server            # helpdesk.llm's HTTP endpoint (llama-server)
    python -m eval.run_heldout --llm none --tag offline           # routing + retrieval only, no LLM
    python -m eval.run_heldout --rescore eval/results/heldout_<tag>.csv   # re-score saved answers

Meant for Kaggle CPU (2 threads, close to the Space); see notebooks/kaggle_phase3. Settings are the
Space's (HELPDESK_CANON=1, HELPDESK_CTX_K=1, Gemma 4 E2B Q4_0, greedy) unless overridden.
Scoring (eval/HELDOUT.md):
  route         final route == expected (out-of-scope: any route that ends in a decline)
  hit@1         the passage the answer was built from is one of expect_kb (in-scope items)
  hit@3         one of expect_kb in the retriever's top 3 for the question
  answer        in-scope: every key fact (alternatives split by '|'), no must-not fact, not declined
  fee           items with an expected fee: the number is in the answer
  refusal P/R   positives: out-of-scope items; predictions: declined. In-scope items are the negatives;
                KB-gap items are reported apart (a decline is acceptable there)
  latency       per graph node (router, calc, retrieve, answer, ...) from LangGraph's update stream,
                plus the whole turn; p50 / p90
Every answer goes to a CSV with the automatic verdict and an empty human_verdict column for spot checks.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

from eval.run_answer_eval import has_fact

HERE = Path(__file__).parent
SET = HERE / "heldout_v1.jsonl"
RESULTS = HERE / "results"


def fact_ok(answer: str, fact: str) -> bool:
    return any(has_fact(answer, alt.strip()) for alt in fact.split("|") if alt.strip())


def score(item: dict, r: dict) -> dict:
    """Automatic verdicts for one answered item (r: route, refused, answer, top ids, district...)."""
    cat, ans, refused = item["category"], r["answer"], r["refused"]
    out = {"route_ok": None, "hit1": None, "hit3": None, "verdict": "", "fee_ok": None}
    if cat == "out_of_scope":
        out["route_ok"] = refused
        out["verdict"] = "correct" if refused else "wrong"
    elif cat in ("weather", "news"):
        out["route_ok"] = r["route"] == cat
        good = out["route_ok"] and not refused             # "can't get that now" is not an answer
        if item.get("district") and cat == "weather":
            good = good and r.get("district") == item["district"]
        if item.get("day") == "tomorrow":
            good = good and ("നാളെ" in ans)
        out["verdict"] = "correct" if good else "wrong"
    else:
        out["route_ok"] = r["route"] == "service_faq"
        if cat == "in_scope":
            exp = set(item["expect_kb"])
            out["hit1"] = bool(r["top1"]) and r["top1"][0] in exp
            out["hit3"] = bool(exp & set(r["top3"]))
            if refused:
                out["verdict"] = "false_refusal"
            else:
                hits = [fact_ok(ans, f) for f in item["key_facts"]]
                bad = any(fact_ok(ans, m) for m in item.get("must_not", []))
                out["verdict"] = "correct" if all(hits) and not bad else ("partial" if any(hits) and not bad else "wrong")
            fee = item.get("expect_fee")
            if fee and fee.replace(",", "").isdigit():
                out["fee_ok"] = (not refused) and has_fact(ans, fee) and \
                    not any(fact_ok(ans, m) for m in item.get("must_not", []))
        else:                                                   # kb_gap: by hand
            out["verdict"] = "declined" if refused else "answered (check by hand)"
    return out


def weather_district(question: str) -> str:
    """The district the weather tool uses for this question (same lookup as tools.weather)."""
    from helpdesk import tools
    place = tools.find_place(question)                     # (name, district, lat, lon)
    return place[1] if place else tools.find_district(question) or tools.DEFAULT_DISTRICT


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarise(items: list[dict], rows: list[dict]) -> dict:
    by_id = {i["id"]: i for i in items}
    s = defaultdict(lambda: [0, 0])
    for r in rows:
        it = by_id[r["id"]]
        for key in ("route_ok", "hit1", "hit3", "fee_ok"):
            if r[key] not in (None, ""):
                s[key][0] += r[key] in (True, "True")
                s[key][1] += 1
        s[f"answer_{it['category']}"][0] += r["verdict"] == "correct"
        s[f"answer_{it['category']}"][1] += 1
    # refusal precision / recall over in-scope (negatives) + out-of-scope (positives)
    tp = sum(1 for r in rows if by_id[r["id"]]["category"] == "out_of_scope" and r["refused"] in (True, "True"))
    fp = sum(1 for r in rows if by_id[r["id"]]["category"] == "in_scope" and r["refused"] in (True, "True"))
    pos = sum(1 for i in items if i["category"] == "out_of_scope")
    gap = [r for r in rows if by_id[r["id"]]["category"] == "kb_gap"]
    lat = defaultdict(list)
    for r in rows:
        for k, v in json.loads(r["timings"]).items():
            lat[k].append(v)
    return {
        "n": len(rows),
        **{k: f"{a}/{b}" for k, (a, b) in sorted(s.items())},
        "refusal_precision": f"{tp}/{tp + fp}", "refusal_recall": f"{tp}/{pos}",
        "kb_gap": {"declined": sum(r["refused"] in (True, "True") for r in gap), "answered": sum(r["refused"] not in (True, "True") for r in gap)},
        "latency_s": {k: {"p50": round(pct(v, .5), 2), "p90": round(pct(v, .9), 2), "n": len(v)} for k, v in sorted(lat.items())},
    }


def setup_llm(mode: str, threads: int):
    from helpdesk import llm
    if mode == "none":
        os.environ["HELPDESK_LLM"] = "none"
        return {}
    if mode == "http":
        return {}
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    path = os.getenv("GEMMA_GGUF") or hf_hub_download("ggml-org/gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-Q4_0.gguf")
    # n_threads_batch (prefill) defaults to every core in llama-cpp-python: pin both to the Space's 2.
    llama = Llama(model_path=path, n_ctx=2048, n_threads=threads, n_threads_batch=threads, verbose=False)
    llm.use_llama_cpp(llama)
    return {"model": Path(path).name, "threads": threads}


def run(items: list[dict], args) -> list[dict]:
    os.environ["HELPDESK_KB_AUTO"] = "1" if args.kb == "expanded" else "0"
    os.environ["HELPDESK_DENSE"] = "1" if args.dense else "0"
    os.environ.setdefault("HELPDESK_CANON", "1")
    os.environ.setdefault("HELPDESK_CTX_K", "1")
    meta = setup_llm(args.llm, args.threads)
    from helpdesk import tools
    from helpdesk.graph import build_graph
    from helpdesk.retriever import Retriever
    t = time.perf_counter()
    retr = Retriever(dense=args.dense)
    graph = build_graph(retr)
    print(f"index built in {time.perf_counter() - t:.1f}s; {len(retr.docs)} passages; {meta}", flush=True)
    if args.llm == "inproc":                                    # one warm-up turn, not scored
        graph.invoke({"query": "ജനന സർട്ടിഫിക്കറ്റ് എങ്ങനെ കിട്ടും?"})
    rows = []
    for it in items:
        state, timings = {"query": it["question"]}, {}
        t0 = t = time.perf_counter()
        for upd in graph.stream({"query": it["question"]}, stream_mode="updates"):
            now = time.perf_counter()
            for node, delta in upd.items():
                timings[node] = round(timings.get(node, 0) + now - t, 3)
                state.update(delta or {})
            t = now
        timings["total"] = round(time.perf_counter() - t0, 3)
        srcs = [x.get("id") for x in state.get("sources") or [] if x.get("id")]
        top3 = [h.doc["id"] for h in retr.search(it["question"], k=3)] if it["category"] in ("in_scope", "kb_gap") else []
        r = {"id": it["id"], "category": it["category"], "style": it["style"], "question": it["question"],
             "expected_route": it["route"], "route": state.get("route"), "route_by": state.get("route_by"),
             "tool": state.get("tool"), "refused": bool(state.get("refused")),
             "top1": srcs[:1], "top3": top3,
             "top1_title": (state.get("sources") or [{}])[0].get("title", "") if srcs else "",
             "district": weather_district(it["question"]) if state.get("route") == "weather" else "",
             "expect_kb": json.dumps(it["expect_kb"], ensure_ascii=False), "expect_fee": it.get("expect_fee") or "",
             "answer": state.get("answer", ""), "timings": json.dumps(timings)}
        r.update(score(it, r))
        r["human_verdict"] = ""
        rows.append(r)
        print(f"{r['id']} {r['verdict']:24} {r['route'] or '-':12} {','.join(r['top1']) or '-':22} {timings['total']:6.2f}s", flush=True)
    return rows


def write(rows: list[dict], items: list[dict], tag: str, info: dict) -> dict:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"heldout_{tag}.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for k, v in r.items()})
    summary = {"tag": tag, **info, **summarise(items, rows)}
    (RESULTS / f"heldout_{tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", choices=["old", "expanded"], default="old")
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--llm", choices=["inproc", "http", "none"], default="inproc")
    ap.add_argument("--threads", type=int, default=int(os.getenv("N_THREADS", "2")))
    ap.add_argument("--tag", default="")
    ap.add_argument("--only", help="comma-separated ids")
    ap.add_argument("--set", default=str(SET), help="another file in the held-out schema (smoke tests)")
    ap.add_argument("--rescore", help="saved heldout_<tag>.csv: re-apply the verdicts, no graph")
    args = ap.parse_args()
    items = [json.loads(l) for l in Path(args.set).open(encoding="utf-8") if l.strip()]
    if args.only:
        keep = set(args.only.split(","))
        items = [i for i in items if i["id"] in keep]
    if args.rescore:
        by_id = {i["id"]: i for i in items}
        with open(args.rescore, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["refused"] = r["refused"] == "True"
            r["top1"], r["top3"] = json.loads(r["top1"]), json.loads(r["top3"])
            # Older CSVs stored a place's latitude here (index bug); only those rows are looked up again,
            # and against the PLACES list of that run's code (a latitude only came from a hand-written place).
            if r["route"] == "weather" and r["district"].replace(".", "").isdigit():
                r["district"] = weather_district(r["question"])
            r.update(score(by_id[r["id"]], r))
        write(rows, [by_id[r["id"]] for r in rows], Path(args.rescore).stem.removeprefix("heldout_") + "_rescored", {})
        return
    tag = args.tag or f"{args.kb}_{'dense' if args.dense else 'bm25'}_{args.llm}"
    rows = run(items, args)
    write(rows, items, tag, {"kb": args.kb, "dense": args.dense, "llm": args.llm, "threads": args.threads,
                             "canon": os.environ.get("HELPDESK_CANON"), "ctx_k": os.environ.get("HELPDESK_CTX_K"),
                             "auto_boost": os.environ.get("HELPDESK_AUTO_BOOST", "default")})


if __name__ == "__main__":
    main()
