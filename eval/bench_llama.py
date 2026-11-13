"""llama-cpp-python vs llama.cpp's own llama-server on identical prompts (Phase 3).

    python -m eval.bench_llama --server-bin ./llama-server --gguf gemma-4-E2B-it-Q4_0.gguf --threads 2

Prompts: the help desk's real answer prompts (ANSWER_PROMPT + top-1 passage + question) for the
held-out set's service questions, rendered once with the GGUF's chat template, so both back ends get
the same text. Each prompt runs greedy with the answer cap (HELPDESK_ANSWER_TOKENS, 100) in two modes:
  cold   no reuse of earlier prompts (Python: llama.reset(); server: cache_prompt=false)
  warm   prefix reuse (Python's default; server: cache_prompt=true). The system prompt is shared, as
         on the Space, so most of it is cached after the first request.
Per request: client wall time, prompt/completion tokens, and llama.cpp's own compute time
(server: timings.prompt_ms + predicted_ms; Python: llama_perf context timings). Overhead = wall -
compute. Python also times create_chat_completion (template rendered per call) on the same messages.
Back ends run one after the other, never at the same time. Results: eval/results/bench_llama.json.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import requests

HERE = Path(__file__).parent


def prompts(limit: int) -> list[list[dict]]:
    os.environ.setdefault("HELPDESK_CANON", "1")
    os.environ.setdefault("HELPDESK_CTX_K", "1")
    os.environ["HELPDESK_DENSE"] = "0"
    from helpdesk.graph import ANSWER_PROMPT, answer_message, passage
    from helpdesk.intent import fix_asr
    from helpdesk.retriever import Retriever
    retr = Retriever(dense=False)
    items = [json.loads(l) for l in (HERE / "heldout_v1.jsonl").open(encoding="utf-8")]
    out = []
    for it in items:
        if it["category"] not in ("in_scope", "kb_gap"):
            continue
        hits = retr.search(it["question"], k=1)
        if hits:
            out.append([{"role": "system", "content": ANSWER_PROMPT},
                        {"role": "user", "content": answer_message(passage(hits[0].doc), fix_asr(it["question"], snap=False))}])
    return out[:limit]


def stats(xs):
    xs = sorted(xs)
    if not xs:
        return {}
    q = lambda p: xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]
    return {"mean": round(statistics.mean(xs), 4), "p50": round(q(.5), 4), "p90": round(q(.9), 4), "n": len(xs)}


def python_side(gguf: str, threads: int, texts: list[str], chats: list[list[dict]], max_tokens: int) -> dict:
    import llama_cpp
    from llama_cpp import Llama
    t = time.perf_counter()
    llama = Llama(model_path=gguf, n_ctx=2048, n_threads=threads, n_threads_batch=threads, verbose=False)
    load = time.perf_counter() - t

    def compute_s() -> float | None:
        try:
            d = llama_cpp.llama_perf_context(llama._ctx.ctx)
            return (d.t_p_eval_ms + d.t_eval_ms) / 1000
        except Exception:
            return None

    res = {"version": llama_cpp.__version__, "load_s": round(load, 2)}
    for mode in ("cold", "warm"):
        rows = []
        for p in texts:
            if mode == "cold":
                llama.reset()
            try:
                llama_cpp.llama_perf_context_reset(llama._ctx.ctx)
            except Exception:
                pass
            t = time.perf_counter()
            out = llama.create_completion(p, max_tokens=max_tokens, temperature=0, top_k=1)
            wall = time.perf_counter() - t
            comp = compute_s()
            rows.append({"wall": wall, "compute": comp, "prompt_tokens": out["usage"]["prompt_tokens"],
                         "completion_tokens": out["usage"]["completion_tokens"], "text": out["choices"][0]["text"]})
        res[mode] = rows
    rows = []
    for m in chats:                                            # the Space's call path (template per call)
        llama.reset()
        t = time.perf_counter()
        out = llama.create_chat_completion(messages=m, max_tokens=max_tokens, temperature=0, top_k=1)
        rows.append({"wall": time.perf_counter() - t, "prompt_tokens": out["usage"]["prompt_tokens"],
                     "completion_tokens": out["usage"]["completion_tokens"]})
    res["chat_cold"] = rows
    del llama
    return res


def server_side(binary: str, gguf: str, threads: int, texts: list[str], max_tokens: int, port: int = 8091) -> dict:
    proc = subprocess.Popen([binary, "-m", gguf, "-c", "2048", "-t", str(threads), "-tb", str(threads),
                             "--port", str(port), "-np", "1", "--no-webui"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    url = f"http://127.0.0.1:{port}"
    t = time.perf_counter()
    try:
        while time.perf_counter() - t < 600:
            try:
                if requests.get(url + "/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            if proc.poll() is not None:
                raise RuntimeError(proc.stdout.read()[-3000:])
            time.sleep(1)
        load = time.perf_counter() - t
        version = requests.get(url + "/props", timeout=10).json().get("build_info", "")
        res = {"version": version, "load_s": round(load, 2)}
        for mode in ("cold", "warm"):
            rows = []
            for p in texts:
                t = time.perf_counter()
                r = requests.post(url + "/completion", timeout=600, json={
                    "prompt": p, "n_predict": max_tokens, "temperature": 0, "top_k": 1,
                    "cache_prompt": mode == "warm"}).json()
                wall = time.perf_counter() - t
                tm = r.get("timings", {})
                rows.append({"wall": wall, "compute": (tm.get("prompt_ms", 0) + tm.get("predicted_ms", 0)) / 1000,
                             "prompt_tokens": tm.get("prompt_n"), "prompt_cached": r.get("tokens_cached"),
                             "completion_tokens": tm.get("predicted_n"), "text": r.get("content", "")})
            res[mode] = rows
        return res
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


def summarise(side: dict) -> dict:
    out = {"version": side["version"], "load_s": side["load_s"]}
    for mode in ("cold", "warm", "chat_cold"):
        rows = side.get(mode)
        if not rows:
            continue
        s = {"wall_s": stats([r["wall"] for r in rows]),
             "completion_tokens": stats([r["completion_tokens"] or 0 for r in rows]),
             "prompt_tokens": stats([r["prompt_tokens"] or 0 for r in rows])}
        comp = [r["wall"] - r["compute"] for r in rows if r.get("compute")]
        if comp:
            s["overhead_s"] = stats(comp)
            s["compute_s"] = stats([r["compute"] for r in rows if r.get("compute")])
        out[mode] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-bin", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()
    max_tokens = int(os.getenv("HELPDESK_ANSWER_TOKENS", "100"))
    chats = prompts(args.limit)
    from llama_cpp import Llama
    tmpl = Llama(model_path=args.gguf, n_ctx=256, vocab_only=True, verbose=False)
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter
    fmt = Jinja2ChatFormatter(template=tmpl.metadata["tokenizer.chat_template"],
                              eos_token=tmpl._model.token_get_text(tmpl.token_eos()),
                              bos_token=tmpl._model.token_get_text(tmpl.token_bos()))
    # The BOS token is added by both back ends' tokenizers, so drop the template's own.
    bos = tmpl._model.token_get_text(tmpl.token_bos())
    texts = [fmt(messages=m).prompt.removeprefix(bos) for m in chats]
    del tmpl
    py = python_side(args.gguf, args.threads, texts, chats, max_tokens)
    sv = server_side(args.server_bin, args.gguf, args.threads, texts, max_tokens)
    same = sum(a["text"] == b["text"] for a, b in zip(py["cold"], sv["cold"]))
    report = {"n_prompts": len(texts), "threads": args.threads, "max_tokens": max_tokens,
              "identical_outputs_cold": f"{same}/{len(texts)}",
              "python": summarise(py), "server": summarise(sv)}
    for mode in ("cold", "warm"):
        d = [a["wall"] - b["wall"] for a, b in zip(py[mode], sv[mode])]
        report[f"python_minus_server_wall_{mode}"] = stats(d)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "bench_llama.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    (HERE / "results" / "bench_llama_raw.json").write_text(json.dumps({"python": py, "server": sv}, ensure_ascii=False))
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
