"""Pre-fill the LLM prompt cache with every knowledge-base passage.

On a CPU, most of the reply time is prefill: Gemma reading the system prompt
and passages. llama-cpp-python can save the model state after a prompt and
reuse it when a later prompt starts with the same tokens. Start the server
with a persistent cache:

    python -m llama_cpp.server --model gemma-4-E2B-it-Q4_0.gguf --port 8080 \
        --n_ctx 4096 --n_threads 2 --cache true --cache_type disk

then run   python -m helpdesk.warmup   once. Every question whose top passage
was warmed skips that part of the prefill. The disk cache lives in the
server's working directory (.cache/llama_cache) and survives restarts;
re-run warmup after changing the KB or the system prompt.
"""
import time

from .graph import build_graph
from .retriever import load_kb


def main():
    g = build_graph()
    docs = load_kb()
    t0 = time.perf_counter()
    for i, d in enumerate(docs, 1):
        q = (d.get("questions") or [d["title"]])[0]
        t = time.perf_counter()
        s = g.invoke({"query": q})
        top = s.get("sources", [{}])[0].get("id")
        print(f"[{i}/{len(docs)}] {d['id']:14} top={top} {time.perf_counter() - t:5.1f}s")
    print(f"warm-up done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
