"""Pre-fill the LLM prompt cache with every knowledge-base passage.

On a CPU most of the reply time is prefill (Gemma reading the system prompt
and the passage). llama-cpp-python can save the model state after a prompt and
reuse it when a later prompt starts with the same tokens.

Use the RAM cache, not the disk cache: in llama-cpp-python 0.3.x the disk
cache removes an entry when it is read (LlamaDiskCache.__getitem__ pops it),
and since every prompt shares the system prompt, each warm-up request evicts
the previous one. The RAM cache keeps entries (LRU), so warm up after each
server start:

    python -m llama_cpp.server --model gemma-4-E2B-it-Q4_0.gguf --port 8080 \\
        --n_ctx 4096 --n_threads 2 --cache true --cache_type ram
    HELPDESK_CTX_K=1 python -m helpdesk.warmup

With HELPDESK_CTX_K=1 every answer prompt is 'system prompt + one passage +
question', so warming one prompt per KB entry covers every question.
Each warm-up request generates a single token: only the prefill matters.
"""
import time

from . import llm
from .graph import ANSWER_PROMPT, CTX_K, answer_message, passage
from .retriever import load_kb


def main():
    if CTX_K != 1:
        print(f"note: HELPDESK_CTX_K={CTX_K}; warm-up covers single-passage prompts only, "
              "so most answers will not hit the cache. Use HELPDESK_CTX_K=1.")
    docs = load_kb()
    t0 = time.perf_counter()
    for i, d in enumerate(docs, 1):
        t = time.perf_counter()
        llm.chat(ANSWER_PROMPT, answer_message(passage(d), d["title"]), max_tokens=1, temperature=0)
        print(f"[{i}/{len(docs)}] {d['id']:14} {time.perf_counter() - t:5.1f}s", flush=True)
    print(f"warm-up done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
