# Kerala Voice Helpdesk — reply stage (v1, day 1)

LangGraph agent that answers Malayalam questions about Kerala local-body
(LSGD / K-SMART) services, today's weather and today's news. It is the reply
stage between Whisper v2 (ASR) and VITS (TTS) in the voice assistant.

```
router ──► weather (Open-Meteo) ──┐
       ├─► news (RSS headlines) ──┤
       ├─► retrieve (BM25+BGE-M3)─┼─► answer ─► END
       └─► other (scope message) ─┘
```

Design choices
- Routing is explicit (keyword stems first, LLM label only if no rule fires):
  Gemma 4 E2B is unreliable at free-form tool calling.
- Weather/news numbers go out through templates, never through the LLM.
- Service answers use only retrieved passages; the LLM must reply `UNSURE`
  when the answer is not there, which becomes a refusal.
- One Malayalam normaliser (old chillu → atomic chillu, then strip ZWJ/ZWNJ)
  for the index, the query and ASR output.

## Run
```bash
pip install -r requirements.txt
python data/build_kb.py                       # writes data/kb.jsonl
python -m llama_cpp.server --model gemma-4-E2B-it-Q4_0.gguf --port 8080 --n_ctx 4096   # separate terminal
python -m helpdesk.cli "ട്രേഡ് ലൈസൻസ് പുതുക്കാൻ എന്ത് ചെയ്യണം?"
```
No LLM? `HELPDESK_LLM=none` reads out the best passage instead.
No dense model? `HELPDESK_DENSE=0` uses BM25 only.

## Faster replies on CPU (prompt cache)
Most reply time on a CPU is prefill (reading system prompt + passages).
Start the server with a disk cache, then warm it once:
```bash
python -m llama_cpp.server --model gemma-4-E2B-it-Q4_0.gguf --port 8080 \
    --n_ctx 4096 --n_threads 2 --cache true --cache_type disk
python -m helpdesk.warmup        # one pass over the KB; re-run after KB/prompt changes
```

## Test
```bash
HELPDESK_DENSE=0 pytest -q                    # 17 offline tests
python -m eval.run_eval                       # route / hit@3 / refusal gate, BM25 only
python -m eval.run_eval --dense               # same with BGE-M3; use it to tune HELPDESK_COS_MIN
python -m eval.run_eval --dense --full        # end-to-end with LLM + live tools
```

## Status (7 Nov 2026, day 1)

**Setup:** Gemma 4 E2B (Q4_0) via llama-cpp-python, BM25 retrieval only,
2 passages per answer, Intel MacBook Air (2 cores).

| Check | Result |
|---|---|
| Routing (keyword rules) | 20/20 |
| Retrieval hit@1 / hit@2 | 14/15, 15/15 (after adding example questions) |
| Out-of-scope questions refused | 2/2 (ration card, passport) |
| End-to-end correct, first run | 15/20 |

**What failed and why**
- 3 timeouts (120 s): CPU prefill is the bottleneck. A prompt is ~340 tokens,
  of which the 112-token system prompt is reused from cache.
- 2 wrong fees: Gemma mapped "2 months" and "₹10 lakh" to the wrong fee band.
  Fix planned: compute fee bands in code (a tool), not in the LLM.
- 1 wrong passage: fixed by indexing example questions per entry.

**Latency note:** the MacBook throttled to 66% CPU speed under sustained load
(`pmset -g therm`), so latency will be measured on the Hugging Face Space.

**Next:** integrate into the voice-assistant Space, a fee-calculator tool,
expand the KB to 50+ entries, and a second held-out test set.
