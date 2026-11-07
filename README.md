# Kerala Voice Helpdesk — reply stage (v1, day 1)

LangGraph agent that answers Malayalam questions about Kerala local-body
(LSGD / K-SMART) services, today's weather and today's news. It is the reply
stage between Whisper v2 (ASR) and VITS (TTS) in the voice assistant.

```
router ─► weather (Open-Meteo) ──────────────────────► answer ─► END
       ├► news (RSS headlines) ──────────────────────► answer
       ├► calc (fee bands, in code) ─────────────────────────► END
       │     └─ not a fee question ─► retrieve (BM25+dense) ─► answer
       └► other (scope message) ─────────────────────────────► END
```

Design choices
- Routing is explicit (keyword stems first, LLM label only if no rule fires):
  Gemma 4 E2B is unreliable at free-form tool calling.
- Weather/news numbers go out through templates, never through the LLM.
- Fee questions that need band arithmetic (late birth/death registration,
  licence fee by investment and sector) are computed in code: Gemma got
  these wrong, and the calculator needs no LLM call, so it answers instantly.
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
    --n_ctx 4096 --n_threads 2 --cache true --cache_type ram
HELPDESK_CTX_K=1 python -m helpdesk.warmup    # after every server start
```
Use `ram`, not `disk`: llama-cpp-python 0.3.x's disk cache deletes an entry
when it is read, so warm-up entries evict each other. With `HELPDESK_CTX_K=1`
every prompt is system prompt + one passage + question, so one warm-up per
KB entry covers all questions.
```
```

## Test
```bash
HELPDESK_DENSE=0 pytest -q                    # 17 offline tests
python -m eval.run_eval                       # route / hit@3 / refusal gate, BM25 only
python -m eval.run_eval --dense               # same with BGE-M3; use it to tune HELPDESK_COS_MIN
python -m eval.run_eval --dense --full        # end-to-end with LLM + live tools
```

## Status (7 Nov 2026)
- BM25 only, offline: route 20/20, retrieval hit@3 15/15, district 2/2.
- Known gap: the BM25 score cannot tell answerable from unanswerable
  questions (ration card and passport questions score as high as real ones),
  so the retrieval gate lets them through: 15/17. Refusal for these depends
  on the dense cosine threshold and the LLM's `UNSURE`; measure with `--full`.
- Knowledge base: 15 entries, each with example questions that are indexed
  for retrieval but never sent to the LLM (written independently of the eval set).
- Knowledge base: Entries with `verify: true` come from a 2018
  SOP or a secondary site; check them against current official pages.
- Open-Meteo free API is non-commercial only.
