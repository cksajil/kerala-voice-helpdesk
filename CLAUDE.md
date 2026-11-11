# Notes for Claude

Malayalam voice helpdesk for Kerala local-body (LSGD / K-SMART) services: Whisper (ASR) ->
`helpdesk/` reply stage (LangGraph, BM25 + optional BGE-M3, Gemma 4 E2B) -> VITS (TTS).
README.md is the running log of what changed and why; read its latest sections first.

## Running things
- No GPU in cloud sessions. Install only what the reply stage needs; `requirements.txt` pulls
  PyTorch through sentence-transformers (slow, not needed with `HELPDESK_DENSE=0`):
  `pip install langgraph rank_bm25 feedparser requests pytest`
- Checks before pushing: `HELPDESK_DENSE=0 python -m pytest -q`, `python -m eval.run_eval`.
- ASR checks without audio: `python -m eval.asr_bias --from-log` (24 logged clips),
  `python -m eval.asr_robust` (synthetic ASR slips; tune on `--split dev`, report `--split test` once).
- `python -m eval.run_synth` rewrites the tracked `eval/synth_report.json`.
- GPU work (Whisper on real audio, OCR) runs on Kaggle via `notebooks/`. The Kaggle CLI 2.x wants
  `KAGGLE_API_TOKEN` (or `kaggle auth login`); the old `KAGGLE_USERNAME`/`KAGGLE_KEY` were not accepted.
- Visitor audio is private: never commit `eval/asr_audio/` or send it to an API.

## The Space
- A separate Hugging Face Space runs the app (`REPLY_ENGINE=helpdesk`); it holds its own copy of
  `helpdesk/` and `data/`. Changes reach it only when files are copied/uploaded there after merging.
- Space logs are the main source of bugs: most wrong answers came from the wrong passage or an ASR
  mishearing, not from Gemma misreading a correct passage.

## ASR correction (`helpdesk/asr_snap.py`, PR #1)
- Runs inside `intent.fix_asr`, after the hand-written `ASR_FIXES`. Snaps unknown words to the
  closest-sounding service/facet/title word (threshold 0.15 per sound), splits merged words that start
  with a service name, and restores a service name heard as another real word (ജനറൽ -> ജനന before
  രജിസ്ട്രേഷൻ). `HELPDESK_ASR_SNAP=0` turns it off.
- Test split: same answer under slips 86.2% -> 92.6% (p=0.15), 75.0% -> 84.8% (p=0.3); logged clips
  16 -> 18 of 24; no correctly heard question changed.
- Each guard exists because of a real failure; keep them: onset change costs extra (അനുകുതി -> നികുതി),
  service words never overwritten (cat -> dog), a service name is introduced only with a facet cue
  (food question -> dog licence), same-word ending swaps are not corrections.
- Number words are deliberately not guessed: the fee calculator computes nothing from an unreadable
  number rather than a confident wrong fee.
- Known open issue: "പനി വന്നാൽ കഴിക്കാൻ പറ്റിയ കഞ്ഞി ഏതാ?" (food, out of scope) retrieves a marriage
  passage without the LLM; on the Space it relies on Gemma answering UNSURE.

## Working with the owner
- Measure before shipping: report numbers on a held-out split, say which data was seen while tuning.
- Commit and push to the session's designated branch; open PRs only when asked.
