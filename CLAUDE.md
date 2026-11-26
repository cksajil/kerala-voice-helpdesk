# Notes for Claude

## Where to run compute
- Run any CPU- or GPU-heavy work (model inference, ASR/LLM evals, replays with Gemma, training) on the
  owner's **Kaggle**, through the `kaggle` CLI / API (`KAGGLE_USERNAME`, `KAGGLE_API_TOKEN` are set in the
  cloud environment). Don't run it in the Claude Code container. Light work (reading logs, editing code,
  small scripts) is fine locally.
- Push notebooks as **private** kernels (`kaggle kernels push`, `"is_private": true`), CPU unless a GPU is
  needed; fetch results with `kaggle kernels output` (needs `www.kaggleusercontent.com`, allowed).
- **Kaggle secrets are not passed to runs started through the API.** Anything that needs `HF_TOKEN`
  (gated models such as `sajilck/whisper-small-malayalam-v2` or `ai4bharat/indic-conformer-600m-multilingual`,
  the private usage log `sajilck/tts-demo-evidence-log`) must be started by the owner in the Kaggle editor:
  Add-ons → Secrets → `HF_TOKEN`, then Save Version → **Save & Run All** (Quick Save does not run).
  Push the notebook, ask the owner to run it, then fetch the output.

## Privacy
- Voice recordings from the usage log (the owner's and the owner's wife's) are private: don't upload them to Kaggle,
  public repos, notebooks or eval sets without asking first. Prefer public corpus clips
  (`sajilck/malayalam-asr-corpus`) when a test only needs audio of a given length.

## ASR on the Space
- Since 12 Nov 2026 the Space uses IndicConformer-600M RNNT with chillu restoration (`asr_chillu.py`),
  `ASR_ENGINE=whisper` switches back to Whisper v2 int8. IndicConformer often writes chillus as
  consonant + virama without ZWJ; the help desk's normaliser and `mlasr_eval`'s `norm_ml` don't map that
  form, so compare its output only after `restore_chillus`.

## State after v1 (26 Nov 2026)
- v1 is live and announced on LinkedIn (26 Nov). `main` and the Space (commit 848461d) run the same help-desk
  code. The open items after v1 are at the end of `docs/ENGINEERING_LOG.md` (widow-pension ranking, on-time
  birth-registration entry, building rules, held-out v2 from real users, learned router, fine-tuning).
- Owner's records: Google Docs "Kerala Voice Helpdesk v1 – Engineering Log & Methodology" (numbered sections;
  14 is the latest) and "Daily Report" (dated entries). Append a dated entry to both, and to
  docs/ENGINEERING_LOG.md, after each piece of work.
- Usage log `sajilck/tts-demo-evidence-log` was emptied on 26 Nov before the announcement (history kept).
  New rows are real users' questions: material for held-out v2. Never send them to an external API.
- LangSmith is for local runs only (owner's decision): the API switches tracing off unless
  HELPDESK_TRACING=1 (`helpdesk/tracing.py`); the Space has no LangSmith key.
- Docker on the owner's Mac (Docker Desktop 4 GB): run with `HELPDESK_CACHE_MB=512`. Local voice test:
  `scripts/voice_chat.py` (Whisper v2; the microphone works only through its ffmpeg fallback there).
- `test_12_nov_retrieval_fixes` needs HELPDESK_CANON=1 when run alone (the full suite sets it via helpdesk.api).
- Ranking changes must be measured on held-out v1 before adopting: the title boost (HELPDESK_TITLE_BOOST,
  default 0) helped tuned sets but not held-out, so it stays off.
- Demo video: `docs/video/` (build scripts, marks for the owner's recording; the recording itself is not
  in the repo). This session cannot push to branches other than its own or delete branches: ask the owner.
