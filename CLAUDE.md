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
- Voice recordings from the usage log (the owner's and his wife's) are private: don't upload them to Kaggle,
  public repos, notebooks or eval sets without asking first. Prefer public corpus clips
  (`sajilck/malayalam-asr-corpus`) when a test only needs audio of a given length.

## ASR on the Space
- Since 12 Nov 2026 the Space uses IndicConformer-600M RNNT with chillu restoration (`asr_chillu.py`),
  `ASR_ENGINE=whisper` switches back to Whisper v2 int8. IndicConformer often writes chillus as
  consonant + virama without ZWJ; the help desk's normaliser and `mlasr_eval`'s `norm_ml` don't map that
  form, so compare its output only after `restore_chillus`.
