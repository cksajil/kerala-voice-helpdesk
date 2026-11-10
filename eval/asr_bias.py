"""Does a Whisper initial_prompt (service vocabulary) help the helpdesk? Measured, not guessed.

    python -m eval.asr_bias --audio-dir eval/asr_audio            # every prompt in eval/asr_prompts.txt
    python -m eval.asr_bias --from-log                            # no audio: score the logged transcripts
    python -m eval.asr_bias --wer-dataset sajilck/malayalam-asr-corpus --wer-clean-csv own_corpus_test_clean.csv
(notebooks/kaggle_asr_bias.ipynb runs all of this on Kaggle.)

For each prompt it transcribes the logged questions (eval/asr_clips.csv: clip id, what the Space heard,
the reference text) with the Space's settings and reports
  CER           character error rate against the reference (normalised text)
  terms         share of the reference's service words (ജനന, ഫീസ്, കാലാവധി...) found in the transcript
  same answer   the helpdesk routes the transcript as it routes the reference: same route, same tool,
                same top passage. This is what the visitor feels.
A prompt is worth shipping only if "same answer" goes up here AND WER on the general held-out test set
(github.com/cksajil/malayalam-asr-eval) does not get worse: these clips are few and were used to write
the prompts, so they cannot show harm on other speech by themselves.

Audio stays local (it is visitors' speech): never commit eval/asr_audio/, never send it to an API.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
CLIPS, PROMPTS = HERE / "asr_clips.csv", HERE / "asr_prompts.txt"
REPO = "sajilck/whisper-small-malayalam-v2"
CT2 = Path(os.getenv("CT2_CACHE", Path.home() / ".cache" / "ct2")) / f"{REPO.replace('/', '__')}-int8"
# Stems of the words a wrong hearing costs an answer for (10 Nov logs: ജനന -> ജനറൽ, ഫീസ് -> പീസ്/ഈസ്/ഭീസ്,
# കാലാവധി -> കാലാവരി, ഇരുപത്തൊന്ന് -> ഇരുപത്തെന്ന).
TERMS = ["ജനന", "മരണ", "വിവാഹ", "ഫീസ", "കാലാവധി", "പെർമിറ്റ", "പെൻഷൻ", "വിധവ", "രജിസ്ട്രേഷ", "സർട്ടിഫിക്കറ്റ",
         "ദിവസ", "ഇരുപത്തൊന്ന", "മുപ്പത", "നാല്പത", "കെട്ടിട", "തുക", "രേഖ", "കാലാവസ്ഥ", "വാർത്ത"]


def cer(ref: str, hyp: str) -> tuple[int, int]:
    """(edit distance, reference length) on characters, spaces kept."""
    a, b = ref, hyp
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1], len(a)


def terms_of(text: str) -> set[str]:
    words = text.split()
    return {t for t in TERMS if any(w.startswith(t) for w in words)}


def load_model():
    from faster_whisper import WhisperModel
    if not (CT2 / "model.bin").exists():
        print(f"converting {REPO} to CTranslate2 int8 in {CT2} (one-off, ~1 min)", flush=True)
        code = ("import sys\nfrom ctranslate2.converters import TransformersConverter\n"
                "from transformers import WhisperFeatureExtractor, WhisperTokenizerFast\n"
                "class C(TransformersConverter):\n"            # newer ctranslate2 passes dtype=, older
                "    def load_model(self, cls, name, **kw):\n"  # transformers only knows torch_dtype
                "        try:\n            return cls.from_pretrained(name, **kw)\n"
                "        except TypeError:\n            kw['torch_dtype'] = kw.pop('dtype', None)\n"
                "            return cls.from_pretrained(name, **kw)\n"
                "r, o = sys.argv[1], sys.argv[2]\nC(r).convert(o, quantization='int8', force=True)\n"
                "WhisperTokenizerFast.from_pretrained(r).save_pretrained(o)\n"
                "WhisperFeatureExtractor.from_pretrained(r).save_pretrained(o)\n")
        CT2.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-c", code, REPO, str(CT2)], check=True)
    m = WhisperModel(str(CT2), device="cpu", compute_type="int8")
    m.max_length = 888                              # as on the Space: lifts the 224-token cap
    return m


def transcribe(model, audio, prompt: str | None) -> str:
    """Same settings as the Space (app.py WhisperCT2), including its max_new_tokens rule.
    audio: a file path, or a 16 kHz float array."""
    n = len(model.hf_tokenizer.encode(" " + prompt, add_special_tokens=False).ids) if prompt else 0
    new = 888 - 4 - ((1 + min(n, 888 // 2 - 1)) if n else 0)
    if isinstance(audio, (str, Path)):
        import librosa
        wav, _ = librosa.load(str(audio), sr=16000, mono=True)
    else:
        wav = audio
    segs, _ = model.transcribe(wav[: 30 * 16000], language="ml", task="transcribe", beam_size=1, best_of=1,
                               temperature=0.0, condition_on_previous_text=False, without_timestamps=True,
                               vad_filter=False, max_new_tokens=new, initial_prompt=prompt or None)
    return " ".join(s.text.strip() for s in segs).strip()


def wer_set(name: str, split: str, per_source: int, clean_csv: Path | None):
    """(16 kHz arrays, reference texts) for the general WER check, streamed from a Hugging Face dataset
    with an audio column, a text column and (optionally) a 'source' column: up to per_source clips per
    source, and only sentences listed in clean_csv (own_corpus_test_clean.csv, the leakage-free rows)
    when given. Used to compare prompts with each other, not to reproduce the 231-clip leaderboard."""
    import numpy as np
    from datasets import Audio, load_dataset
    keep = None
    if clean_csv:
        rows = list(csv.DictReader(clean_csv.open(encoding="utf-8")))
        col = next(c for c in ("sentence", "text", "transcription") if c in rows[0])
        keep = {r[col].strip() for r in rows}
    ds = load_dataset(name, split=split, streaming=True)
    feats = ds.features or {}
    acol = next((c for c, f in feats.items() if isinstance(f, Audio)), "audio")
    tcol = next(c for c in ("sentence", "text", "transcription", "transcript") if c in feats)
    ds = ds.cast_column(acol, Audio(sampling_rate=16000))
    arrays, texts, count = [], [], {}
    for row in ds:
        src = row.get("source", "all")
        if count.get(src, 0) >= per_source or (keep is not None and row[tcol].strip() not in keep):
            continue
        a = row[acol]
        arr = a["array"] if isinstance(a, dict) else a.get_all_samples().data.mean(0).numpy()
        arrays.append(np.asarray(arr, dtype=np.float32))
        texts.append(row[tcol])
        count[src] = count.get(src, 0) + 1
        if len(count) >= 6 and all(v >= per_source for v in count.values()):
            break
    print(f"general WER set: {name}[{split}] {len(arrays)} clips {count}", flush=True)
    return arrays, texts


def make_router():
    """The helpdesk as on the Space (BM25, canon as set in the environment), with no LLM and no network."""
    os.environ["HELPDESK_LLM"] = "none"
    os.environ.setdefault("HELPDESK_DENSE", "0")
    from helpdesk import tools

    def offline(*a, **k):
        raise RuntimeError("offline")
    tools.weather = tools.news = offline
    from helpdesk.graph import build_graph
    g = build_graph()

    def outcome(q: str) -> tuple:
        s = g.invoke({"query": q})
        top = (s.get("sources") or [{}])[0].get("id")
        return s["route"], s.get("tool") or "", top
    return outcome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", type=Path, default=HERE / "asr_audio",
                    help="folder with <id>_in.wav from the usage-log dataset (searched recursively)")
    ap.add_argument("--from-log", action="store_true", help="score the logged transcripts only (no ASR)")
    ap.add_argument("--show", type=int, default=0, help="print this many changed transcripts per prompt")
    ap.add_argument("--wer-dataset", default="", help="HF dataset for the general WER check (no harm elsewhere)")
    ap.add_argument("--wer-split", default="test")
    ap.add_argument("--wer-per-source", type=int, default=40)
    ap.add_argument("--wer-clean-csv", type=Path, default=None, help="own_corpus_test_clean.csv: leakage-free rows")
    ap.add_argument("--out", type=Path, default=HERE / "asr_bias_report.json")
    args = ap.parse_args()
    from helpdesk.normalise import normalise

    clips = [r for r in csv.DictReader(CLIPS.open(encoding="utf-8")) if r["reference"].strip()]
    prompts = [l.strip() for l in PROMPTS.open(encoding="utf-8") if l.strip() and not l.startswith("#")]
    outcome = make_router()
    target = {c["id"]: outcome(c["reference"]) for c in clips}

    runs: dict[str, dict[str, str]] = {}
    if args.from_log:
        runs["logged (Space, no prompt)"] = {c["id"]: c["heard_v0"] for c in clips}
    else:
        wavs = {p.name.split("_")[0]: p for p in args.audio_dir.rglob("*_in.wav")}
        missing = [c["id"] for c in clips if c["id"] not in wavs]
        if missing:
            print(f"{len(missing)} clips have no audio in {args.audio_dir} and are skipped: {missing[:5]}")
        clips = [c for c in clips if c["id"] in wavs]
        model = load_model()
        for p in prompts:
            prompt = None if p == "none" else p
            runs[p] = {c["id"]: transcribe(model, wavs[c["id"]], prompt) for c in clips}
            print(f"transcribed {len(clips)} clips with prompt {p[:30]!r}", flush=True)

    general = {}
    if args.wer_dataset and not args.from_log:
        arrays, texts = wer_set(args.wer_dataset, args.wer_split, args.wer_per_source, args.wer_clean_csv)
        for p in prompts:
            prompt = None if p == "none" else p
            e = n = 0
            for arr, ref in zip(arrays, texts):
                r = normalise(ref, True).split()
                d, L = cer(r, normalise(transcribe(model, arr, prompt), True).split())   # same DP on words
                e, n = e + d, n + L
            general[p] = 100 * e / max(n, 1)
            print(f"general WER {general[p]:.1f}% with prompt {p[:30]!r}", flush=True)

    print(f"{len(clips)} clips; {sum(c['check'] == 'verify' for c in clips)} references still marked verify\n")
    print(f"{'prompt':48} {'CER':>6} {'terms':>8} {'same answer':>12} {'general WER':>12}")
    report = []
    for name, hyps in runs.items():
        e = n = th = tt = same = 0
        changed = []
        for c in clips:
            ref, hyp = normalise(c["reference"], True), normalise(hyps[c["id"]], True)
            d, L = cer(ref, hyp)
            e, n = e + d, n + L
            want = terms_of(ref)
            th, tt = th + len(want & terms_of(hyp)), tt + len(want)
            ok = outcome(hyps[c["id"]]) == target[c["id"]]
            same += ok
            if hyps[c["id"]] != c["heard_v0"]:
                changed.append((ok, c["heard_v0"], hyps[c["id"]]))
        label = name if len(name) <= 46 else name[:43] + "..."
        gw = f"{general[name]:.1f}%" if name in general else "-"
        print(f"{label:48} {100 * e / max(n, 1):5.1f}% {th:>3}/{tt:<4} {same:>5}/{len(clips)} {gw:>12}")
        report.append({"prompt": name, "cer": round(100 * e / max(n, 1), 2), "terms": [th, tt],
                       "same_answer": [same, len(clips)], "general_wer": general.get(name),
                       "transcripts": hyps})
        for ok, old, new in changed[:args.show]:
            print(f"    {'ok ' if ok else 'BAD'} {old}  ->  {new}")

    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
