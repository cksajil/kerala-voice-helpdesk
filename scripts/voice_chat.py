"""Talk to the help desk on your own computer: microphone -> Whisper -> local API -> Malayalam voice.

    python3 scripts/voice_chat.py                  # the Docker API at http://127.0.0.1:8000
    python3 scripts/voice_chat.py --no-speak       # print the answer, no voice
    python3 scripts/voice_chat.py --file q.wav     # one recorded question instead of the microphone

Press Enter, ask your question, press Enter again (up to 30 s). Ctrl+C quits. Follow-ups work:
each reply's "context" goes back with the next question, as in scripts/smoke_test.py.

Everything runs on this computer; nothing is saved unless you pass --save-dir. The models are the
Space's: sajilck/whisper-small-malayalam-v2 (its pre-converted CTranslate2 int8 copy, ~250 MB, gated:
`huggingface-cli login` once with a token that has accepted its terms) and the VITS voice
sajilck/mms-tts-malayalam-sajil (~150 MB). Same decoding settings as the Space's Whisper path
(greedy, language ml, no timestamps, 444-token budget). The Space's default ASR, IndicConformer-600M,
is not used here: it is a 2.4 GB ONNX model, too much next to Docker on an 8 GB laptop, so expect
somewhat more misheard words than on the Space.

Install (outside Docker, Python 3.11 or 3.12):  pip install -r requirements-voice.txt
"""
from __future__ import annotations

import os

# CTranslate2 (Whisper) and PyTorch (the voice) each bring an OpenMP runtime (libomp). In one process on an
# Intel Mac they first hung, then (with KMP_DUPLICATE_LIB_OK) segfaulted after the first transcription
# (25 Nov). So the voice runs in its own process (--tts-worker, PyTorch only) and this one never imports torch.
os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse
import json
import shutil
import subprocess
import tempfile
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

WHISPER_REPO = "sajilck/whisper-small-malayalam-v2"
CT2_SUBDIR = "ct2-int8"
TTS_REPO = "sajilck/mms-tts-malayalam-sajil"
SR = 16000
MAX_S = 30
MAX_NEW_TOKENS, PROMPT_LEN = 444, 4         # as the Space's WhisperCT2
SPEAKING_RATE = 1.2


# ---------------------------------------------------------------- speech to text
def load_asr(threads: int):
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download
    d = Path(snapshot_download(WHISPER_REPO, allow_patterns=[f"{CT2_SUBDIR}/*"])) / CT2_SUBDIR
    m = WhisperModel(str(d), device="cpu", compute_type="int8", cpu_threads=threads, num_workers=1)
    m.max_length = 2 * MAX_NEW_TOKENS       # CTranslate2 emits max_length // 2 tokens
    return m


def transcribe(model, wav: np.ndarray) -> str:
    segments, _ = model.transcribe(
        wav, language="ml", task="transcribe", beam_size=1, best_of=1, temperature=0.0,
        condition_on_previous_text=False, without_timestamps=True, vad_filter=False,
        word_timestamps=False, max_new_tokens=2 * MAX_NEW_TOKENS - PROMPT_LEN)
    return " ".join(s.text.strip() for s in segments).strip()


# ---------------------------------------------------------------- text to speech
class Voice:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, VitsModel
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(TTS_REPO)
        self.model = VitsModel.from_pretrained(TTS_REPO).eval()
        self.model.speaking_rate = SPEAKING_RATE
        self.sr = self.model.config.sampling_rate
        self.vocab = set(self.tok.get_vocab())

    def __call__(self, text: str) -> np.ndarray | None:
        from helpdesk.speech import to_speech        # digits / English -> Malayalam words, as on the Space
        gap = np.zeros(int(0.15 * self.sr), dtype=np.float32)
        chunks = []
        for part in re.split(r"(?<=[.?!।])\s+|\n+", to_speech(text)):
            kept = re.sub(r"\s+", " ", "".join(c if c in self.vocab or c == " " else " " for c in part)).strip()
            if not kept:
                continue
            ids = self.tok(kept, return_tensors="pt")
            if ids["input_ids"].shape[-1] < 2:
                continue
            with self.torch.inference_mode():
                chunks += [self.model(**ids).waveform[0].numpy().astype(np.float32), gap]
        return np.concatenate(chunks) if chunks else None


def tts_worker() -> int:
    """--tts-worker: one JSON line in ({"text", "out"}), the wav written to "out", one JSON line back."""
    import wave
    voice = Voice()
    print(json.dumps({"ready": True, "sr": voice.sr}), flush=True)
    for line in sys.stdin:
        job = json.loads(line)
        audio = voice(job["text"])
        if audio is not None:
            with wave.open(job["out"], "wb") as w:
                w.setnchannels(1), w.setsampwidth(2), w.setframerate(voice.sr)
                w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
        print(json.dumps({"ok": audio is not None}), flush=True)
    return 0


class VoiceProcess:
    """The voice in a separate Python process (see the top of this file)."""

    def __init__(self):
        self.p = subprocess.Popen([sys.executable, __file__, "--tts-worker"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, encoding="utf-8")
        ready = self.p.stdout.readline()
        if not ready:
            raise RuntimeError("the voice process did not start (see its error above)")
        self.tmp = tempfile.mkdtemp(prefix="voice_chat_")

    def say(self, text: str) -> bool:
        out = str(Path(self.tmp) / "reply.wav")
        self.p.stdin.write(json.dumps({"text": text, "out": out}, ensure_ascii=False) + "\n")
        self.p.stdin.flush()
        reply = self.p.stdout.readline()
        if not reply:
            print("  (the voice process stopped; continuing without voice)")
            return False
        if json.loads(reply).get("ok"):
            play(out)
        return True


def play(path: str) -> None:
    if sys.platform == "darwin":
        subprocess.run(["afplay", path], check=False)       # CoreAudio directly, no PortAudio
    else:
        import sounddevice as sd
        from faster_whisper.audio import decode_audio
        sd.play(decode_audio(path, sampling_rate=22050), 22050)
        sd.wait()


# ---------------------------------------------------------------- audio in
def to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16 kHz (Whisper's rate). Mac microphones run at 44.1/48 kHz and CoreAudio refused
    to open them at 16 kHz (PortAudio -9986, AUHAL err -66740, 25 Nov)."""
    if sr == SR or not len(wav):
        return wav.astype(np.float32)
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(SR, sr)
        return resample_poly(wav, SR // g, sr // g).astype(np.float32)
    except ImportError:                                   # linear, after a crude low-pass (box filter)
        k = max(1, round(sr / SR))
        smooth = np.convolve(wav, np.ones(k) / k, mode="same")
        t = np.arange(0, len(wav) / sr, 1 / SR)
        return np.interp(t, np.arange(len(wav)) / sr, smooth).astype(np.float32)


_USE_FFMPEG = False                 # set after PortAudio fails once: don't retry it every turn


def record() -> np.ndarray:
    global _USE_FFMPEG
    if _USE_FFMPEG:
        input("\nEnter: start speaking (ffmpeg) ")
        return record_ffmpeg()
    import sounddevice as sd
    dev = sd.query_devices(kind="input")
    sr = int(dev["default_samplerate"])
    input(f"\nEnter: start speaking ({dev['name']}, {sr} Hz) ")
    frames = []
    try:
        stream = sd.InputStream(samplerate=sr, channels=1, dtype="float32",
                                callback=lambda d, *_: frames.append(d.copy()))
    except sd.PortAudioError as e:
        if shutil.which("ffmpeg"):
            print(f"  PortAudio cannot open it ({e}); recording with ffmpeg from now on")
            _USE_FFMPEG = True
            return record_ffmpeg()
        sys.exit(f"cannot open the microphone ({e}). Install ffmpeg (brew install ffmpeg) and run again: the "
                 "script then records through macOS AVFoundation. Or check System Settings -> Privacy & "
                 "Security -> Microphone (Terminal on), or use --file with a recording.")
    with stream:
        t0 = time.time()
        input("  recording... Enter: stop ")
    wav = np.concatenate(frames)[:, 0] if frames else np.zeros(0, np.float32)
    if time.time() - t0 > MAX_S:
        print(f"  (only the first {MAX_S} s are used)")
    wav = to_16k(wav[: MAX_S * sr], sr)
    level = float(np.sqrt(np.mean(wav ** 2))) if len(wav) else 0.0
    if level < 1e-4:
        print("  (silence: is the right microphone selected, and is Terminal allowed to use it?)")
    return wav


def record_ffmpeg() -> np.ndarray:
    """Fallback recorder: ffmpeg through AVFoundation (macOS) or ALSA (Linux), 16 kHz mono, until Enter.
    PortAudio failed with AUHAL err -66740 on the owner's Intel Mac (25 Nov) even with Terminal allowed."""
    src = ["-f", "avfoundation", "-i", os.getenv("VOICE_FFMPEG_INPUT", ":0")] if sys.platform == "darwin" \
        else ["-f", "alsa", "-i", os.getenv("VOICE_FFMPEG_INPUT", "default")]
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "q.wav"
        p = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *src, "-t", str(MAX_S),
                              "-ac", "1", "-ar", str(SR), str(out)], stdin=subprocess.PIPE)
        try:
            input("  recording (ffmpeg)... Enter: stop ")
        finally:
            try:
                p.communicate(b"q", timeout=5)               # "q" makes ffmpeg finish the file cleanly
            except subprocess.TimeoutExpired:
                p.kill()
        if not out.exists():
            print("  (ffmpeg recorded nothing: see its error above; list devices with "
                  "ffmpeg -f avfoundation -list_devices true -i \"\")")
            return np.zeros(0, np.float32)
        return load_file(str(out))


def load_file(path: str) -> np.ndarray:
    from faster_whisper.audio import decode_audio
    return decode_audio(path, sampling_rate=SR)


# ---------------------------------------------------------------- API
def ask(url: str, question: str, context: dict | None) -> dict:
    body = {"question": question[:500], **({"context": context} if context else {})}
    req = urllib.request.Request(url + "/ask", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"answer": f"[HTTP {e.code}] {e.read()[:200]!r}", "route": str(e.code)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--file", help="a recorded question (wav/mp3/m4a) instead of the microphone")
    ap.add_argument("--no-speak", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tts-worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--save-dir", help="also keep each recording and a JSON line per turn here")
    args = ap.parse_args()
    if args.tts_worker:
        return tts_worker()

    try:
        with urllib.request.urlopen(args.url + "/health", timeout=5) as r:
            print("API:", json.load(r).get("status"))
    except OSError as e:
        print(f"API not reachable at {args.url} ({e}). Start it: HELPDESK_CACHE_MB=512 docker compose up -d")
        return 1
    voice = None
    if not args.no_speak:
        print("loading the voice (separate process)...", flush=True)
        voice = VoiceProcess()
        print("  voice ready", flush=True)
    print("loading Whisper...", flush=True)
    asr = load_asr(args.threads)
    print("  Whisper ready", flush=True)
    save = Path(args.save_dir) if args.save_dir else None
    if save:
        save.mkdir(parents=True, exist_ok=True)

    context = None
    while True:
        try:
            wav = load_file(args.file) if args.file else record()
        except (KeyboardInterrupt, EOFError):
            print()
            return 0
        if len(wav) < SR // 4:
            print("  (nothing recorded)")
            continue
        t = time.perf_counter()
        text = transcribe(asr, wav)
        t_asr = time.perf_counter() - t
        print(f"  heard ({t_asr:.1f} s): {text}")
        if not text:
            continue
        t = time.perf_counter()
        r = ask(args.url, text, context)
        t_ask = time.perf_counter() - t
        context = r.get("context")
        print(f"  [{r.get('route')} via {r.get('route_by')}, {t_ask:.1f} s]\n  {r.get('answer')}")
        for s in r.get("sources") or []:
            print("   -", s.get("title") or s.get("id"))
        t_tts = 0.0
        if voice:
            t = time.perf_counter()
            if not voice.say(r.get("answer") or ""):
                voice = None
            t_tts = time.perf_counter() - t                 # includes playing it
        stages = ", ".join(f"{k} {v:.1f}" for k, v in (r.get("timings") or {}).items() if k != "total")
        print(f"  timings: asr {t_asr:.1f} s, answer {t_ask:.1f} s ({stages}), voice {t_tts:.1f} s")
        if save:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            import wave
            with wave.open(str(save / f"{stamp}.wav"), "wb") as w:
                w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR)
                w.writeframes((np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes())
            with (save / "turns.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"t": stamp, "heard": text, "route": r.get("route"), "answer": r.get("answer"),
                                    "asr_s": round(t_asr, 2), "answer_s": round(t_ask, 2)}, ensure_ascii=False) + "\n")
        if args.file:
            return 0


if __name__ == "__main__":
    sys.exit(main())
