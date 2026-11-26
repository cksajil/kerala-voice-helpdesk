"""LinkedIn video for Kerala Voice Helpdesk, 1920x1080, Lee LeFever style (plain, one idea per screen).

    python3 build.py                       # rough cut with a placeholder phone
    python3 build.py --demo demo.mp4 --marks marks.json   # with the owner's screen recording

marks.json: {"hook": [start, end], "5a": [start, end], ..., "5e": [start, end]} in seconds of demo.mp4.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE / "build"
W, H, FPS = 1920, 1080, 30
FM = HERE / "fonts/fontsource-noto-sans-malayalam-5.3.0/files"
FI = HERE / "fonts/fontsource-inter-5.3.0/files"

# palette: warm paper, navy ink, one accent per idea
BG, INK, MUTED = "#FAF7F0", "#1B2A41", "#5B6B7F"
GREEN, ORANGE, BLUE, GREY = "#2E8B57", "#E07A1F", "#2F6FDB", "#7A8594"


def font(size, weight=400, ml=False):
    if ml:
        f = FM / f"noto-sans-malayalam-malayalam-{weight}-normal.woff"
    else:
        f = FI / f"inter-latin-{weight}-normal.woff"
    return ImageFont.truetype(str(f), size, layout_engine=ImageFont.Layout.RAQM)


def canvas():
    im = Image.new("RGB", (W, H), BG)
    return im, ImageDraw.Draw(im)


def wrap(d, text, f, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + ([cur] if cur else [])


def para(d, xy, text, f, fill=INK, width=1500, gap=1.3, center=False):
    x, y = xy
    for line in wrap(d, text, f, width):
        lx = (W - d.textlength(line, font=f)) / 2 if center else x
        d.text((lx, y), line, font=f, fill=fill)
        y += int(f.size * gap)
    return y


def pill(d, xy, text, color, f=None):
    f = f or font(30, 700)
    x, y = xy
    tw = d.textlength(text, font=f)
    d.rounded_rectangle((x, y, x + tw + 48, y + f.size + 28), radius=(f.size + 28) // 2, fill=color)
    d.text((x + 24, y + 12), text, font=f, fill="white")
    return x + tw + 48


# ------------------------------------------------------------------ cards (full screen)
def card_big_idea():
    frames = []
    for n in (1, 2, 3):
        im, d = canvas()
        pill(d, (210, 190), "THE BIG IDEA", BLUE)
        y = para(d, (210, 290), "Sovereign AI on everyday hardware can bring AI's benefits to every citizen.",
                 font(64, 700), width=1500)
        if n >= 2:
            y = para(d, (210, y + 40), "Open-weight models  ·  open data  ·  modest hardware", font(44, 400), MUTED)
        if n >= 3:
            para(d, (210, y + 30), "Edge AI is often all you need.", font(52, 700), GREEN)
        frames.append(im)
    return frames


def card_use_case():
    frames = []
    items = ["Birth & death certificates", "Marriage registration", "Welfare pensions", "Building permits",
             "Live weather", "Today's news"]
    for n in (1, 2):
        im, d = canvas()
        pill(d, (210, 150), "A PRACTICAL USE CASE", BLUE)
        para(d, (210, 245), "A voice help desk for Kerala's local-body (LSGD) services", font(60, 700))
        para(d, (210, 420), "Ask in Malayalam. Hear the answer in Malayalam.", font(44, 400), MUTED)
        if n == 2:
            x, y = 210, 560
            for i, t in enumerate(items):
                col = ORANGE if i >= 4 else GREEN
                x2 = pill(d, (x, y), t, col, font(34, 600))
                x = x2 + 24
                if i == 2:
                    x, y = 210, y + 100
            para(d, (210, 820), "Runs on modest CPU hardware, no GPU needed.", font(44, 700), INK)
        frames.append(im)
    return frames


def card_models():
    im, d = canvas()
    pill(d, (210, 190), "OPEN MODELS", BLUE)
    para(d, (210, 290), "Speech-to-text, the LLM and the voice all run on CPU.", font(58, 700))
    y = 470
    rows = [("Voice (text-to-speech)", "my fine-tune of MMS-TTS for Malayalam"),
            ("Speech model", "my Whisper fine-tune for Malayalam"),
            ("Live demo speech-to-text", "AI4Bharat's open IndicConformer"),
            ("LLM", "Gemma 4 E2B, open weights, via llama.cpp")]
    for k, v in rows:
        d.text((210, y), k, font=font(38, 700), fill=INK)
        d.text((820, y), v, font=font(38, 400), fill=MUTED)
        y += 90
    return [im]


def box(d, xy, w, h, title, sub, color):
    x, y = xy
    d.rounded_rectangle((x, y, x + w, y + h), radius=24, fill="white", outline=color, width=5)
    tf, sf = font(36, 700), font(23, 400)
    d.text((x + (w - d.textlength(title, font=tf)) / 2, y + 30), title, font=tf, fill=color)
    for i, line in enumerate(sub.split("\n")):
        d.text((x + (w - d.textlength(line, font=sf)) / 2, y + 90 + i * 34), line, font=sf, fill=MUTED)


def arrow(d, x1, x2, y):
    d.line((x1, y, x2 - 18, y), fill=INK, width=5)
    d.polygon([(x2, y), (x2 - 22, y - 13), (x2 - 22, y + 13)], fill=INK)


def card_how_built():
    im, d = canvas()
    pill(d, (150, 110), "HOW IT'S BUILT", BLUE)
    para(d, (150, 200), "Designed, built and deployed end to end", font(56, 700))
    y, w, h = 400, 300, 190
    xs = [150, 500, 850, 1200, 1550]
    spec = [("Ear", "speech-to-text\nIndicConformer", BLUE), ("Router", "which kind of\nquestion?", INK),
            ("Rule book", "verified answers\nfrom official sources", GREEN), ("Gemma", "answers only\nfrom the passage", INK),
            ("Voice", "Malayalam\ntext-to-speech", BLUE)]
    for x, (t, s, c) in zip(xs, spec):
        box(d, (x, y), w - 40, h, t, s, c)
    for a, b in zip(xs, xs[1:]):
        arrow(d, a + w - 40, b, y + h // 2)
    # tools branch under the router
    box(d, (500, 680), 610, 150, "Tools", "live weather  ·  news  ·  fee calculator", ORANGE)
    d.line((500 + 130, y + h, 500 + 130, 680), fill=ORANGE, width=5)
    d.polygon([(630, 680), (617, 658), (643, 658)], fill=ORANGE)
    para(d, (150, 900), "LangGraph  ·  llama.cpp on CPU  ·  Hugging Face Space  ·  Docker  ·  CI tests",
         font(36, 600), MUTED, width=1700)
    return [im]


def card_challenge():
    frames = []
    for n in (1, 2):
        im, d = canvas()
        pill(d, (210, 190), "THE REAL CHALLENGE", ORANGE)
        y = para(d, (210, 290), "Not compute. Grounding.", font(80, 700))
        if n == 2:
            y = para(d, (210, y + 30), "Verified answers from scattered official sources,", font(46, 400), MUTED)
            para(d, (210, y), "and search that finds them in Malayalam, where one word carries many endings.",
                 font(46, 400), MUTED)
        frames.append(im)
    return frames


def card_status():
    im, d = canvas()
    pill(d, (210, 170), "HONEST STATUS", GREY)
    y = para(d, (210, 270), "It's v1: the answers still need refining.", font(58, 700))
    y = para(d, (210, y + 30), "But open AI is ready to make everyday citizens' lives easier: private, low-compute, "
             "and with tools. The same tool-calling pattern could, for example, track an application's status.",
             font(42, 400), MUTED, width=1500)
    para(d, (210, y + 40), "We need more people with regional-language and domain knowledge building this.",
         font(44, 700), GREEN, width=1500)
    return [im]


def card_close():
    im, d = canvas()
    para(d, (0, 170), "Try it  ·  Read the code", font(64, 700), center=True)
    rows = [("Live demo", "huggingface.co/spaces/sajilck/malayalam-voice-assistant"),
            ("Code (MIT)", "github.com/cksajil/kerala-voice-helpdesk"),
            ("Voice model", "huggingface.co/sajilck/mms-tts-malayalam-sajil"),
            ("Speech model", "huggingface.co/sajilck/whisper-small-malayalam-v2")]
    y = 340
    for k, v in rows:
        d.text((330, y), k, font=font(36, 700), fill=INK)
        d.text((680, y), v, font=font(36, 400), fill=BLUE)
        y += 80
    para(d, (0, 760), "Want something similar built?  DM me on LinkedIn.", font(52, 700), GREEN, center=True)
    return [im]


def card_hook_caption():
    """Lower-third over the opening demo clip."""
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((260, 880, 1660, 1010), radius=30, fill=(27, 42, 65, 230))
    t = "A Malayalam voice help desk, running without a GPU"
    f = font(50, 700)
    d.text(((W - d.textlength(t, font=f)) / 2, 915), t, font=f, fill="white")
    return im


# ------------------------------------------------------------------ demo panels (right half)
DEMO = [
    ("5a", "GROUNDED ANSWER", GREEN, "ജനന സർട്ടിഫിക്കറ്റ് ഓൺലൈനിൽ കിട്ടാൻ എന്തു ചെയ്യണം?",
     "How do I get a birth certificate online?",
     "Apply on K-SMART (Civil Registration) and track it online.", "Source: K-SMART, Information Kerala Mission"),
    ("5b", "GROUNDED ANSWER", GREEN, "തൊഴിലുറപ്പ് കാർഡിന് എന്തൊക്കെ രേഖകൾ വേണം?",
     "What documents do I need for an MGNREGS job card?",
     "Ration card, Aadhaar, bank passbook and a photo, through the ward member.", "Source: panchayat pages, lsgkerala.gov.in"),
    ("5c", "TOOL: FEE CALCULATOR", ORANGE, "വിവാഹം കഴിഞ്ഞിട്ട് രണ്ടു മാസമായി, രജിസ്റ്റർ ചെയ്യാൻ എത്ര രൂപ?",
     "Married two months ago. What does registration cost?",
     "Rs 120 fee + Rs 100 late fine = Rs 220", "Worked out in code, not guessed by the LLM"),
    ("5d", "TOOL: LIVE WEATHER", ORANGE, "മൂന്നാറിൽ ഇന്നത്തെ കാലാവസ്ഥ എന്താണ്?",
     "What's the weather in Munnar today?",
     "Current temperature, today's high and low, chance of rain.", "Live data from Open-Meteo, not memory"),
    ("5e", "OUT OF SCOPE", GREY, "ബിരിയാണി ഉണ്ടാക്കുന്നതെങ്ങനെ?",
     "How do I make biryani?",
     "Politely says what it can help with.", "No made-up answers"),
]
PHONE = (120, 60, 120 + 540, 60 + 960)        # where the portrait recording sits (9:16, 540x960)


def demo_panel(item, placeholder=False, stage=4):
    _, label, color, ml, en, ans, note = item
    im, d = canvas()
    if placeholder:
        x0, y0, x1, y1 = PHONE
        d.rounded_rectangle((x0 - 14, y0 - 14, x1 + 14, y1 + 14), radius=50, fill=INK)
        d.rectangle((x0, y0, x1, y1), fill="#DDE3EA")
        d.text((x0 + 90, y0 + 450), "your screen recording", font=font(34, 600), fill=MUTED)
    else:
        x0, y0, x1, y1 = PHONE
        d.rounded_rectangle((x0 - 14, y0 - 14, x1 + 14, y1 + 14), radius=50, fill=INK)
    X = 800
    pill(d, (X, 120), label, color, font(34, 700))                          # stage 1: what this shows
    if stage >= 2:                                                           # stage 2: the question
        d.text((X, 230), "Asked in Malayalam", font=font(28, 600), fill=MUTED)
        y = para(d, (X, 275), ml.replace(",", "").rstrip("?"), font(46, 700, ml=True), width=1000, gap=1.45)
        y = para(d, (X, y + 10), "“" + en + "”", font(40, 400), INK, width=1000)
    if stage >= 3:                                                           # stage 3: the answer
        d.line((X, y + 30, X + 1000, y + 30), fill="#E2DDD2", width=3)
        d.text((X, y + 60), "Answer", font=font(28, 600), fill=MUTED)
        y = para(d, (X, y + 105), ans, font(42, 700), color, width=1000)
    if stage >= 4:                                                           # stage 4: source / note
        para(d, (X, y + 25), note, font(32, 400), MUTED, width=1000)
    return im


# ------------------------------------------------------------------ music (synthesised: no copyright)
def music(seconds, path):
    sr = 44100
    t = np.arange(int(seconds * sr)) / sr
    bpm = 96
    beat = 60 / bpm
    # I–V–vi–IV in C major, one chord per bar (4 beats)
    chords = [[60, 64, 67], [55, 59, 62], [57, 60, 64], [53, 57, 60]]
    hz = lambda m: 440 * 2 ** ((m - 69) / 12)
    out = np.zeros_like(t)
    bar = 4 * beat
    for i in range(int(seconds / bar) + 1):
        ch = chords[i % 4]
        s0, s1 = int(i * bar * sr), int(min(seconds, (i + 1) * bar) * sr)
        if s0 >= len(t):
            break
        tt = t[s0:s1] - t[s0]
        env = np.minimum(1, tt / 0.4) * np.minimum(1, (bar - tt) / 0.4)
        for m in ch:                                   # soft pad
            out[s0:s1] += 0.06 * env * (np.sin(2 * np.pi * hz(m) * tt) + 0.3 * np.sin(2 * np.pi * hz(m) * 2 * tt))
        out[s0:s1] += 0.07 * env * np.sin(2 * np.pi * hz(ch[0] - 12) * tt)     # bass
        for k in range(8):                             # plucked arpeggio, eighth notes
            a0 = s0 + int(k * beat / 2 * sr)
            if a0 >= s1:
                break
            n = int(0.35 * sr)
            tk = np.arange(min(n, len(t) - a0)) / sr
            m = ch[[0, 1, 2, 1][k % 4]] + 12
            out[a0:a0 + len(tk)] += 0.05 * np.exp(-tk * 7) * np.sin(2 * np.pi * hz(m) * tk)
    fade = int(2 * sr)
    out[:fade] *= np.linspace(0, 1, fade)
    out[-fade:] *= np.linspace(1, 0, fade)
    out = out / (np.abs(out).max() + 1e-9) * 0.5
    st = np.stack([out, out], 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2), w.setsampwidth(2), w.setframerate(sr)
        w.writeframes((st * 32767).astype("<i2").tobytes())


# ------------------------------------------------------------------ assembly
def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def still_clip(frames, seconds, name, xf=0.8):
    """One clip per card: its frames revealed one after another with xf-second cross-fades; a silent track."""
    paths = []
    for i, im in enumerate(frames):
        p = OUT / f"{name}_{i}.png"
        im.save(p)
        paths.append(p)
    n = len(paths)
    per = (seconds + xf * (n - 1)) / n
    o = OUT / f"{name}.mp4"
    cmd = ["ffmpeg", "-y"]
    for p in paths:
        cmd += ["-framerate", str(FPS), "-loop", "1", "-t", f"{per:.3f}", "-i", str(p)]
    chain, last = [], "[0:v]"
    for i in range(1, n):
        chain.append(f"{last}[{i}:v]xfade=transition=fade:duration={xf}:offset={i * (per - xf):.3f}[x{i}]")
        last = f"[x{i}]"
    vf = (";".join(chain) + ";" if chain else "") + f"{last}fps={FPS},format=yuv420p[v]"
    cmd += ["-f", "lavfi", "-t", f"{seconds:.3f}", "-i", "anullsrc=r=44100:cl=stereo", "-filter_complex", vf,
            "-map", "[v]", "-map", f"{n}:a", "-t", f"{seconds:.3f}", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "18", "-c:a", "aac", str(o)]
    run(cmd)
    return [o]


def panel_video(item, placeholder, times, total, name, xf=0.4):
    """The right-hand panel revealed in stages at the given times (seconds), as a video of length total."""
    paths = []
    for st in (1, 2, 3, 4):
        p = OUT / f"panel_{item[0]}_{st}.png"
        demo_panel(item, placeholder, st).save(p)
        paths.append(p)
    starts = [0.0] + list(times)
    cmd, chain, last = ["ffmpeg", "-y"], [], "[0:v]"
    for i, p in enumerate(paths):
        end = starts[i + 1] + xf if i + 1 < len(starts) else total
        cmd += ["-framerate", str(FPS), "-loop", "1", "-t", f"{end - starts[i] + (xf if i else 0):.3f}", "-i", str(p)]
    for i in range(1, len(paths)):
        chain.append(f"{last}[{i}:v]xfade=transition=fade:duration={xf}:offset={starts[i]:.3f}[x{i}]")
        last = f"[x{i}]"
    o = OUT / f"{name}.mp4"
    cmd += ["-filter_complex", ";".join(chain) + f";{last}fps={FPS},format=yuv420p[v]", "-map", "[v]",
            "-t", f"{total:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(o)]
    run(cmd)
    return o


def demo_clip(item, demo, mark, placeholder, seconds=8.0):
    o = OUT / f"demo_{item[0]}.mp4"
    x0, y0, x1, y1 = PHONE
    if placeholder:
        ask = seconds / 2
        ranges = [[0, ask], [ask, seconds]]
    else:
        ranges = mark if isinstance(mark[0], list) else [mark]
    total = sum(e - s for s, e in ranges)
    ask = ranges[0][1] - ranges[0][0] if len(ranges) > 1 else total / 3
    # label at once, question 0.8 s in, answer 1 s after the cut to the answer, source 2.5 s later
    times = [0.8, min(ask + 1.0, total - 3.5), min(ask + 3.5, total - 1.5)]
    pv = panel_video(item, placeholder, times, total, f"panelv_{item[0]}")
    if placeholder:
        run(["ffmpeg", "-y", "-i", str(pv), "-f", "lavfi", "-t", f"{total:.3f}", "-i", "anullsrc=r=44100:cl=stereo",
             "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest", str(o)])
        return o
    parts, labels = [], ""
    for i, (s, e) in enumerate(ranges):
        parts.append(f"[1:v]trim={s}:{e},setpts=PTS-STARTPTS[v{i}];[1:a]atrim={s}:{e},asetpts=PTS-STARTPTS[a{i}]")
        labels += f"[v{i}][a{i}]"
    # the phone frame is drawn on every panel stage (demo_panel draws it), the recording goes inside it
    run(["ffmpeg", "-y", "-i", str(pv), "-i", demo, "-filter_complex",
         ";".join(parts) + f";{labels}concat=n={len(ranges)}:v=1:a=1[cv][ca];"
         f"[cv]scale={x1 - x0}:{y1 - y0}:force_original_aspect_ratio=decrease,"
         f"pad={x1 - x0}:{y1 - y0}:(ow-iw)/2:(oh-ih)/2:color=black,fps={FPS}[ph];"
         f"[0:v][ph]overlay={x0}:{y0}:shortest=1,fps={FPS},format=yuv420p[v]",
         "-map", "[v]", "-map", "[ca]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
         "-c:a", "aac", "-ar", "44100", "-ac", "2", str(o)])
    return o


def hook_clip(demo, mark, placeholder, seconds=4.0):
    cap = OUT / "hook_caption.png"
    card_hook_caption().save(cap)
    o = OUT / "hook.mp4"
    if placeholder:
        im, d = canvas()
        d.rectangle((0, 0, W, H), fill="#DDE3EA")
        d.text((700, 480), "opening demo clip", font=font(48, 600), fill=MUTED)
        base = OUT / "hook_base.png"
        im.save(base)
        run(["ffmpeg", "-y", "-loop", "1", "-t", str(seconds), "-i", str(base), "-i", str(cap), "-f", "lavfi",
             "-t", str(seconds), "-i", "anullsrc=r=44100:cl=stereo", "-filter_complex",
             f"[0:v][1:v]overlay=0:0,fps={FPS},format=yuv420p[v]", "-map", "[v]", "-map", "2:a",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", "-shortest", str(o)])
    else:
        s, e = mark
        # the portrait recording, centred on a blurred copy of itself
        run(["ffmpeg", "-y", "-ss", str(s), "-to", str(e), "-i", demo, "-i", str(cap), "-filter_complex",
             f"[0:v]split[a][b];[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=30[bg];"
             f"[b]scale=-2:{H}[fg];[bg][fg]overlay=(W-w)/2:0[m];[m][1:v]overlay=0:0,fps={FPS},format=yuv420p[v]",
             "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
             "-c:a", "aac", "-ar", "44100", "-ac", "2", str(o)])
    return o


def duration(p):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                          "-of", "csv=p=0", str(p)]).decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo")
    ap.add_argument("--marks")
    ap.add_argument("--out", default=str(HERE / "kerala_voice_helpdesk_v4.mp4"))
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    ph = not a.demo
    marks = json.load(open(a.marks)) if a.marks else {}

    seq = []   # (clip, is_demo)
    seq.append((hook_clip(a.demo, marks.get("hook"), ph), True))
    import cards2 as C
    for p in still_clip(C.card_big_idea(), 8, "idea"):
        seq.append((p, False))
    for p in still_clip(C.card_use_case(), 10, "usecase"):
        seq.append((p, False))
    for item in DEMO:
        seq.append((demo_clip(item, a.demo, marks.get(item[0]), ph), True))
    for frames, secs, name in [(card_how_built(), 8, "built"), (C.card_challenge(), 9, "challenge"),
                               (C.card_status(), 7, "status"), (C.card_call(), 6, "call"),
                               (C.card_close(), 9, "close")]:
        for p in still_clip(frames, secs, name):
            seq.append((p, False))

    # concatenate video (with short fades at each cut) and the recording's own audio
    lst = OUT / "list.txt"
    faded = []
    for i, (p, _) in enumerate(seq):
        d = duration(p)
        o = OUT / f"f_{i:02d}.mp4"
        run(["ffmpeg", "-y", "-i", str(p), "-vf", f"fps={FPS},fade=in:st=0:d=0.7:color=0xFAF7F0,fade=out:st={max(0, d - 0.7):.3f}:d=0.7:color=0xFAF7F0",
             "-af", "aresample=44100", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac",
             "-ar", "44100", "-ac", "2", str(o)])
        faded.append((o, d, seq[i][1]))
    lst.write_text("".join(f"file '{o}'\n" for o, _, _ in faded))
    body = OUT / "body.mp4"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-vf", f"fps={FPS},format=yuv420p",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-c:a", "aac", "-ar", "44100", "-ac", "2", str(body)])

    # music: full under cards, ducked to 12% under the demo segments
    total = sum(d for _, d, _ in faded)
    mus = OUT / "music.wav"
    music(total + 1, mus)
    t, expr = 0.0, []
    for _, d, is_demo in faded:
        if is_demo:
            expr.append(f"between(t,{t:.2f},{t + d:.2f})")
        t += d
    duck = "+".join(expr) or "0"
    run(["ffmpeg", "-y", "-i", str(body), "-i", str(mus), "-filter_complex",
         f"[1:a]volume='if({duck},0.12,0.55)':eval=frame[m];[0:a][m]amix=inputs=2:duration=first:normalize=0[a]",
         "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
         a.out])
    print(f"{a.out}: {duration(a.out):.1f} s")


if __name__ == "__main__":
    main()
