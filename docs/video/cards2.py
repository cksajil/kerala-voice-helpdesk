"""Cards, second design: one idea per screen, a picture carries it, few words (The Art of Explanation)."""
from __future__ import annotations

import json
from pathlib import Path

import qrcode
from PIL import Image, ImageDraw, ImageFont

from build import BG, BLUE, GREEN, GREY, H, INK, MUTED, ORANGE, W, canvas, font

HERE = Path(__file__).resolve().parent
EMO = HERE / "fonts/noto-emoji/files"
ICONS = json.load(open(HERE / "icons.json"))
LIGHT = "#E9E3D6"


def icon(d, name, cx, cy, size, color=INK):
    fname, cp = ICONS[name]
    f = ImageFont.truetype(str(EMO / fname), size)
    ch = chr(cp)
    box = d.textbbox((0, 0), ch, font=f)
    d.text((cx - (box[0] + box[2]) / 2, cy - (box[1] + box[3]) / 2), ch, font=f, fill=color)


def centred(d, y, text, f, fill=INK):
    d.text(((W - d.textlength(text, font=f)) / 2, y), text, font=f, fill=fill)


def disc(d, cx, cy, r, color):
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)


def arrow(d, x1, y1, x2, y2, color=MUTED, w=6):
    import math
    d.line((x1, y1, x2, y2), fill=color, width=w)
    a = math.atan2(y2 - y1, x2 - x1)
    L = 24
    p1 = (x2 - L * math.cos(a - 0.45), y2 - L * math.sin(a - 0.45))
    p2 = (x2 - L * math.cos(a + 0.45), y2 - L * math.sin(a + 0.45))
    d.polygon([(x2, y2), p1, p2], fill=color)


def tile(d, cx, cy, name, label, color, r=110, lab=38):
    disc(d, cx, cy, r, LIGHT)
    icon(d, name, cx, cy, int(r * 1.05), color)
    f = font(lab, 700)
    d.text((cx - d.textlength(label, font=f) / 2, cy + r + 28), label, font=f, fill=INK)


# 1 · the big idea: three ingredients add up to AI for every citizen
def card_big_idea():
    frames = []
    for n in (1, 2):
        im, d = canvas()
        centred(d, 110, "AI for every citizen", font(84, 700))
        xs = [420, 960, 1500]
        for x, (ic, lab) in zip(xs, [("package", "Open models"), ("doc", "Open data"), ("laptop", "Everyday hardware")]):
            tile(d, x, 470, ic, lab, BLUE)
        for a, b in zip(xs, xs[1:]):
            centred_plus = (a + b) / 2
            f = font(90, 700)
            d.text((centred_plus - d.textlength("+", font=f) / 2, 405), "+", font=f, fill=MUTED)
        if n == 2:
            centred(d, 800, "Edge AI is often enough.", font(64, 700), GREEN)
        frames.append(im)
    return frames


# 2 · the use case as a story: you ask -> the help desk -> it answers aloud; then what it covers
def bubble(d, x, y, text, color, ml=True, size=34, tail="down"):
    f = font(size, 700, ml=ml)
    w = d.textlength(text, font=f) + 60
    d.rounded_rectangle((x - w / 2, y, x + w / 2, y + size + 46), radius=(size + 46) // 2, fill=color)
    d.polygon([(x - 18, y + size + 44), (x + 18, y + size + 44), (x, y + size + 74)], fill=color)
    d.text((x - w / 2 + 30, y + 16), text, font=f, fill="white")


def card_use_case():
    frames = []
    services = [("baby", "Birth & death"), ("rings", "Marriage"), ("elder", "Pensions"),
                ("construction", "Building permits"), ("cloudsun", "Weather"), ("news", "News")]
    for n in (1, 2, 3, 4):
        im, d = canvas()
        centred(d, 70, "Ask in Malayalam. Hear it in Malayalam.", font(66, 700))
        # 1 · you ask
        bubble(d, 360, 220, "ജനന സർട്ടിഫിക്കറ്റ് എങ്ങനെ കിട്ടും", BLUE, size=28)
        disc(d, 360, 470, 120, LIGHT)
        icon(d, "speaking", 360, 470, 130, BLUE)
        fl = font(40, 700)
        d.text((360 - d.textlength("You ask", font=fl) / 2, 610), "You ask", font=fl, fill=INK)
        if n >= 2:   # 2 · the help desk
            arrow(d, 510, 470, 790, 470, MUTED, 8)
            d.rounded_rectangle((875, 330, 1045, 620), radius=34, outline=INK, width=8, fill="white")
            d.rounded_rectangle((935, 348, 985, 360), radius=6, fill=INK)
            icon(d, "robot", 960, 480, 110, INK)
            t = "Voice help desk"
            d.text((960 - d.textlength(t, font=fl) / 2, 640), t, font=fl, fill=INK)
        if n >= 3:   # 3 · it answers aloud
            arrow(d, 1130, 470, 1410, 470, MUTED, 8)
            bubble(d, 1560, 220, "ഓൺലൈനായി അപേക്ഷിക്കാം", GREEN, size=28)
            disc(d, 1560, 470, 120, LIGHT)
            icon(d, "loud", 1560, 470, 130, GREEN)
            t = "It answers aloud"
            d.text((1560 - d.textlength(t, font=fl) / 2, 610), t, font=fl, fill=INK)
        if n >= 4:   # 4 · what it covers
            d.line((200, 745, 1720, 745), fill=LIGHT, width=3)
            xs = [330, 582, 834, 1086, 1338, 1590]
            for x, (ic, lab) in zip(xs, services):
                col = ORANGE if ic in ("cloudsun", "news") else GREEN
                disc(d, x, 845, 58, LIGHT)
                icon(d, ic, x, 845, 64, col)
                fs = font(28, 700)
                d.text((x - d.textlength(lab, font=fs) / 2, 915), lab, font=fs, fill=INK)
            centred(d, 985, "Kerala local-body (LSGD) services  ·  live weather & news  ·  no GPU needed",
                    font(30, 600), MUTED)
        frames.append(im)
    return frames


# 3 · how it's built: one picture of the pipeline, models named under each step
def card_how_built():
    im, d = canvas()
    centred(d, 80, "How it works", font(76, 700))
    steps = [("ear", "Listen", "IndicConformer", BLUE), ("magnifier", "Find", "verified answers", GREEN),
             ("robot", "Answer", "Gemma 4 E2B", INK), ("speaker", "Speak", "my Malayalam voice", BLUE)]
    xs = [300, 740, 1180, 1620]
    for x, (ic, lab, sub, col) in zip(xs, steps):
        disc(d, x, 400, 120, LIGHT)
        icon(d, ic, x, 400, 130, col)
        fl, fs = font(44, 700), font(30, 400)
        d.text((x - d.textlength(lab, font=fl) / 2, 545), lab, font=fl, fill=INK)
        d.text((x - d.textlength(sub, font=fs) / 2, 605), sub, font=fs, fill=MUTED)
    for a, b in zip(xs, xs[1:]):
        arrow(d, a + 140, 400, b - 140, 400)
    # tools branch from "Find"
    d.rounded_rectangle((520, 720, 1480, 860), radius=70, fill="white", outline=ORANGE, width=6)
    for x, ic in zip((620, 720, 820), ("cloudsun", "abacus", "news")):
        icon(d, ic, x, 790, 70, ORANGE)
    d.text((890, 762), "Tools: weather · fees · news", font=font(38, 700), fill=ORANGE)
    arrow(d, 740, 640, 740, 712, ORANGE)
    centred(d, 940, "All on CPU  ·  deployed on Hugging Face  ·  Docker  ·  CI tests", font(36, 600), MUTED)
    return [im]


# 4 · the challenge: compute was not the problem; grounding was
def card_challenge():
    frames = []
    for n in (1, 2):
        im, d = canvas()
        if n == 1:
            centred(d, 110, "The hard part? Not compute.", font(80, 700))
            disc(d, 960, 480, 170, LIGHT)
            icon(d, "laptop", 960, 480, 180, BLUE)
            disc(d, 1090, 590, 55, "white")
            icon(d, "check", 1090, 590, 80, GREEN)
            centred(d, 720, "A modest CPU was enough.", font(48, 600), MUTED)
        else:
            centred(d, 110, "Not compute. Grounding.", font(80, 700))
            for x, y, r in [(330, 380, -12), (470, 520, 8), (300, 650, 15), (520, 330, -4), (420, 760, -9)]:
                ic = Image.new("RGBA", (180, 180), (0, 0, 0, 0))
                icon(ImageDraw.Draw(ic), "doc", 90, 90, 130, MUTED)
                ic = ic.rotate(r, resample=Image.BICUBIC)
                im.paste(ic, (x - 90, y - 90), ic)
            d.text((250, 880), "scattered official sources", font=font(34, 600), fill=MUTED)
            arrow(d, 680, 560, 1180, 560, INK, 8)
            d.text((735, 470), "verify  ·  find in Malayalam", font=font(36, 600), fill=INK)
            d.text((720, 600), "one word, many endings", font=font(30, 400), fill=MUTED)
            disc(d, 1440, 560, 170, LIGHT)
            icon(d, "check", 1440, 560, 170, GREEN)
            d.text((1270, 780), "one right answer", font=font(40, 700), fill=GREEN)
        frames.append(im)
    return frames


# 5 · honest status and where it goes next
def card_status():
    im, d = canvas()
    centred(d, 110, "v1: works, still improving", font(76, 700))
    cols = [("check", "Grounded answers", GREEN), ("tools", "Tools", ORANGE), ("hourglass", "Answer quality: improving", GREY),
            ("clip", "Next: e.g. track an application", BLUE)]
    xs = [300, 740, 1180, 1620]
    for x, (ic, lab, col) in zip(xs, cols):
        disc(d, x, 470, 110, LIGHT)
        icon(d, ic, x, 470, 110, col)
        fl = font(34, 700)
        words, lines, cur = lab.split(), [], ""
        for w in words:
            t = (cur + " " + w).strip()
            if d.textlength(t, font=fl) > 330:
                lines.append(cur); cur = w
            else:
                cur = t
        lines.append(cur)
        for i, line in enumerate(lines):
            d.text((x - d.textlength(line, font=fl) / 2, 615 + i * 46), line, font=fl, fill=INK)
    return [im]


# 6 · the call: more regional-language builders
def card_call():
    im, d = canvas()
    disc(d, 960, 380, 170, LIGHT)
    icon(d, "people", 960, 380, 190, GREEN)
    centred(d, 620, "We need more regional-language builders.", font(68, 700))
    centred(d, 720, "Open models + free compute: if I can, you can.", font(46, 400), MUTED)
    return [im]


def qr(url, size):
    q = qrcode.QRCode(border=1, box_size=10)
    q.add_data(url)
    q.make(fit=True)
    return q.make_image(fill_color=INK, back_color="white").convert("RGB").resize((size, size), Image.NEAREST)


# 7 · close: two QR codes, one line each, the call to action
def card_close():
    im, d = canvas()
    centred(d, 90, "Try it  ·  Read the code", font(72, 700))
    for cx, url, lab, short in [(620, "https://huggingface.co/spaces/sajilck/malayalam-voice-assistant", "Live demo",
                                 "hf.co/spaces/sajilck/malayalam-voice-assistant"),
                                (1300, "https://github.com/cksajil/kerala-voice-helpdesk", "Code (MIT)",
                                 "github.com/cksajil/kerala-voice-helpdesk")]:
        im.paste(qr(url, 340), (cx - 170, 230))
        fl, fs = font(44, 700), font(28, 400)
        d.text((cx - d.textlength(lab, font=fl) / 2, 600), lab, font=fl, fill=INK)
        d.text((cx - d.textlength(short, font=fs) / 2, 665), short, font=fs, fill=BLUE)
    centred(d, 790, "Want something similar built?  DM me on LinkedIn.", font(54, 700), GREEN)
    centred(d, 900, "Models: hf.co/sajilck/mms-tts-malayalam-sajil  ·  hf.co/sajilck/whisper-small-malayalam-v2",
            font(28, 400), MUTED)
    return [im]
