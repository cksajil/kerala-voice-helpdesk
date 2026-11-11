"""How much does an ASR slip cost the visitor, and does the snap layer win it back? No audio needed.

    python -m eval.asr_robust                       # dev split (tuning), all variants
    python -m eval.asr_robust --split test          # report numbers: run once, after tuning on dev
    python -m eval.asr_robust --show 10             # print some changed outcomes

Questions: eval/questions.jsonl + the synthetic questions of one split (eval/synth_questions.jsonl).
Noise: each word is hit with probability p by one Malayalam ASR-style slip (aspiration/voicing
swap ഫ->പ, ധ->ദ; ര/റ, ല/ള, ന/ണ; dental/retroflex ത/ട; vowel-sign slip ാ dropped, െ/േ, ി/ീ; chillu
written as consonant+virama; word merged with the next). These are the kinds of slips in the
10 Nov logs, generated at random, so the snap layer cannot have been fitted to them.
Metric, as in eval.asr_bias: "same answer" = the noisy question gets the same route, tool and top
passage as the clean one. Also on the 24 logged clips (real Whisper output), and "clean kept":
the clean question with the layer on must still get its own answer (the layer must do no harm).
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import random
import time
from pathlib import Path

HERE = Path(__file__).parent

SWAPS = [set("കഖഗഘ"), set("ചഛജഝ"), set("ടഠഡഢ"), set("തഥദധ"), set("പഫബഭ"), set("രറ"), set("ലള"),
         set("നണ"), set("തട"), set("ശഷസ")]
VOWEL_SWAPS = {"ി": "ീ", "ീ": "ി", "ു": "ൂ", "ൂ": "ു", "െ": "േ", "േ": "െ", "ൊ": "ോ", "ോ": "ൊ"}
CHILLU = {"ൻ": "ന്", "ർ": "ര്", "ൽ": "ല്", "ൾ": "ള്", "ൺ": "ണ്"}


def slip(word: str, rng: random.Random) -> str:
    ops = []
    for i, ch in enumerate(word):
        for g in SWAPS:
            if ch in g:
                ops.append(lambda w, i=i, g=g: w[:i] + rng.choice(sorted(g - {w[i]})) + w[i + 1:])
        if ch in VOWEL_SWAPS:
            ops.append(lambda w, i=i: w[:i] + VOWEL_SWAPS[w[i]] + w[i + 1:])
        if ch == "ാ":
            ops.append(lambda w, i=i: w[:i] + w[i + 1:])
        if ch in CHILLU:
            ops.append(lambda w, i=i: w[:i] + CHILLU[w[i]] + w[i + 1:])
    return rng.choice(ops)(word) if ops else word


def add_noise(text: str, p: float, rng: random.Random) -> str:
    words, out = text.split(), []
    i = 0
    while i < len(words):
        w = words[i]
        if rng.random() < p:
            if i + 1 < len(words) and rng.random() < 0.2:
                out.append(w.rstrip("?.,") + words[i + 1])      # merged with the next word
                i += 2
                continue
            w = slip(w, rng)
        out.append(w)
        i += 1
    return " ".join(out)


def load_questions(split: str) -> list[str]:
    qs = [json.loads(l)["question"] for l in (HERE / "questions.jsonl").open(encoding="utf-8")]
    for line in (HERE / "synth_questions.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        if split == "all" or d["split"] == split:
            qs.append(d["q"])
    return qs


def make_outcome():
    from eval.asr_bias import make_router
    return make_router()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--p", type=float, nargs="+", default=[0.15, 0.3])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (quick runs)")
    ap.add_argument("--variants", nargs="+", default=["off", "on"],
                    help="off = rules only (as shipped), on = rules + snap; or settings such as max=0.2,targets=all")
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    qs = load_questions(args.split)
    if args.limit:
        qs = qs[: args.limit]
    clips = [r for r in csv.DictReader((HERE / "asr_clips.csv").open(encoding="utf-8")) if r["reference"].strip()]
    noisy = {(p, s): [add_noise(q, p, random.Random(f"{s}|{q}")) for q in qs]
             for p in args.p for s in range(args.seeds)}

    results = {}
    os.environ["HELPDESK_ASR_SNAP"] = "0"
    outcome = make_outcome()
    clean = [outcome(q) for q in qs]
    clip_target = [outcome(c["reference"]) for c in clips]
    import helpdesk.asr_snap as snapmod
    for v in args.variants:
        t0 = time.time()
        os.environ["HELPDESK_ASR_SNAP"] = "0" if v == "off" else "1"
        if v not in ("off", "on"):                     # e.g. max=0.2,targets=all
            for kv in v.split(","):
                k, val = kv.split("=")
                os.environ[f"HELPDESK_ASR_SNAP_{k.upper()}"] = val
            importlib.reload(snapmod)
        r = {"clean_kept": sum(outcome(q) == c for q, c in zip(qs, clean)),
             "clips_same": sum(outcome(c["heard_v0"]) == t for c, t in zip(clips, clip_target))}
        shown = 0
        for (p, s), nq in noisy.items():
            same = 0
            for q, n, c in zip(qs, nq, clean):
                o = outcome(n)
                same += o == c
                if o != c and shown < args.show and v != "off":
                    shown += 1
                    print(f"  [{v} p={p}] {q}\n      heard {n}\n      got {o[0]}/{o[2]}  want {c[0]}/{c[2]}")
            r.setdefault(f"p={p}", []).append(same)
        r["seconds"] = round(time.time() - t0, 1)
        results[v] = r

    n = len(qs)
    print(f"\n{n} questions ({args.split}), {args.seeds} noise seeds; {len(clips)} logged clips\n")
    hdr = f"{'variant':10} {'clean kept':>11} {'logged clips':>13} " + " ".join(f"{'p=' + str(p):>12}" for p in args.p)
    print(hdr)
    for v, r in results.items():
        cells = " ".join(f"{100 * sum(r[f'p={p}']) / (n * args.seeds):11.1f}%" for p in args.p)
        print(f"{v:10} {r['clean_kept']:>5}/{n:<5} {r['clips_same']:>6}/{len(clips):<6} {cells}")
    if args.out:
        args.out.write_text(json.dumps({"split": args.split, "n": n, "seeds": args.seeds, "results": results},
                                       ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
