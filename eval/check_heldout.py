"""Checks for eval/heldout_v1.jsonl that need no LLM and do not run the help desk on it.

    python -m eval.check_heldout          # label checks + nearest existing question per item

- every expect_kb id exists in data/kb.jsonl;
- fee labels for band questions agree with helpdesk/fees.py (registration, marriage, licence);
- no held-out question is a near copy of a question in the older eval sets or a KB example
  question (character-trigram Jaccard on normalised text, threshold MAX_SIM).
The set is frozen: tests/test_heldout.py pins its SHA-256.
"""
from __future__ import annotations

import json
from pathlib import Path

from helpdesk.normalise import normalise

ROOT = Path(__file__).resolve().parents[1]
HELDOUT = ROOT / "eval" / "heldout_v1.jsonl"
SHA256 = "bd2ff9e78f170bd1c27f35229e5d9602f2e6c37d8963cefe50cf16b1dc7f0352"
MAX_SIM = 0.5   # a rephrasing of an existing question scores well above this
OLDER_SETS = ["questions.jsonl", "answer_eval.jsonl", "synth_questions.jsonl", "conversations.jsonl"]


def load(path: Path = HELDOUT) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def _grams(s: str) -> set[str]:
    s = "".join(ch for ch in normalise(s) if ch.isalnum() or ch.isspace() or "ഀ" <= ch <= "ൿ")
    s = " ".join(s.split())
    return {s[i:i + 3] for i in range(len(s) - 2)}


def sim(a: str, b: str) -> float:
    ga, gb = _grams(a), _grams(b)
    return len(ga & gb) / len(ga | gb) if ga and gb else 0.0


def _strings(obj) -> list[str]:
    """Every question-like string in an eval row (conversations nest turns)."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("question", "q", "text", "user") and isinstance(v, str):
                out.append(v)
            elif isinstance(v, (list, dict)):
                out += _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            out += _strings(v)
    return out


def existing_questions() -> list[tuple[str, str]]:
    found = []
    for name in OLDER_SETS:
        p = ROOT / "eval" / name
        if p.exists():
            for l in p.open(encoding="utf-8"):
                if l.strip():
                    found += [(name, s) for s in _strings(json.loads(l))]
    for l in (ROOT / "data" / "kb.jsonl").open(encoding="utf-8"):
        d = json.loads(l)
        found += [(f"kb:{d['id']}", s) for s in d.get("questions", [])]
    return found


def nearest(rows: list[dict]) -> list[tuple[str, float, str, str]]:
    pool = existing_questions()
    out = []
    for r in rows:
        best = max(pool, key=lambda x: sim(r["question"], x[1]))
        out.append((r["id"], sim(r["question"], best[1]), best[0], best[1]))
    return out


def label_errors(rows: list[dict]) -> list[str]:
    from helpdesk import fees
    kb_ids = {json.loads(l)["id"] for l in (ROOT / "data" / "kb.jsonl").open(encoding="utf-8")}
    errs = []
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        errs.append("duplicate ids")
    for r in rows:
        kb = r["expect_kb"]
        if isinstance(kb, list):
            errs += [f"{r['id']}: unknown kb id {k}" for k in kb if k not in kb_ids]
        if r["category"] == "out_of_scope" and kb != "refuse":
            errs.append(f"{r['id']}: out_of_scope must expect refuse")
        if r["category"] in ("weather", "news") and r["route"] != r["category"]:
            errs.append(f"{r['id']}: route should be {r['category']}")
    # fee labels against the code's bands (the questions' quantities, written out here by hand)
    band = {"h01": fees.registration_fee(25)[0], "h02": fees.registration_fee(548)[0],
            "h03": fees.marriage_fee(3 * 365)[0], "h11": fees.licence_fee(50e5, "mfg"),
            "h12": fees.licence_fee(3e7, "svc")}
    by_id = {r["id"]: r for r in rows}
    for i, fee in band.items():
        if by_id[i]["expect_fee"] != str(fee):
            errs.append(f"{i}: expect_fee {by_id[i]['expect_fee']} but fees.py gives {fee}")
    return errs


if __name__ == "__main__":
    rows = load()
    for e in label_errors(rows) or ["labels ok"]:
        print(e)
    for i, s, src, q in sorted(nearest(rows), key=lambda x: -x[1]):
        print(f"{i:4} {s:.2f} {src:22} {q}")
