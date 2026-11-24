"""Check data/kb_drafts.csv before entries are promoted into data/kb.csv.

    python -m ingest.check_drafts

For every draft: the source text file(s) (';'-separated) exist and are the pages named in `source`; every number in the
Malayalam text appears in that source (as digits, or as an English number word: "three years",
"twenty-one days"); required columns are filled; facets are known. Prints one line per draft and
exits 1 if any check fails. It does not judge wording: the owner's spot check does that.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FACETS = {"amount", "documents", "time", "validity", "compulsory", "eligibility", "process"}
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
         "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "twenty-one": 21, "thirty": 30,
         "forty": 40, "forty-five": 45, "sixty": 60, "ninety": 90, "hundred": 100}
# Malayalam number words as stems (inflected forms contain them: പത്തുരൂപ, അറുപത് ദിവസ, അഞ്ഞുറ് രൂപ)
ML_WORDS = {"ഒന്ന": 1, "രണ്ട": 2, "മൂന്ന": 3, "നാല": 4, "അഞ്ച": 5, "പത്ത": 10, "പതിനഞ്ച": 15, "ഇരുപത്": 20,
            "ഇരുപത്തിയൊന്ന": 21, "മുപ്പത": 30, "നാൽപത": 40, "അൻപത": 50, "അറുപത": 60, "തൊണ്ണൂറ": 90,
            "നൂറ": 100, "അഞ്ഞൂറ": 500, "അഞ്ഞുറ": 500, "ആയിരം": 1000}


def numbers(text: str) -> set[int]:
    return {int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", text) if n.replace(",", "").isdigit()}


def source_numbers(text: str) -> set[int]:
    found = numbers(text)
    low = text.lower()
    found |= {v for w, v in WORDS.items() if re.search(rf"(?<![a-z-]){re.escape(w)}(?![a-z-])", low)}
    found |= {v for w, v in ML_WORDS.items() if re.search(rf"(?<![ഀ-ൿ]){w}", text)}
    return found


def check(row: dict) -> list[str]:
    errs = []
    for col in ("id", "service", "title", "text", "source", "checked", "source_text"):
        if not (row.get(col) or "").strip():
            errs.append(f"missing {col}")
    bad = set(f for f in (row.get("facets") or "").split("|") if f) - FACETS
    if bad:
        errs.append(f"unknown facets {sorted(bad)}")
    paths = [ROOT / p.strip() for p in (row.get("source_text") or "").split(";") if p.strip()]
    if not paths or not all(p.is_file() for p in paths):
        return errs + [f"no source text {row.get('source_text')}"]
    src = "\n".join(p.read_text(encoding="utf-8") for p in paths)
    for p in paths:
        url = p.read_text(encoding="utf-8").split("\n", 1)[0].removeprefix("URL: ").strip()
        if url not in row["source"]:
            errs.append(f"source text {p.name} is {url}, not in the source column")
    missing = sorted(numbers(row["text"]) - source_numbers(src))
    if missing:
        errs.append(f"numbers not in the source: {missing}")
    return errs


def main() -> int:
    rows = list(csv.DictReader((ROOT / "data" / "kb_drafts.csv").open(encoding="utf-8")))
    failed = 0
    for r in rows:
        errs = check(r)
        failed += bool(errs)
        print(f"{'ok  ' if not errs else 'FAIL'} {r['id']:22} {'; '.join(errs)}")
    print(f"{len(rows) - failed}/{len(rows)} drafts pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
