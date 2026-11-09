"""Check hand-written Tier-1 entries against the OCR of the charter pages they cite.

    python -m ingest.check_kb        # writes ingest/kb_check.md, exit code 1 if anything is flagged

For every data/kb.csv row whose source names a charter in ingest/sources.csv (kb_alias column)
and a page ("PDF p.11"), it looks for each number in the entry on that page (and the page
after, for rows that run over) and checks the application method: an entry that says
"വെള്ളക്കടലാസ്" (plain paper) where the page says K-SMART is flagged. That is the error
the old font-decoding path introduced in four entries (fixed 9 Nov).
A flag is a prompt to look at the page image, not proof of an error: OCR misses digits too.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

from helpdesk.normalise import normalise

from .fetch import load_sources

HERE = Path(__file__).parent
KB_CSV = HERE.parent / "data" / "kb.csv"
_PAGE = re.compile(r"p\.\s*(\d+)")
_NUM = re.compile(r"(?<![\d.])\d[\d,]*(?![\d])")
METHODS = {"plain paper": ["വെള്ളക്കടലാസ", "വെള്ള പേപ്പ"], "k-smart": ["k-smart", "ksmart", "കെ സ്മാർട്ട", "കെ-സ്മാർട്ട"],
           "sevana": ["sevana", "സേവന പെൻഷൻ"]}


def page_text(slug: str, page: int) -> str | None:
    pf = HERE / "pages" / f"{slug}.jsonl"
    if not pf.exists():
        return None
    out = []
    for line in pf.open(encoding="utf-8"):
        pg = json.loads(line)
        if pg["page"] in (page, page + 1):
            out.append(pg.get("text", ""))
            out += [" ".join(r.values()) for r in pg.get("rows", [])]
    return normalise(" ".join(out)).replace(" ", "")


def methods(text: str) -> set[str]:
    t = normalise(text).replace(" ", "")
    return {m for m, stems in METHODS.items() if any(normalise(s).replace(" ", "") in t for s in stems)}


def check(row: dict, slug: str, page: int) -> list[str]:
    text = page_text(slug, page)
    if text is None:
        return [f"no OCR for {slug} (run python -m ingest.extract)"]
    issues = []
    nums = sorted(set(_NUM.findall(row["text"])))
    found = {n for n in nums if n.replace(",", "") in text.replace(",", "")}
    vals = [int(n.replace(",", "")) for n in found]
    for n in nums:
        if n in found:
            continue
        v = int(n.replace(",", ""))
        if any(a + b == v for i, a in enumerate(vals) for b in vals[i + 1:]):
            continue                            # a total the entry computed (100 + 20 = 120)
        issues.append(f"number {n} not found on p.{page}")
    said, page_m = methods(row["text"]), methods(text)
    if "plain paper" in said and "plain paper" not in page_m and page_m:
        issues.append(f"says plain-paper application; page says {', '.join(sorted(page_m))}")
    return issues


def main() -> int:
    aliases = {s["kb_alias"]: s["slug"] for s in load_sources() if s.get("kb_alias")}
    lines, flagged, checked = ["# Tier-1 entries vs. the charter pages they cite", ""], 0, 0
    with KB_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cites = [(a, int(m.group(1))) for a in aliases
                     for m in re.finditer(re.escape(a) + r"[^)]*?p\.\s*(\d+)", row["source"])]
            if not cites:
                continue
            checked += 1
            # An entry citing two charters passes a check if either page has it.
            per = [check(row, aliases[a], pg) for a, pg in cites]
            issues = [i for i in per[0] if all(i.split(" on p.")[0] in " ".join(x) for x in per[1:])]
            if issues:
                flagged += 1
                where = ", ".join(f"{a} p.{pg}" for a, pg in cites)
                lines.append(f"- **{row['id']}** ({where}): " + "; ".join(issues))
    lines.insert(2, f"{checked} entries checked, {flagged} flagged.\n")
    (HERE / "kb_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{checked} entries checked, {flagged} flagged -> {HERE / 'kb_check.md'}")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
