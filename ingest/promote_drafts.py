"""Promote checked drafts (data/kb_drafts.csv) into data/kb.csv, then rebuild data/kb.jsonl.

    python -m ingest.promote_drafts            # all drafts that pass ingest.check_drafts
    python -m ingest.promote_drafts --ids a,b  # only these

A draft with `replaces` set overwrites that kb.csv row (same id); others are appended. Drafts that
fail the checker are never promoted. Run only after the owner's spot check.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys

from ingest.check_drafts import ROOT, check

KB = ROOT / "data" / "kb.csv"
DRAFTS = ROOT / "data" / "kb_drafts.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="", help="comma-separated draft ids (default: all)")
    args = ap.parse_args()
    want = {i for i in args.ids.split(",") if i}
    drafts = [d for d in csv.DictReader(DRAFTS.open(encoding="utf-8")) if not want or d["id"] in want]
    bad = [(d["id"], check(d)) for d in drafts if check(d)]
    if bad:
        print("not promoted, checker failed:", bad)
        return 1
    with KB.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols, rows = reader.fieldnames, list(reader)
    by_id = {r["id"]: i for i, r in enumerate(rows)}
    for d in drafts:
        row = {c: d.get(c, "") for c in cols}
        row["verify"] = "yes" if d.get("verify", "").lower() in ("yes", "true", "1") else ""
        row["notes"] = (d.get("notes") or "").strip()
        if d["id"] in by_id:
            rows[by_id[d["id"]]] = row
        else:
            rows.append(row)
    with KB.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"promoted {len(drafts)} drafts into {KB.name}")
    return subprocess.call([sys.executable, str(ROOT / "data" / "build_kb.py")])


if __name__ == "__main__":
    sys.exit(main())
