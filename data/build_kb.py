"""Build data/kb.jsonl (what the app loads) from data/kb.csv (what you edit).

Edit kb.csv in any spreadsheet (Google Sheets: File > Import, then
File > Download > CSV) and run   python data/build_kb.py

Columns
  id         unique, lowercase, e.g. cr-fee-01, charter-pension-01
  service    birth_death | marriage | trade_licence | building_permit | property_tax |
             grievance | pension | certificate | general | ...
  title      short Malayalam title (counts double in retrieval)
  text       the answerable fact block, one topic, Malayalam; this is what Gemma reads
  questions  example questions separated by " | " (retrieval only, never sent to the LLM;
             do NOT copy eval/questions.jsonl here)
  source     URL, or "file.pdf p.14" for a charter page
  verify     yes = old, secondary or single-panchayat source: re-check before relying on it
  checked    date the source was read (YYYY-MM-DD)
  notes      anything for you; not used by the app
The build stops on duplicate ids, empty text, or a missing source.
"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
REQUIRED = ["id", "service", "title", "text", "source"]


def main():
    rows, errors, seen = [], [], set()
    with (HERE / "kb.csv").open(encoding="utf-8", newline="") as f:
        for n, r in enumerate(csv.DictReader(f), start=2):
            r = {k: (v or "").strip() for k, v in r.items()}
            if not any(r.values()):
                continue
            for k in REQUIRED:
                if not r.get(k):
                    errors.append(f"line {n}: empty '{k}'")
            if r["id"] in seen:
                errors.append(f"line {n}: duplicate id {r['id']}")
            seen.add(r["id"])
            rows.append({
                "id": r["id"], "service": r["service"], "title": r["title"], "text": r["text"],
                "questions": [q.strip() for q in r.get("questions", "").split("|") if q.strip()],
                "source": r["source"], "verify": r.get("verify", "").lower() in ("yes", "true", "1"),
                "checked": r.get("checked", ""),
            })
    if errors:
        sys.exit("kb.csv has problems:\n  " + "\n  ".join(errors))
    with (HERE / "kb.jsonl").open("w", encoding="utf-8") as f:
        for e in rows:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} entries to {HERE / 'kb.jsonl'}")


if __name__ == "__main__":
    main()
