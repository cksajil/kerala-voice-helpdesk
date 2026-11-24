"""Draft KB entries (data/kb_drafts.csv) are grounded in their saved source text."""
from ingest.check_drafts import ROOT, check, source_numbers
import csv


def test_every_draft_passes():
    rows = list(csv.DictReader((ROOT / "data" / "kb_drafts.csv").open(encoding="utf-8")))
    assert rows and all(check(r) == [] for r in rows), [(r["id"], check(r)) for r in rows if check(r)]


def test_checker_catches_a_number_not_in_the_source():
    row = next(csv.DictReader((ROOT / "data" / "kb_drafts.csv").open(encoding="utf-8")))
    bad = {**row, "text": row["text"] + " ഫീസ് 750 രൂപ."}
    assert any("750" in e for e in check(bad))


def test_number_words_count():
    assert {3, 21, 60} <= source_numbers("more than three years; within twenty-one days; 60 years")
