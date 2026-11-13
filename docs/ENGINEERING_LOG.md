# Engineering log (for the Google Doc)

Entries are appended after each phase, newest last. Copy them into
"Kerala Voice Helpdesk v1 – Engineering Log & Methodology".

## 13 Nov 2026 — Phase 1: held-out test set v1

**What changed**
- `eval/heldout_v1.jsonl`: 46 new Malayalam questions: 24 in scope and answerable from the Tier-1 KB,
  5 in scope with no Tier-1 entry yet (KB gaps), 5 weather, 5 news, 7 out of scope. 22 colloquial,
  11 with ASR-style spellings (IndicConformer's virama chillus, loanword and sibilant slips), 13 plain.
  Each item has expected route, acceptable KB ids (or `refuse` / `gap`), key facts, must-not facts and,
  for 10 questions, the expected fee. Schema and scoring rubric: `eval/HELDOUT.md`.
- Frozen: `tests/test_heldout.py` pins the file's SHA-256 and checks labels, composition and overlap.
- `eval/check_heldout.py`: KB ids exist, band fees agree with `helpdesk/fees.py`, and no question is a near
  copy of the older eval sets or the KB examples (max character-trigram Jaccard 0.48, threshold 0.5).

**Method notes**
- The questions were written from the KB texts, not from the 20-question set or the KB examples.
  Nine first drafts that asked the same fact as an `answer_eval.jsonl` question (that set was used for
  tuning) were replaced before the freeze.
- The help desk was not run on the set before the freeze, so the first Phase 3 run is a blind run.

**Numbers:** none yet (no system run, on purpose). pytest: 234 passed, 1 skipped (230 before + 4 new).

**Open issues**
- Topic overlap with the tuned-on sets is unavoidable with 50 KB entries; only near-copies are excluded.
- ASR errors are simulated in text, not recorded speech.
- KB-gap items are scored by hand (grounded answer of the right service / decline / wrong).
