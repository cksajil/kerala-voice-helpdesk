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

## 13 Nov 2026 — Phase 2: KB expansion by automatic ingestion

**What changed**
- `ingest/kb_auto.py` (new pipeline step after `canon`): promotes canonical charter services that appear in
  2+ panchayats' charters to entries with the curated KB's schema (`data/kb_auto.jsonl`). No new LLM calls:
  it reuses Gemini's cached structuring (`ingest/canon_cache.jsonl`, public charter text only).
  - Facts: a fee or time is stated as a fact only when 2+ charters give the same normalised value;
    otherwise each panchayat's value is listed with "check with your panchayat".
  - Provenance: charter URL + PDF page per member row; `checked`/`fetched` = fetch date (13 Nov).
  - `verify=false` only if every source is an official government domain *and* every stated fee/time is
    agreed by 2+ charters. A test checks every number in every entry against its charter rows.
  - Dedupe against the curated 50 (unchanged): asymmetric title-word coverage with stem matching, plus
    "half or more of Gemini's Malayalam synonyms are covered by curated titles/questions" (catches
    അഗതി പെൻഷൻ = widow pension). 35 dropped as duplicates; 20 kept but listed for a hand check.
- Loaded with `HELPDESK_KB_AUTO=1` (off by default) at `HELPDESK_AUTO_BOOST=1.5`; promoted entries replace
  their canon passage and charter rows; answers keep "according to the charters" attribution.
- Kaggle refetch (`notebooks/kaggle_phase2`, private CPU kernel): 20 charters unchanged; Koppam and
  Poothrikka pages changed (same extracted text); Thalappalam and Chakkupallam still 403; website: no
  changes. Rebuilding `docs.jsonl` with the current router retagged 74 "പട്ടിക" passages (pet ->
  lsgd_other; the file predated the 12 Nov fix), so canon is 415 -> 410 services with 13 groups waiting
  for a Gemini run, and the synth eval is 1,674 -> 1,657 questions.
- K-SMART (`ksmart.lsgkerala.gov.in`) is a JavaScript app: even /robots.txt returns the app shell and no
  page has server-rendered text, so it is not crawled.
- `ingest.fetch.allowed` now follows RFC 9309 (an unreadable robots.txt disallows the site).
- Weekly refresh workflow rebuilds canon (from the cache) and kb_auto.

**Numbers**
- Entries: 130 new (curated 50 unchanged), all from official domains; verify=true 60 (46%) — the facts
  of those differ between charters. By service: lsgd_other 37, health 27, certificate 20,
  trade_licence 13, general 9, pet 7, property_tax 4, building_permit 4, marriage 3, birth_death 3,
  unemployment 1, grievance 1, pension 1 (full table: `ingest/kb_auto_report.md`).
- AUTO_BOOST sweep, Kaggle CPU, BM25 only, **tuned-on data only** (synth dev split n=1,109, pytest,
  20-question set); canon alone = strict 245 (22.1%), same service 387 (34.9%):

  | boost | strict | same service | fixes/breaks (strict) | pytest |
  |---|---|---|---|---|
  | 1.0 | 245 | 385 | 2/2 | pass |
  | 1.25 | 258 | 400 | 16/3 | pass |
  | **1.5** | **267 (24.1%)** | **410 (37.0%)** | **27/5, p=1e-4** | **pass** |
  | 1.75 | 271 | 416 | 35/9 | 1 fail |
  | 2.0 | 270 | 417 | 39/14 | 3 fail |
- pytest 254 passed, 1 skipped (default flags); 253 with KB_AUTO on.

**Open issues**
- 13 regrouped charter groups need a Gemini run (needs GEMINI_API_KEY: owner-run Kaggle notebook).
- 20 possible duplicates to check by hand (report table); one known false duplicate (building-permit
  recommendation letter) and one missed (Kudumbashree registration matched the help-desk entry).
- Gemini's summaries are number-checked but other wording is not; verify=false means sources and
  fee/time agree, not that a person read the entry.
- Not yet measured on the held-out set (Phase 3).
