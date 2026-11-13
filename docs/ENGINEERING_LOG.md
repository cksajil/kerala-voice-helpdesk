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

## 13 Nov 2026 — Phase 3: held-out evaluation (blind run) and routing fix

**Set-up.** `eval/run_heldout.py`, the whole LangGraph graph, on **heldout_v1 (n=46, held out, first
and only blind run)**. Kaggle CPU (Intel Xeon 2.2 GHz, 4 vCPU, of which **2 threads** are used:
llama.cpp prefill and decode, torch). Gemma 4 E2B Q4_0 (ggml-org GGUF) in-process via
llama-cpp-python 0.3.36 (built with GGML_NATIVE), greedy, 100 answer tokens, HELPDESK_CANON=1,
CTX_K=1, AUTO_BOOST 1.5. Weather and news call Open-Meteo and RSS live. Code revision db0d29f.
Outputs: `eval/results/phase3/` (one CSV per run with every answer, JSON summaries, hand labels).

**Numbers (automatic scorer)**

| | old KB, BM25 | expanded KB, BM25 | old KB, BM25+dense | expanded KB, BM25+dense |
|---|---|---|---|---|
| routing | 38/46 | 37/46 | 38/46 | 37/46 |
| hit@1 / hit@3 (in-scope, n=24) | 19 / 23 | 19 / 23 | 18 / 24 | 18 / 24 |
| answers correct, in-scope (n=24) | 16 | 16 | 15 | 15 |
| fee correct (n=8) | 5 | 5 | 5 | 5 |
| weather (n=5) / news (n=5) correct | 1 / 3 | 1 / 3 | 1 / 3 | 0 / 3 |
| out-of-scope declined (n=7) | 6 | 5 | 6 | 5 |
| refusal precision / recall | 6/8 / 6/7 | 5/7 / 5/7 | 6/9 / 6/7 | 5/9 / 5/7 |
| KB-gap (n=5) declined / answered | 1 / 4 | 1 / 4 | 1 / 4 | 1 / 4 |

Latency per stage, p50 / p90 seconds (old KB, BM25; the others are within 0.5 s except retrieval):
router 0.03 / 0.19, retrieve 0.24 / 0.39 (dense: 0.57 / 0.78), answer (LLM) 12.2 / 18.5,
weather 0.10 / 0.11, news 0.48 / 0.75, whole turn 10.0 / 18.5 (n=46; fee and decline turns are
instant, so the whole-turn p50 is below the LLM stage's). Dense index build: 2,776 passages,
44 min with 4 threads (cached as .npy afterwards).

**Hand labels (Claude's, not the owner's; `hand_labels.csv`).** In-scope: h04 is right (the scorer's
must-not "120" caught the general fee mentioned beside the SC/ST fee), h19 is wrong (key word present,
garbled claim from a charter row), h12 partial. Net in-scope correct stays **16/24 (BM25)**. KB gaps:
3 acceptable (job card, well permit from charter text with attribution; Haritha Karma Sena: asked which
service), 2 misleading (death registration got late-registration documents; occupancy certificate got
residence-certificate documents).

**Findings**
- Expanded KB vs old: no in-scope difference — all 24 in-scope items point at curated entries, so this
  set cannot show the kb_auto gain seen on the synth dev split (tuned-on). One regression: an
  out-of-scope question ("പ്രഥമൻ", a dessert) matched an auto entry about first aid ("പ്രഥമ ശുശ്രൂഷ")
  and was answered. Recommendation: keep HELPDESK_KB_AUTO off until a held-out set with charter-only
  services exists; owner to decide.
- Dense vs BM25: +1 hit@3, −1 answer (a false refusal from the untuned cosine gate), +0.3 s retrieval,
  44 min index build. Keep BM25 only.
- Failures by cause (old KB, BM25): routing 8 (weather words not matched in inflected or ASR form 3,
  "ഡിഗ്രി" (college degree) sent to weather 1, news synonyms 2, an electricity-bill question sent to
  property tax by the open title match 1, a KB-gap question answered "which service?" 1); weather for
  a town not in the list given for the default district 1; in-scope wrong passage 3; right passage,
  wrong reading 2 (birth in a vehicle, building-permit fee band); strict-facet decline 1.
- LLM stage is 95% of a service turn. p90 18.5 s on 2 threads.

**Routing fix (after the blind run — later heldout_v1 numbers count as seen-failure numbers)**
- Weather and news stems now match inflected forms (ചൂടാണോ, തണുപ്പുണ്ടോ, ASR കാലാവസ്ത, ന്യൂസ്,
  തലക്കെട്ടുകൾ). Words with an everyday second sense (മഴ, ചൂട്, ഡിഗ്രി, സംഭവ) give way to a named
  service (മഴവെള്ള സംഭരണി is a permit rule; "തെറ്റ് സംഭവിച്ചു" went to news before).
- Weather places: 68 Kerala municipalities/towns (`data/places_src.csv`), geocoded with the
  Open-Meteo geocoding API on Kaggle (`ingest/places.py`, URL and date kept per row, accepted only
  inside Kerala within 60 km of the district centre). A town that is not found uses its district
  centre and the reply says so.
- New **tuned-on** routing dev set `eval/route_dev.jsonl` (43 questions, written for this, no item
  within 0.5 trigram similarity of heldout_v1): rule route 28/43 → 43/43; whole graph 42/43 (the miss,
  "sing a song" sent to services by the open title match, is unchanged by the fix); weather
  district 14/27 → 27/27 (68 towns: 63 geocoded, 5 on the district centre).

**Open issues**
- The open title match answers some off-topic questions (electricity bill → property tax; with kb_auto,
  dessert → first aid; on the dev set, "sing a song").
- Building-permit fee bands are left to the LLM; a calculator like the marriage/registration ones
  would make them exact.
- The cosine gate for dense retrieval was never tuned (dense stays off).
