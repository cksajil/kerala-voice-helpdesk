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

## 13 Nov 2026 — Phase 4: FastAPI service and Docker

**What changed**
- `helpdesk/api.py`: `POST /ask` (question + optional `context` → answer, route, refused, sources,
  per-stage timings, and the `context` for the next turn; no server-side sessions), `GET /health`
  (loading → warming → ok / error, warm-up progress, settings). Gemma loads in-process with a
  llama-cpp-python RAM prompt cache (`HELPDESK_CACHE_MB`, default 2048); at start-up one turn runs,
  then every curated passage's answer prompt is prefilled (one token each) in a background thread,
  so `/ask` answers while warming. One turn at a time on the model (lock). Back ends: inproc
  (default), http (llama-server), none.
- `Dockerfile`: two stages; `python:3.12.16-slim-bookworm` pinned by digest; llama-cpp-python 0.3.36
  compiled once into a wheel with portable flags (AVX2/FMA, no native tuning, no BLAS, no GPU); all
  other packages pinned in `requirements-api.txt`; non-root user (uid 10001); healthcheck on
  status "ok". The GGUF (2.8 GB) downloads on first start into a volume.
- `docker-compose.yml`: 2 CPUs, 6 GB, read-only root filesystem, tmpfs /tmp, no-new-privileges,
  port bound to 127.0.0.1. README section "Run locally with Docker".
- `scripts/smoke_test.py` (standard library only): waits for "ok", then 5 checks — service answer
  with a source, fee worked out in code, follow-up through the returned context, off-topic
  declined, empty question rejected (422).
- `tests/test_api.py`: 4 tests without a model (health, answer, context follow-up, validation,
  warm-up prefills each curated passage once).

**Numbers** (Kaggle CPU, 2 threads, the image's pins and environment run with uvicorn — Kaggle has
no Docker daemon, so the image itself was not built there; revision c622fd8)
- Smoke test: 5/5 pass, warm and cold servers. Start to "ok": 298 s with warm-up (model load + one
  turn ≈ 40 s, 50 passages prefilled in 239 s); 30 s without warm-up.
- Warm vs cold, n=20 curated titles asked as questions, same 20 passages retrieved in both runs:
  LLM stage p50 / p90 **5.2 / 8.9 s warm** vs **8.7 / 12.3 s cold** (cold = no cache; only the
  previous prompt's prefix is reused). These are title questions, so a best case for the cache.

**Open issues**
- The Docker image itself has not been built anywhere yet (no Docker in this container or on Kaggle):
  the owner should run `docker compose up --build -d` then `python3 scripts/smoke_test.py`.
- The smoke test's follow-up ("എത്ര ദിവസം എടുക്കും?" after marriage documents) routes correctly but
  Gemma declined it with the model loaded: the passage it got did not state a time.
- Warm-up covers curated passages only; with HELPDESK_KB_AUTO=1, `HELPDESK_WARMUP=all` adds the 130
  auto entries (≈10 more minutes at start-up).

## 13 Nov 2026 — Phase 3 addendum: llama-server vs llama-cpp-python

**Set-up.** `eval/bench_llama.py`, Kaggle CPU, 2 threads, n_ctx 2048, greedy, 100-token cap. 29 real
answer prompts (answer prompt + top-1 BM25 passage + question for the held-out service and KB-gap
questions; prompt p50 356 tokens), rendered once with the GGUF's chat template so both back ends get
the same text. llama-cpp-python 0.3.36 in-process vs llama-server built from **the same llama.cpp
source** (the copy vendored in the 0.3.36 sdist, build b1-0c1e570), both GGML_NATIVE. Run one after the
other. Cold = no prefix reuse; warm = reuse of the previous request's prefix (shared system prompt).
Results: `eval/results/phase3/bench/`.

| per request, n=29 | llama-cpp-python | llama-server (HTTP) |
|---|---|---|
| cold wall p50 / p90 | 23.2 / 27.3 s | 22.8 / 26.3 s |
| cold overhead (wall − llama.cpp compute) p50 | 0.05 s | 0.02 s |
| warm wall p50 / p90 (mean) | 14.6 / 19.5 s (14.7) | 17.9 / 24.3 s (14.8) |
| model load | 8.1 s | 6.0 s |
| `create_chat_completion` (template per call), cold p50 | 23.0 s | — |

- The wrapper costs ~0.05 s per request against 15–23 s of compute: there is no speed reason to move to
  llama-server. Python minus server, cold: p50 +0.20 s, mean +0.57 s, which is within the run-to-run
  spread of compute time. Warm means are equal (14.7 vs 14.8 s); the server's higher warm p50 comes from
  how much prefix each request happened to share with the one before.
- Greedy outputs were identical for 19/29 prompts. The rest differ in wording (one changes a fact: "free
  for a year, then 5 rupees" vs "5 rupees after a year"), most likely from the back ends' different
  default batch sizes changing float summation order. Neither back end is more correct on this sample.
- Rendering the chat template per call (the Space's call path) costs nothing measurable (23.0 vs 23.2 s).
- The first plan (the latest llama.cpp release binary) failed: the repo's "latest" release no longer
  carries binaries. Building from the vendored source is the fairer comparison anyway.

## 14 Nov 2026 — Phase 6: Space update and held-out re-run

**Space update** (owner's decision: HELPDESK_KB_AUTO stays off). Space commit 66c47d9 (previous
27d2ca2, for rollback): `helpdesk/graph.py`, `retriever.py`, `tools.py` from this branch;
`data/docs.jsonl` and `canon.jsonl` from the 13 Nov charter rebuild; `data/places.jsonl` (68 towns);
`data/kb_auto.jsonl` (loaded only if HELPDESK_KB_AUTO=1). Space variables unchanged (HELPDESK_CANON=1);
transformers stays at 4.46.3 in the Space's requirements. `app.py` and the ASR/TTS stack untouched.
Live check: `docs/SPACE_CHECKLIST.md` (10 new questions; the owner asks them by voice).

**Held-out v1 re-run (not blind: the fixes followed the blind run's failures).** Kaggle CPU, 2 threads,
BM25, revision 7456b0b; `eval/results/phase6/`. Rescored after the scorer fixes below.

| | blind, old KB | re-run, old KB | re-run, expanded KB |
|---|---|---|---|
| routing | 38/46 | 44/46 | 43/46 |
| hit@1 / hit@3 | 19 / 23 of 24 | 20 / 23 | 20 / 23 |
| service answers, automatic / by hand (Claude) | 16 / 16 of 24 | 16 / 17 | 16 / — |
| news | 3/5 | 5/5 | 5/5 |
| weather | 1/5 | 0/5 (Open-Meteo failed all 5 calls; route and district right) | 5/5 |
| off-topic declined | 6/7 | 6/7 | 5/7 |
| whole turn p50 / p90 | 10.0 / 18.5 s | 10.6 / 19.4 s | 10.6 / 18.6 s |

**Scorer fixes** (`eval/run_heldout.py`): the weather district was read from the wrong tuple field
(a latitude for hand-written places); a "can't get that now" reply no longer counts as a correct
weather/news answer. All saved runs rescored (`*_rescored.*`); the blind numbers did not change.
Tool failures are now logged by the graph (exception type only, no question text).

**Kaggle use, 13 Nov:** 13 runs on 7 private CPU notebooks, ≈6 h 25 min; ≈3 h of it avoidable (three
failed benchmark attempts, two 44-min dense index builds for a configuration that stays off).

## 24–25 Nov 2026 — Core-service content from official sources; Space update

**Why.** The blind held-out run and the 14 Nov live check showed the failures were mostly content (KB gaps,
loose charter rows) and phrasing the rules never saw; the charters list services but do not explain them.
Owner approved a list of 30 core services (`docs/CORE_SERVICES.md`) and a v1 split: answer well where
official text exists, and give a cautious "rules changed in 2025, confirm at the panchayat" answer for
building rules.

**Sources** (fetched on Kaggle, robots.txt respected, 2 s between requests, URL / date / SHA-256 kept in
`ingest/core/manifest_*.json`, extracted text in `ingest/core/texts/`): births and deaths rules and
circulars, building tax rules 2011, RTI fee rules, pig and dog licence rules, Right to Service Act and
rules, marriage rules, trade licence rules (LSGD Panchayat Wiki, 3,424 pages surveyed: ~3,150 are 2018
law-manual scans, 177 articles, ~15 relevant); official panchayat pages for pensions, MGNREGS and
grievances; G.O.(Rt) 41/2023/LSGD (Haritha Karma Sena); Kerala Municipality Building (Amendment) Rules
2025. Not found or not fetchable: KPBR 2019 consolidated text with the 2021/2025 amendments, the
civil-registration fee page, KSEB (robots.txt disallows), LIFE Mission guidelines (only a scanned 2017
circular).

**Entries.** 16 drafts, each number checked against the saved source (`ingest/check_drafts.py`, digits
and English/Malayalam number words), titles and example questions kept below 0.4 trigram similarity to
held-out v1. Owner review 25 Nov: all OK; promoted (`ingest/promote_drafts.py`), curated KB 50 → 63.
3 replace old entries — the old-age pension entry had three errors (house, car and residence rules), the
pension amount now cites an official page, the reporting entry quotes rule 6 (vehicle) — and 13 are new.

**Checks** (offline, BM25, no LLM): 266 tests, 20/20 eval, route dev 43/43. Top-1 passage changed for
39 of 1,804 questions (synth, answer eval, route dev, live-check turns; `eval/results/drafts/`): by
Claude's judgement ~19 better, ~8 worse (vague questions caught by a shared word), the rest neutral.
Draft wording caused most first-trial regressions ("അപേക്ഷിക്കേണ്ട വിധവും" contains "വിധവ" and pulled
widow questions; generic title words pulled vague questions); fixed by retitling, not by rules.

**Code found by this work**
- ASR snap: "കെട്ടിടത്തിലാണ്" was snapped to a new KB word; case + copula endings added to the inflection
  guard.
- Out-of-scope check reads the question as heard too: the snap turned "സെക്രട്ടറിയറ്റ്" into
  "സെക്രട്ടറിയുടെ" once entries said "secretary" (0 of 1,757 eval questions change their decline).

**Space.** 9347d60 (previous 721605b): kb.jsonl, asr_snap.py, graph.py, intent.py (with the 14 Nov
live-check fixes). Startup: 2,789 passages, prompt cache 63 passages in 109 s. Also 25 Nov: usage log no
longer records `bad_input` calls (bots calling the endpoint; 721605b); 143 such rows removed from
`turns_cd03ebe6.jsonl` at the owner's request (history kept).

**Open.** Live check of the new entries; building rules after v1; gaps above; held-out v2 from real users.

### 25 Nov — Live checks after the update (owner, by voice, 25 turns)

- Checklist (`docs/SPACE_CHECKLIST.md`): 9/10. Fixed since 14 Nov: the marriage documents follow-up, and the
  rainwater question (was "permit in 30 days", now the cautious entry). Failure: a misheard town
  ("കട്ടുപനിയിൽ") silently got Thiruvananthapuram's weather. Fix: an unknown "<place>ിൽ" place is now
  said, as "<place>ിലെ" already was (everyday "in the …" and time words excluded; Space f0fd6b7).
- New entries: 5/7 right (certificate correction, job card, marriage compulsory + documents + follow-up,
  ration-card decline, well permit). Wrong: a 1200 cc car question got the house-size rule — the right
  entry, but four conditions in one passage; split into vehicle / house / land-and-income entries (KB 65;
  titles phrased as people ask, which also keeps a garbled Elikulam charter row from winning the title
  match). Partial: "demolished building, still pay tax?" got a Kanhirapuzha charter row (show the building
  number, pay tax up to that year) instead of the rule-based entry. A guard that keeps curated entries over
  same-service charter rows was tried and rejected: 66 of 1,824 questions changed, many for the worse.
- Gemma sometimes drops the cautious "rules changed in 2025" sentence (well permit), within the 1–2
  sentence answer limit.
- Usage log: `engine`/`mode` were the page's stale values ("whisper", "direct"); now the ASR and reply
  engine that actually ran (Space baae505). Older rows keep the old values.
- Turn times 0.7–10.9 s end to end; the slowest are LLM answers (up to 8.6 s).

## 2026-11-25 (evening): Docker image built and tested on the Mac

What changed (branch `claude/tender-cannon-awgv85`, after PR #6 was merged):
- `requirements-api.txt`: pinned llama-cpp-python's own dependencies (diskcache 5.6.3, jinja2 3.1.6,
  markupsafe 3.0.4). The Dockerfile builds it with `--no-deps`, so the first build failed at the offline install.
- `helpdesk/prompt_cache.py`: llama-cpp-python 0.3.36's `LlamaRAMCache` counts only the llama.cpp state
  of each entry, not the copy of the prompt's logits it also keeps (up to 512 x 262k float32 for Gemma).
  With Docker Desktop at 4.1 GB the container was killed during warm-up at passage 9–13, whatever
  `HELPDESK_CACHE_MB` was. The new cache keeps only the last logits row (the rest is never read:
  `load_state` sets `_requires_eval`, the last token is evaluated again) and counts every array.
- `docker-compose.yml` passes `HELPDESK_CACHE_MB`; `/health` shows cache entries/MB and RSS.

Numbers (Intel MacBook Air, Docker Desktop 4 CPUs / 4.1 GB, `N_THREADS=2`, `HELPDESK_CACHE_MB=512`, n=5 smoke checks):
- Build: about 15 min the first time (llama-cpp-python compile 830 s), about 1 min after a code change.
- Warm-up 924 s for 65 curated passages; cache 66 entries, 449 MB; RSS 2,832 MB.
- `scripts/smoke_test.py`: 5/5 pass. Answer turns 8.3 s (service) and 15.6 s (follow-up, router 4.8 s);
  fee in code 0.03 s; off-topic decline 0.26 s.

Open issues: the follow-up's router took 4.8 s (not looked at yet); 2 GB cache not tried on a 6 GB Docker.

## 2026-11-25 (late): review note on the ration-card rule; slow first follow-up

- **Codex review note (PR #6, `helpdesk/intent.py:98`)**: it suggested narrowing the certificate exception
  of the ration-card decline. Tried a rule that keeps only "…പേരില്ലെന്ന സർട്ടിഫിക്കറ്റ്": it changed 1 of
  1,866 eval questions (all `eval/*.jsonl`, tuned sets included), and that one for the worse. "റേഷൻകാർഡിൽ
  പേര് ചേർക്കാനുള്ള ജനനത്തീയതി സർട്ടിഫിക്കറ്റിന് എന്ത് വേണം?" is a panchayat certificate (charter
  canon-g98971f9957) and would have been declined. Kept the rule; the case is now a test.
- **Slow first follow-up** (4.8 s router on the Mac smoke test): `retriever._content_words` normalised the
  whole generic-word list again for every word, 1.6M `normalise()` calls to build one service's
  vocabulary on the first follow-up. Normalised once now; same words on 399 sampled passages; router
  1.6 s -> 0.07 s on the first follow-up (this container, CPU); the test suite 66 s -> 23 s.
- Bots: already kept out of the evidence log by the 25 Nov Space change (non-app calls are skipped).
- **Voice on the Mac**: `scripts/voice_chat.py` (microphone -> Whisper v2 int8 -> local API -> VITS voice),
  `requirements-voice.txt`. Not run end to end yet: the gated Whisper model needs the owner's HF login,
  so the first run is on the owner's Mac. IndicConformer is not used locally (2.4 GB ONNX next to Docker).
- **LangSmith, local only** (owner's decision): LangGraph traces when `LANGSMITH_TRACING=true`; Gemma calls
  are `llm` runs (`langsmith.traceable`, a no-op without tracing). `helpdesk/tracing.py` switches tracing
  off in the API unless `HELPDESK_TRACING=1`; `/health` reports it. 272 tests pass.
- **First local voice test** (owner's Intel Mac, 5 turns, Whisper v2 int8; PortAudio could not open the
  microphone, AUHAL err -66740, so the script now records with ffmpeg/AVFoundation): ASR 3.9–5.1 s a turn.
  1. Fragment "പി" was answered as a follow-up from another panchayat's charter row (31 s). Fixed in the
     router: a question with no word of 3+ letters gets "please repeat" (0 of 1,854 eval questions affected).
  2. "മൂന്നാറിലെ കാലാസ്ഥ എന്താണ്?" (Whisper dropped "വ") was declined. Not fixed: the snap layer cannot
     restore a dropped syllable without loosening its cost limit, which earlier rewrote correct words.
     It is an ASR error; IndicConformer on the Space is the better fix. Open.
  3. Answer turns took 21–31 s against 8–16 s in the smoke test; the script now prints the API's per-stage
     timings to find out where. Open.
- voice_chat with voice on: segfault right after the first transcription (PyTorch and CTranslate2 in one
  process, two OpenMP runtimes). The voice now runs in its own process; playback through `afplay` on macOS.
- **Second local voice test** (same Mac, voice on, 4 turns; Docker still on the image from before the
  fragment fix, so "പി" got the scope reply there; the current code asks to repeat it):
  - The voice works in its own process; ffmpeg records; no crash.
  - ASR 3.4–5.3 s a turn (Whisper v2 int8, 2 threads).
  - Heard right, 1 of 4: "ജനനരജിസ്ട്രേഷൻ എത്ര ദിവസത്തിനകം ചെയ്യണം" -> "21 ദിവസത്തിനകം" (correct), answer 6.2 s
    (Gemma 6.0 s, retrieve 0.2 s). The earlier 21–31 s answers did not recur; cause unknown.
  - Misheard, 2 of 4, declined with the scope reply: "സത്രിപ്പിക്കറ്റിന്…" (സർട്ടിഫിക്കറ്റ്) and
    "ജനകരജിസ്ട്രേഷന് … ഏകകളാവശ്യമാണ്" (ജനന, രേഖകൾ). The snap layer correctly leaves words this far from
    any known word; not patched. Next: the same questions on the Space (IndicConformer) to see how much is
    Whisper's.
  - Voice (VITS on CPU, including playback) 8–23 s; the long scope reply is the slowest.
  - n = 4 turns, one speaker, an informal check rather than an eval number.
