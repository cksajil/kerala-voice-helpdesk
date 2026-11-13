# Kerala Voice Helpdesk

**Ask a question in Malayalam, hear the answer in Malayalam, all on a CPU.**
A voice assistant for Kerala local-body (panchayat / K-SMART) services, today's weather and today's news.

🎙️ **[Try the live demo](https://sajilck-malayalam-voice-assistant.hf.space)** ([Hugging Face Space](https://huggingface.co/spaces/sajilck/malayalam-voice-assistant))

| Stage | What runs |
|---|---|
| Speech to text | Malayalam ASR on CPU (see [malayalam-asr-eval](https://github.com/cksajil/malayalam-asr-eval) for how the models compare) |
| **Reply (this repo)** | LangGraph agent: explicit routing, BM25 retrieval (BGE-M3 dense optional, off) over a knowledge base built from panchayat citizen charters, Gemma 4 E2B through llama.cpp |
| Text to speech | [sajilck/mms-tts-malayalam-sajil](https://huggingface.co/sajilck/mms-tts-malayalam-sajil) (VITS) |

**Highlights**
- Answers only from retrieved passages; when the passage does not hold the answer the model must say `UNSURE`, which becomes a polite refusal.
- Fees with band arithmetic are computed in code, and weather and news numbers go out through templates, so the LLM never does arithmetic or copies figures.
- `ingest/` turns citizen-charter PDFs into ~600 cited passages (Malayalam OCR with Tesseract) and checks hand-written entries against the pages they cite.
- An eval suite (routing, retrieval hit@3, refusals, a 100-question answer eval and multi-turn conversations) and fixes driven by real conversation logs, recorded below.
- A whole spoken turn takes 1.7–6.2 s on the Space's shared CPU.

## Results (13 Nov 2026)

**Held-out set** `eval/heldout_v1.jsonl`: 46 Malayalam questions written for evaluation only and frozen
by hash (24 service questions, 5 the knowledge base cannot answer, 5 weather, 5 news, 7 off-topic;
plain, colloquial and ASR-style wording). Whole graph, Gemma 4 E2B Q4_0 through llama-cpp-python,
Kaggle CPU with **2 threads**, BM25, curated + charter knowledge base (`HELPDESK_KB_AUTO` off).

| | Blind run (first and only look) | Re-run after fixes (**not blind**: fixes were made after seeing the blind failures) |
|---|---|---|
| Routing | 38/46 | 44/46 |
| Right passage first (hit@1) / in top 3 (hit@3), service questions | 19/24 / 23/24 | 20/24 / 23/24 |
| Service answers correct: automatic / by hand (Claude's labels) | 16/24 / 16/24 | 16/24 / 17/24 |
| Fees exactly right | 5/8 | 5/8 |
| News | 3/5 | 5/5 |
| Weather (routing and district right) | 1/5 | 5/5 (answer needs Open-Meteo up: 0/5 in one re-run when it was down) |
| Off-topic questions declined | 6/7 | 6/7 |
| Refusal precision / recall | 6/8 / 6/7 | 6/8 / 6/7 |
| Whole turn p50 / p90 | 10.0 s / 18.5 s | 10.6 s / 19.4 s |

- Knowledge-base gaps (n=5): by hand (Claude's labels), 3 handled acceptably, 2 misleading answers.
- Expanded knowledge base (+130 charter entries): no gain on this set (all 24 service questions point at
  curated entries) and one off-topic question answered; its gain (+22 exact hits, 1,109 questions) is on
  the tuned-on synthetic dev split only. Dense BGE-M3: +1 hit@3, −1 answer, +0.3 s; off.
- Prompt cache warmed at start-up (FastAPI service): LLM time p50 5.2 s warm vs 8.7 s cold (n=20).
- Details, per-question CSVs and hand labels: `eval/results/phase3/`, `eval/results/phase6/`,
  `docs/ENGINEERING_LOG.md`.

### What's left / known limitations
- **n is small.** 24 service questions: one question is 4 points. Treat differences of one or two as noise.
- **Wrong passage, confident answer.** The open title match sometimes answers an off-topic question
  from a loosely matching passage (an electricity-bill question got property tax).
- **Fee bands for building permits** are read by the LLM, which got one wrong (500 instead of 150);
  birth/death, marriage and trade-licence fees are computed in code.
- **Charter-only services** (130 auto entries) are built from 2–6 panchayats' charters; 46% are marked
  "may vary" because charters disagree. Fees differ by panchayat; the reply says so.
- **Live tools.** Weather and news depend on Open-Meteo and RSS feeds; when they fail the reply says
  "try later". 5 of 68 towns have no coordinates and use their district centre (the reply says so).
- **Latency.** A service answer takes 5–19 s on 2 CPU threads; prefill dominates.
- **Not yet done:** the Docker image has not been built on a real Docker host (the same pins ran with
  uvicorn on Kaggle); the 13 charter groups regrouped on 13 Nov need a Gemini run; 20 possible
  duplicate auto entries need a hand check; LangSmith tracing (optional) is not set up.

---

The rest of this README is the development log, newest sections last.

---

## Reply stage (v1, day 1)

LangGraph agent that answers Malayalam questions about Kerala local-body
(LSGD / K-SMART) services, today's weather and today's news. It is the reply
stage between Whisper v2 (ASR) and VITS (TTS) in the voice assistant.

```
router ─► weather (Open-Meteo) ──────────────────────► answer ─► END
       ├► news (RSS headlines) ──────────────────────► answer
       ├► calc (fee bands, in code) ─────────────────────────► END
       │     └─ not a fee question ─► retrieve (BM25+dense) ─► answer
       └► other (scope message) ─────────────────────────────► END
```

Design choices
- Routing is explicit (keyword stems first, LLM label only if no rule fires):
  Gemma 4 E2B is unreliable at free-form tool calling.
- Weather/news numbers go out through templates, never through the LLM.
- Fee questions that need band arithmetic (late birth/death registration,
  licence fee by investment and sector) are computed in code: Gemma got
  these wrong, and the calculator needs no LLM call, so it answers instantly.
- Service answers use only retrieved passages; the LLM must reply `UNSURE`
  when the answer is not there, which becomes a refusal.
- One Malayalam normaliser (old chillu → atomic chillu, then strip ZWJ/ZWNJ)
  for the index, the query and ASR output.

## Run
```bash
pip install -r requirements.txt
python data/build_kb.py                       # data/kb.csv (you edit) -> data/kb.jsonl (app loads)
python -m llama_cpp.server --model gemma-4-E2B-it-Q4_0.gguf --port 8080 --n_ctx 4096   # separate terminal
python -m helpdesk.cli "ട്രേഡ് ലൈസൻസ് പുതുക്കാൻ എന്ത് ചെയ്യണം?"
```
No LLM? `HELPDESK_LLM=none` reads out the best passage instead.
No dense model? `HELPDESK_DENSE=0` uses BM25 only.

## Run locally with Docker
CPU only; works on an Intel Mac (Docker Desktop, give it 2 CPUs and 6 GB RAM in Settings → Resources).
```bash
docker compose up --build -d                  # first build compiles llama-cpp-python (~10 min)
docker compose logs -f helpdesk               # first start downloads Gemma 4 E2B Q4_0 (2.8 GB) into a volume
python3 scripts/smoke_test.py                 # waits for /health "ok", asks 5 questions, exit 0 = pass
```
Ask it yourself:
```bash
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "ജനന സർട്ടിഫിക്കറ്റ് എങ്ങനെ കിട്ടും?"}' | python3 -m json.tool --no-ensure-ascii
```
- `POST /ask` takes `{"question", "context"}` and returns `answer`, `route`, `refused`, `sources`,
  per-stage `timings` and a `context` to send back with the next question (follow-ups such as
  "അതിന് എത്ര ദിവസം?" stay on the same service). The service keeps no session state.
- `GET /health` is `loading` → `warming` → `ok` (or `error`). `/ask` works while warming.
- Warm-up: one turn, then the answer prompt of every curated passage is prefilled into
  llama-cpp-python's RAM cache (`HELPDESK_WARMUP=kb`; `all` adds the kb_auto entries, `none` skips).
  `HELPDESK_CACHE_MB` sizes the cache (default 2048, LRU).
- Image: `python:3.12.16-slim-bookworm` pinned by digest, every package pinned (`requirements-api.txt`,
  llama-cpp-python 0.3.36 built for portable AVX2), runs as a non-root user, read-only filesystem
  in compose, port bound to 127.0.0.1. Already have the GGUF? Mount it and set `GEMMA_GGUF`
  (commented lines in `docker-compose.yml`).
- Without Docker: `pip install -r requirements-api.txt llama-cpp-python` then
  `uvicorn helpdesk.api:app --port 8000`. `HELPDESK_LLM_BACKEND=http` uses a separate
  llama-server instead; `none` runs without a model.

## Faster replies on CPU (prompt cache)
Most reply time on a CPU is prefill (reading system prompt + passages).
Start the server with a disk cache, then warm it once:
```bash
python -m llama_cpp.server --model gemma-4-E2B-it-Q4_0.gguf --port 8080 \
    --n_ctx 4096 --n_threads 2 --cache true --cache_type ram
HELPDESK_CTX_K=1 python -m helpdesk.warmup    # after every server start
```
Use `ram`, not `disk`: llama-cpp-python 0.3.x's disk cache deletes an entry
when it is read, so warm-up entries evict each other. With `HELPDESK_CTX_K=1`
every prompt is system prompt + one passage + question, so one warm-up per
KB entry covers all questions.
```
```

## Test
```bash
HELPDESK_DENSE=0 pytest -q                    # 17 offline tests
python -m eval.run_eval                       # route / hit@3 / refusal gate, BM25 only
python -m eval.run_eval --dense               # same with BGE-M3; use it to tune HELPDESK_COS_MIN
python -m eval.run_eval --dense --full        # end-to-end with LLM + live tools
python -m eval.run_answer_eval                # 100 questions, answers scored with Gemma in the loop (~6 min)
```

## Status (7 Nov 2026)
- BM25 only, offline: route 20/20, retrieval hit@3 15/15, district 2/2.
- Known gap: the BM25 score cannot tell answerable from unanswerable
  questions (ration card and passport questions score as high as real ones),
  so the retrieval gate lets them through: 15/17. Refusal for these depends
  on the dense cosine threshold and the LLM's `UNSURE`; measure with `--full`.
- Knowledge base: 50 entries (28 from two panchayat citizen charters, 8 Nov), each with example questions that are indexed
  for retrieval but never sent to the LLM (written independently of the eval set).
- Knowledge base: Entries with `verify: true` come from a 2018
  SOP or a secondary site; check them against current official pages.
- Open-Meteo free API is non-commercial only.

## Space check (7-8 Nov 2026)
16 spoken turns on the Hugging Face Space (`REPLY_ENGINE=helpdesk`): reply stage
0.05-4.4 s (median 1.4 s), whole turn 1.7-6.2 s. Weather, news, out-of-scope
and passport refusals behaved as intended. Fixed after reading the log:
- Gemma sometimes refused in its own words ("...വിവരങ്ങൾ നൽകിയിട്ടില്ല") instead of
  `UNSURE`; these now map to the standard refusal.
- One answer was cut mid-sentence by the token limit; answers now end at the last
  full sentence (limit raised to 100 tokens).
- News read five headlines with publisher names (28 s of speech); now three, without
  publishers, and it says when a district has no local headline.
- Two missing topics added from IKM's K-SMART page (certificates online; Akshaya and
  Kudumbashree help desks), and entry titles weigh double in BM25.
- Still open: ASR mishears (e.g. "ജനന സർട്ടിഫിക്കറ്റ്" heard as "ജനങ്ങൾ സർട്ടിപ്പിക്ക്ഷ്")
  pull the wrong passage; the prompt now asks Gemma to answer UNSURE when the passage
  is about a different service.

## Knowledge base from citizen charters (8 Nov 2026)
Kadanad and Thachanattukara Grama Panchayat Pauravakasa Rekha 2026 (PDFs on
lsgkerala.gov.in). Kadanad's PDF uses the legacy ML-TT ASCII font, so its text was
converted with a Python 3 port of SMC's Payyans maps and checked against page
images; Thachanattukara's pages were read as images. Each entry cites the PDF page.
`verify=no` only where both charters agree (birth/death procedure and fees, marriage
documents and fees, building-permit documents, residence certificate, RTI).
Fee-band questions for marriage registration are computed in code like the birth/death
ones. Retrieval now boosts entries of the service named in the question (hit@1 12/15;
the three misses: two are answered by the fee calculator first, one combines a
marriage fee with the Akshaya service charge, which one passage cannot hold).
Known risk: "റേഷൻ കാർഡിൽ പേര് ചേർക്കാൻ" now retrieves the panchayat's
"name not in ration card" certificate; the answer prompt must decline it (Civil
Supplies handles adding names). Check q16 in the next `--full` run.

## Retrieval and routing rules (8 Nov 2026, night)
From two Space conversation logs (28 + 10 turns): 6 of 10 wrong answers came from a
wrong passage, 2 from missing or ambiguous facts, 2 from follow-ups; none from Gemma
misreading a correct passage. So, no fine-tuning; instead (`helpdesk/intent.py`):
- **Service filter.** Each service stem has a weight (2 names the service, 1 is generic
  such as "ലൈസൻസ്"); only the most specific services count, and they filter the
  candidate passages ("പട്ടിക്ക് ലൈസൻസ്" -> pet, not trade licence).
- **Facets.** Every KB entry lists what it answers (amount, documents, time, validity,
  compulsory, eligibility, process); the question's facet boosts matching entries.
  A strict facet (amount, time, validity, compulsory) that no entry of the service has
  is declined in code ("കെട്ടിട പെർമിറ്റിന്റെ കാലാവധിയെക്കുറിച്ച് എന്റെ പക്കൽ...")
  instead of Gemma borrowing the trade licence's "5 years".
- **No service, no KB answer.** LLM classification is gone: a question that names no
  service and follows none gets the scope message ("which service?" for a bare
  "how much?"). Other departments (passport, Aadhaar, driving licence, distance...)
  are out of scope even right after a service question.
- **Follow-ups** carry the previous question's service and facts: "കല്യാണം കഴിഞ്ഞു രണ്ട്
  മാസമായി." -> "രജിസ്ട്രേഷന് തുക എത്ര?" is computed as 220 rupees.
- **ASR fixes** for frequent mishearings (വേദന->വേതനം with തൊഴിൽ, പർമിത്->പെർമിറ്റ്,
  മുർഗം->മൃഗം, കാലാപതി->കാലാവധി, ...), plus strict fuzzy matching for long stems.
- `eval/conversations.jsonl`: the logged conversations as multi-turn tests (27 turns).

## Fixes from the 9 Nov Space log (21 turns)
- Marriage fee: "തുക" now counts as a money question for the fee calculator, so "രജിസ്ട്രേഷൻ
  തുക എത്ര?" (also as a follow-up) gets the full 120 / 220 / 370 schedule, not Gemma's partial one.
- ASR: "രണ്ടു നാസമായി" -> "രണ്ടു മാസമായി" (only after a number).
- Pets: cats, cows and goats route to the pet service and get "I only know about dogs and pigs"
  in code (was "ട്രേഡ് ലൈസൻസിന്റെ..."). കോഴി/എരുമ/പോത്ത് are left out: കോഴിക്കോട്, എരുമേലി, പോത്തൻകോട്.
- Weather: 15 tourist places (വാഗമൺ, മൂന്നാർ, കോവളം...) with their own coordinates; an unknown
  "<place>ിലെ" says so before the default district's weather instead of silently answering for Thiruvananthapuram.
- News: video-style titles lose their " | channel | date" tail.
- Scope: ministers, "കേരളത്തിൽ എത്ര ... ഉണ്ട്" counts and India questions are never follow-ups.
- KB: `svc-list-01` (services a panchayat offers) and `ptax-rate-01` (property tax: per sq m,
  2023 panchayat residential rates; verify=yes, news source).

## Tier 2: documents ingested automatically (9 Nov 2026)
Hand-written entries do not scale, so `ingest/` builds `data/docs.jsonl` from panchayat citizen
charters: download, OCR (Tesseract Malayalam: the PDFs' fonts are unreliable), table rows and
prose chunks, service tags, cross-charter agreement. See `ingest/README.md`.
- From two charters: ~600 passages, including services Tier 1 never covered (PHC, Krishi
  Bhavan, veterinary, schools, MGNREGS job card, advertisement permission...).
- Questions that name no known service reach them by title overlap; answers from one charter
  say which panchayat's charter they come from.
- `python -m ingest.check_kb` checks Tier-1 entries against the OCR of the pages they cite.
  It found six entries with details the cited page does not support (four "plain-paper
  application" errors from font decoding, one stamp-paper amount, one date); all fixed.
- Dense retrieval embeddings are cached in `data/emb-*.npy` (hundreds of passages).

### Fixes from the first Tier-2 Space log (9 Nov noon, 29 turns)
Tier-2 answers took 2.0-3.9 s on the Space even without prompt-cache warm-up.
- Follow-ups must fit: every distinctive word of a short question has to occur in the previous
  service's passages ("ബിരിയാണി എങ്ങനെ ഉണ്ടാക്കാം" after a marriage question was answered
  from the marriage KB). "ഇല്ല", "ശരി", "നന്ദി" get a short acknowledgement, not a passage.
- Title matching scores exact words above shared stems, prefers table rows over prose
  chunks, ignores place names and pointing words (അതിന്, ഇവിടെ), and unifies spellings
  (ഇൻഷുറൻസ്/ഇൻഷ്വറൻസ്, കുത്തിവയ്പ്/കുത്തിവെയ്പ്). ASR: കുറ്റികൾ, ചനന, ജനനെ.
- The whole log replays correctly in `eval/conversations.jsonl` (weather/news turns left out).

### Websites and weekly refresh (9 Nov)
`python -m ingest.web` crawls chosen sections of lsgkerala.gov.in (robots.txt obeyed) into
Tier 2; `ingest/changes.md` and `ingest/charter_changes.md` list what changed since the last
run, and a weekly GitHub Actions workflow opens a pull request with the refreshed data.
Page images are rendered with pypdfium2, so Tesseract is the only system dependency.

### 22 charters (9 Nov evening)
Kaggle OCR of 22 charters gave 3,712 passages; 242 officer/appeal-table and header rows are now
dropped (3,470 left). More data exposed ranking problems, fixed and covered by the 20-question
check now in `eval/conversations.jsonl`: Tier-1 boost 2.0 and charter prose x0.6; title matching
ignores place/action words (മുന്നിൽ, കൊണ്ടു), treats അനുവാദം = അനുമതി, compares a final chillu as
its consonant, breaks ties by how much of the title the question covers, and only overrides
within the service the keywords named; follow-ups keep the previous passage; a new two-word
service name beats a follow-up; no follow-up after the "other animals" decline; "സാധുവാണോ"
is a validity question; from-X-to-Y questions are out of scope.

## ASR slips: snap misheard words to the knowledge base by sound (10 Nov night)
`ASR_FIXES` lists mishearings one by one, but every log brings new ones (കാലാവരി, പീസ്, ഭീസ്,
പർമിത്തീന്റെ...). `helpdesk/asr_snap.py` handles the general case inside `fix_asr`, so routing, retrieval,
the fee calculator and the LLM prompt all see the corrected text:
- a word found nowhere in the documents (or the helpdesk's own replies) is replaced by the service, facet or
  KB-title word that sounds most like it: phonetic skeletons (ക/ഖ/ഗ/ഘ alike, ത/ദ/ധ alike, ഫ = പ, chillu =
  consonant, doubling ignored, inherent vowel kept), weighted edit distance <= 0.15 per sound, a different
  first sound costs extra, and two different candidates about as close means no change;
- two words heard as one are split if the first is a service name (ജനനരജിസ്ട്രേഷനായി);
- a real word in the wrong place is restored if it sounds like a service name the KB uses before the
  next word: "ജനറൽ രജിസ്ട്രേഷൻ" -> "ജനന രജിസ്ട്രേഷൻ" (3 of 24 logged clips);
- guards from failures seen while tuning: cat never becomes dog (a word that names a service is kept),
  a service name is brought in only when the question asks a service thing (fee, documents, time...)
  ("കഴിക്കാൻ പറ്റിയ കഞ്ഞി" stayed food), another ending of the same word is not a correction, and
  number words are left alone (the fee calculator deliberately computes nothing from an unreadable number).
- Cost: 0.3 s once at start, ~5 ms per unfamiliar word. `HELPDESK_ASR_SNAP=0` turns it off.

Measured with `python -m eval.asr_robust` (new): each word of a question is hit with probability p by one
random Malayalam ASR-style slip (aspiration/voicing, ര/റ, ല/ള, ന/ണ, ത/ട, vowel sign, chillu, merged
words); "same answer" = same route, tool and top passage as the clean question. Tuned on 500 dev
questions, reported on the test split (574 questions: synthetic test split + eval/questions.jsonl, 3 seeds):

| | clean questions unchanged | 24 logged clips, same answer | p = 0.15 | p = 0.3 |
|---|---|---|---|---|
| rules only (before) | 574/574 | 16/24 | 86.2% | 75.0% |
| rules + snap | 574/574 | 18/24 | 92.6% | 84.8% |

So about 46% (p = 0.15) and 39% (p = 0.3) of the answers lost to slips are won back, with no clean question
changed on the test split; `eval.run_synth` on clean questions: 175 -> 176 strict, 233 -> 234 same service;
`eval.run_eval` 20/20 unchanged; 187 tests pass. Caveats: the noise is a model of Whisper's slips, not
Whisper; the 24 clips were also used while developing (they are not a held-out set); and one guard (food
question -> dog licence) came from a test-split case, so the test numbers are slightly optimistic.
Review fix (11 Nov): "clean kept" counts answers, not wording. Checking the wording showed 36 of 1,694
clean eval questions rewritten, some into another meaning (രോഗികളുടെ -> രേഖകളുടെ, "പേര് മാറ്റാൻ വൈകിയാൽ" ->
"പേര് മരണ വൈകിയാൽ", വീടുകളുടെ -> വിത്തുകളുടെ, പതിക്കേണ്ടത് -> പുതുക്കേണ്ടത്), and the LLM read that text; one
out-of-scope question was answered (ഏത് പഴമാ നല്ലത് -> ഫലം -> a fruit-tree scheme). Now: a known stem + a
case or verb ending is not an unknown word; the context rule fires only before a service head noun
(രജിസ്ട്രേഷൻ, സർട്ടിഫിക്കറ്റ്...); a title word (not a service or facet word) comes in only if the question
already names a service or asks a service thing; and the LLM reads the question without snap corrections
(they steer routing and retrieval only). Clean questions rewritten: 36 -> 1 (വ്യാവസായിക -> വ്യവസായ; a test
guards the count). Cost, test split, 3 seeds, canon on: logged clips 18/24 (unchanged), p = 0.15 91.3%,
p = 0.3 83.4% against rules only 86.7%, 75.9% (about a third of the slip losses won back, down from ~40%),
clean kept 573/574; `eval.run_synth` out-of-scope declines back to 35/60 as with snap off; 194 tests pass.
Still wrong on the logged clips: number words (ഇരുപത്തെന്ന, നാപുര: deliberately not guessed),
"പ്രകാരം" heard for "എങ്ങനെ" (a real word: no sound rule can tell), "ഭീസിത്രേ" (ഫീസ് എത്ര run together).

## Answer eval: 100 questions with the LLM in the loop (11 Nov 2026)
`eval/answer_eval.jsonl`: one question per KB entry (50), colloquial variants (15), fee arithmetic (8),
government questions the KB does not cover (10), off-topic (10), weather (4), news (3). Drafted with Gemini
from `data/kb.jsonl`, then checked (every kb_id exists, every key fact is in the cited entry, fee answers match
`helpdesk/fees.py`, no copied KB example questions; 12 labels corrected). `python -m eval.run_answer_eval`
runs Gemma 4 E2B Q4_0 in-process and scores the answer itself (key facts, must-not facts, refusals), which
`eval.run_eval` and `eval.run_synth` do not. About 6 min on 4 CPU threads; `--rescore` re-scores saved answers.

Baseline 78/100: knowledge-base questions 39/50, colloquial 9/15, fee arithmetic 8/8, not-in-KB declined 8/10,
off-topic declined 7/10, weather and news routed 7/7. Fixed from it (now 83/100, 5 fixed, 0 broken):
- Fee calculator: a delay plus "register" no longer makes a fee question. "എത്ര" counts as how-much unless the
  question asks a time, a person or a procedure (എത്ര ദിവസം, ആരുടെ, അനുമതി, എങ്ങനെ, എവിടെ, രേഖ): "whose
  permission after a year" got ₹10 and "how many days for the certificate" got ₹220.
- "Is <document> compulsory?" (സമ്മതപത്രം, കരാർ, സാക്ഷ്യപത്രം, എഞ്ചിനീയർ...) is a documents question; it was
  declined as an unknown "compulsory" fact although the documents entry lists it.
- Out of scope: medical advice (symptoms at word start, "which medicine to take"; a fever question got the
  palliative-care row) and other states' services (a Tamil Nadu birth certificate got Kerala's K-SMART steps,
  now checked before the service rule).
Checked against the other suites: 208 tests pass (13 new); `eval.run_eval` unchanged; `eval.run_synth --split all`
with canonical services strict 425 -> 428, same service 619 -> 619. Tried and dropped: rejecting a one-word title
match in a question of 3+ (or 4+) content words. It fixed two off-topic cases here but declined a logged
conversation turn ("പരസ്യ ബോൾ വെക്കാൻ അനുകുതി വേണോ") and 13 real service questions in the synthetic set
(family planning, gobar gas, job card age): word count cannot tell a distinctive shared word from a generic one.
Still wrong (17): mostly retrieval, a charter row read instead of the KB entry (10); LIFE Mission and a joke
answered from lookalike rows; "ആർ ടി ഐ" not recognised; Gemma misreading a passage (2).
