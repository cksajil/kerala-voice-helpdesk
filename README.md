# Kerala Voice Helpdesk — reply stage (v1, day 1)

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
Still wrong on the logged clips: number words (ഇരുപത്തെന്ന, നാപുര: deliberately not guessed),
"പ്രകാരം" heard for "എങ്ങനെ" (a real word: no sound rule can tell), "ഭീസിത്രേ" (ഫീസ് എത്ര run together).
