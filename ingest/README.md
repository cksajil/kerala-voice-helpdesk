# Tier 2: building the knowledge base from official documents

Hand-written entries (`data/kb.csv`, Tier 1) do not scale past a few dozen services. This
folder turns panchayat citizen charters (പൗരാവകാശ രേഖ) into retrieval passages automatically.

```
ingest/sources.csv  ->  fetch  ->  ingest/raw/<slug>.pdf
                    ->  extract (OCR)  ->  ingest/pages/<slug>.jsonl
                    ->  build_docs  ->  data/docs.jsonl  + ingest/report.md
                    ->  check_kb  ->  ingest/kb_check.md   (Tier 1 vs. the pages it cites)
                    ->  canon (Gemini, cached)  ->  data/canon.jsonl
                    ->  kb_auto  ->  data/kb_auto.jsonl + ingest/kb_auto_report.md
```

Tesseract with Malayalam data is the only system tool (pages are rendered with pypdfium2).
On Colab/Kaggle: `apt-get install -y tesseract-ocr tesseract-ocr-mal`. On a Mac where
Homebrew builds from source (macOS 14 Intel), skip brew upgrades and add only the language:
`curl -L -o "$(brew --prefix)/share/tessdata/mal.traineddata" https://github.com/tesseract-ocr/tessdata_best/raw/main/mal.traineddata`

```bash
pip install -r requirements-ingest.txt
python -m ingest.fetch                          # downloads, 2 s apart; manifest in ingest/raw/
python -m ingest.extract                        # ~3 s a page; only new PDFs
python -m ingest.build_docs
python -m ingest.check_kb                       # exit 1 if a Tier-1 entry disagrees with its page
pytest -q                                       # conversation tests also run on the new data
```

## Why OCR
The charters' text layers are unreliable: ML-TT (ASCII) fonts, or Unicode fonts with broken
glyph maps. Decoding the Kadanad charter's text layer turned "Ksmart അപേക്ഷ" into garbage that
was read as "plain-paper application" in four Tier-1 entries; `check_kb` now catches that kind
of error. Tesseract reads the rendered page, so the font does not matter.

## How a page becomes passages
- **Tables** (most charters): pdfplumber finds the ruled cells; the page is OCRed once
  (sparse mode) and each word goes to the cell containing it. Columns are labelled from the
  header row (service, how to apply, conditions, fee, time) and later pages map cells to
  header columns by x-position. Each row becomes one passage:
  `<service>. അപേക്ഷിക്കേണ്ട വിധം: ... വേണ്ട രേഖകൾ: ... ഫീസ്: ... ലഭിക്കുന്ന സമയം: ...`
  with facets from the filled columns, so "how much / how long" questions can be declined in
  code when no passage of that service has the fact.
- **Prose** (scheme descriptions): ~500-character chunks at line breaks. Chunks with low OCR
  confidence, little Malayalam, phone lists, or interleaved table columns are dropped.
- **Service tag**: `helpdesk.intent.detect_services` on the row title (the section heading only
  as a fallback); untagged rows are `lsgd_other` and are reached by title overlap (below).

## How the helpdesk uses Tier 2
- Tier-1 entries get a 1.5x score boost; the same row from several charters appears once.
- A question that names no known service is matched by **title overlap** with the passages
  ("പരസ്യം വെക്കാൻ അനുമതി" -> "പരസ്യങ്ങൾ പ്രദർശിപ്പിക്കുന്നതിന് അനുമതി"), not by BM25 alone,
  so general-knowledge questions still get the scope message.
- Answers from a single charter end with "(<panchayat> പഞ്ചായത്തിന്റെ പൗരാവകാശ രേഖ പ്രകാരം.
  നിങ്ങളുടെ പഞ്ചായത്തിൽ വ്യത്യാസമുണ്ടാകാം.)". When two or more charters give the same fee and
  time, `build_docs` marks the row verified and the answer says "പൗരാവകാശ രേഖകൾ പ്രകാരം".
- Only Tier-1 passages are pre-filled in the Space's prompt cache; Tier-2 prompts still reuse
  the cached system prompt.

## Adding sources
Add a row to `sources.csv`: a PDF url, or the panchayat's "Citizen Charter" page (the first
charter PDF linked there is downloaded; an inline charter is saved as HTML). `kb_alias` is the
name used in `data/kb.csv` sources, for `check_kb`. More charters mostly add *agreement*
(the same template is used state-wide), which is what turns a single-panchayat fact into a
verified one; read `ingest/report.md` for rows where panchayats disagree.

## Websites (lsgkerala.gov.in and others)
```bash
python -m ingest.web --links https://lsgkerala.gov.in/ml     # see the site's sections, pick some
# add rows to ingest/web_sources.csv: slug, start url, path prefix, max pages
python -m ingest.web                                          # crawl -> ingest/pages/web-<slug>.jsonl
python -m ingest.build_docs
```
- Obeys robots.txt. lsgkerala.gov.in (Drupal) allows content pages and disallows search,
  login, admin and elected members' personal-detail pages; the crawler also never follows
  `/lbelection/` links. 2 s between requests.
- Keeps each page's main text (trafilatura), with its URL, title and Last-Modified date.
  Only Malayalam paragraphs become passages: crawl the `/ml/` version of a page when there
  is one. Answers say "(തദ്ദേശ സ്വയംഭരണ വകുപ്പിന്റെ വെബ്സൈറ്റ് പ്രകാരം.)".
- PDFs the crawl finds (orders, circulars, charters) are listed in `ingest/web_pdfs.txt`;
  add the useful ones to `sources.csv` so they go through OCR.

## Keeping it current
- `ingest.fetch --force` re-downloads every charter; one whose file changed has its OCR
  dropped (so `extract` redoes it) and is listed in `ingest/charter_changes.md`.
- `ingest.web` writes `ingest/changes.md`: new pages, and changed pages with the numbers
  that disappeared or appeared ("removed: 150; added: 200") -- the fee and day changes a
  Tier-1 entry may need.
- `.github/workflows/refresh-kb.yml` runs all of this every Monday 02:47 IST (or by hand from
  the Actions tab), runs the tests, and opens a pull request with the changes. Nothing reaches
  the Space until you merge it and copy `data/docs.jsonl`. GitHub's runners are outside India;
  if LSGD sites refuse them, the run shows FAILED lines and you run the same commands on Colab.

## Running it on Kaggle (long runs)
`notebooks/kaggle_ingest.ipynb` runs the whole pipeline with **Save Version → Save & Run All**, so a
run continues after you close the browser (up to 12 h; 4 CPU cores, `--jobs 4`). Its output folder
can be added as Input to the next version to resume, and `EXTRA_SOURCES` / `EXTRA_WEB_SOURCES` add
charters or website sections for one run. `ingest.extract` writes `<slug>.jsonl.partial` and renames
it only when a document is finished, so a timeout never leaves a half-read charter behind.

## Canonical services and a held-out eval (Gemini, `ingest/canon.py`)
`notebooks/kaggle_canon.ipynb` (Kaggle secret `GEMINI_API_KEY`, free tier, no billing):
1. Groups the charter rows by service (no LLM). Header-only titles ("നല്കേണ്ട രേഖകൾ", "സമയപരിധി")
   are dropped first by `build_docs`.
2. Gemini, 8 groups per request: service or not, clean Malayalam title, synonyms, facts common to the
   rows, each panchayat's fee/time, 8 citizen questions. Tier-1 entries get questions only; one request
   gives out-of-scope questions.
3. Checked in code: any number not in the group's rows is dropped; non-services are dropped.
4. `data/canon.jsonl`: one passage per service, 4 questions indexed. `eval/synth_questions.jsonl`: the
   other 4 (never indexed) + Tier-1 + out-of-scope questions.
5. `python -m eval.run_synth` compares today's rows with the canonical services on those held-out
   questions. The Space uses canon only with `HELPDESK_CANON=1`, after the numbers say so.

Resumable: answers are cached in `ingest/canon_cache.jsonl`; when the daily quota ends, run again the
next day with the previous output as Input.
Known gap: some charters lay a service out as label/value lines (`അപേക്ഷ നല്കേണ്ടത്: …`) instead of
columns; those services are not extracted yet.

## Tier-1-schema entries from the charters (`ingest/kb_auto.py`)
`python -m ingest.kb_auto` (no network, no LLM) promotes canonical services that appear in at least two
panchayats' charters to entries with the schema of `data/kb.jsonl`: title, text, indexed questions,
facets, `source` (charter URL + PDF page), `checked` (fetch date from `ingest/raw/manifest.json`) and
`verify`. A fee or time is stated as a fact only when two or more charters give the same value;
otherwise each panchayat's value is listed with "check with your panchayat". `verify` is false only when
every source is on an official government domain and every stated fee/time is agreed. Entries that a
curated entry already covers are dropped (asymmetric title-word coverage; see the report's "possible
duplicates" table for the ones to check by hand). The help desk loads them with `HELPDESK_KB_AUTO=1`
at `HELPDESK_AUTO_BOOST`; the curated entries are never changed.

robots.txt (RFC 9309): a missing robots.txt (4xx) allows; one that cannot be read (5xx, network error)
disallows the whole site.
