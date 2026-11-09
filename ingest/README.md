# Tier 2: building the knowledge base from official documents

Hand-written entries (`data/kb.csv`, Tier 1) do not scale past a few dozen services. This
folder turns panchayat citizen charters (പൗരാവകാശ രേഖ) into retrieval passages automatically.

```
ingest/sources.csv  ->  fetch  ->  ingest/raw/<slug>.pdf
                    ->  extract (OCR)  ->  ingest/pages/<slug>.jsonl
                    ->  build_docs  ->  data/docs.jsonl  + ingest/report.md
                    ->  check_kb  ->  ingest/kb_check.md   (Tier 1 vs. the pages it cites)
```

```bash
pip install -r requirements-ingest.txt          # + poppler and tesseract (see that file)
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
