# Held-out test set v1 (`eval/heldout_v1.jsonl`)

46 Malayalam questions written on 13 Nov 2026 for measuring v1. Frozen: `tests/test_heldout.py`
pins the file's SHA-256, so a fix can never be tuned by editing it. A corrected set is a new file
(`heldout_v2.jsonl`), and numbers from v1 and v2 are not mixed.

| category | n | what is expected |
|---|---|---|
| `in_scope` | 24 | panchayat service answerable from the current Tier-1 KB; route `service_faq`, one of `expect_kb` retrieved |
| `kb_gap` | 5 | panchayat service with no Tier-1 entry today (death-registration documents, occupancy certificate, MGNREGS job card, well permit, Haritha Karma Sena user fee) |
| `weather` | 5 | route `weather`, right `district` (and `day`) |
| `news` | 5 | route `news` (`district` when one is named) |
| `out_of_scope` | 7 | declined; `expect_kb` = `refuse` |

Styles: 22 colloquial, 11 ASR-style, 13 plain. ASR-style items use the spellings real recognisers produce:
IndicConformer's consonant + virama for a chillu (ലൈസന്സ്, പെര്മിറ്റ്, വാര്ത്തകള്, വോട്ടര്), loanword slips
(സർട്ടിപ്പിക്കറ്റ്), aspiration/sibilant slips (ഭിന്നശേശി, അപെക്ഷിച്ചാൽ, കാലാവസ്ത), and no punctuation.
Traps on purpose: "കൊല്ലം" meaning *year* (h02, h03 vs the Kollam district), two district names in a
service question (h09), a district name in a bus question (o06), lookalikes of in-scope services
(o01 land mutation vs building transfer, o02 electricity bill vs online tax).

## Fields
`id`, `category`, `style` (plain / colloquial / asr), `question`, `route` (`service_faq` / `weather` / `news` /
`other`), `expect_kb` (list of acceptable Tier-1 ids, `"gap"`, `"refuse"` or null for tools), `service`,
`expect_fee` (string, only where the question asks an amount), `key_facts` (each item is a set of
alternatives separated by `|`; numbers are compared after removing commas), `must_not`, and optional
`district`, `day`, `place`, `partial_kb`, `note`.

## Scoring rubric (Phase 3)
- **Routing:** final route equals `route`. Out-of-scope items count as correct if the reply is a decline,
  whatever the route.
- **Retrieval hit@1 / hit@3:** any `expect_kb` id among the top 1 / 3 passages (in-scope items only).
- **Answer correct** (in-scope): every `key_facts` item matches the answer, no `must_not` item does, and the
  reply is not a decline. Every answer is dumped to CSV with the automatic verdict for hand checking.
- **Fee correct:** the answer states `expect_fee` (10 questions with a fee).
- **Refusal precision / recall:** positives are the 7 out-of-scope items; a decline on an in-scope item is
  a false positive.
- **KB gaps:** reported separately, by hand: grounded answer of the right service, decline (acceptable), or
  wrong (another service's facts or invented facts).
- **Weather / news:** route plus `district` / `day` from the tool call; live numbers are not scored.

## How it was made, and what it does not guarantee
Written by Claude Code from the KB entry texts and the fee bands in `helpdesk/fees.py`, not from the
20-question set or the KB example questions. `python -m eval.check_heldout` checks that every `expect_kb` id
exists, that the band fees agree with `fees.py`, and that no question is a near copy of a question in
`questions.jsonl`, `answer_eval.jsonl`, `synth_questions.jsonl`, `conversations.jsonl` or the KB examples
(character-trigram Jaccard below 0.5; the highest is 0.48). Nine first drafts that asked the same fact as an
`answer_eval.jsonl` question (e.g. the permit fee for 250 sq m) were replaced before the freeze.
The help desk was **not** run on these questions before the freeze.
Limits: with 50 KB entries, many items share a *topic* with older (tuned-on) sets, though not the same
question; fee questions necessarily reuse the same fee bands; written questions with simulated ASR
spellings are not real speech.
