# Live check after a Space update (10 questions)

Ask each by voice on the Space (https://sajilck-malayalam-voice-assistant.hf.space) in this order;
2 and 3 are one conversation. Tick when the reply does what the right-hand column says. The
questions are new: none is in the held-out set or the tuned-on sets, so they test the release, not
the tuning. Startup log first: `helpdesk: HELPDESK_CANON=1, ... passages` and the warm-up count.

| # | Ask | Expected |
|---|-----|----------|
| 1 | ജനന സർട്ടിഫിക്കറ്റ് ഓൺലൈനായി കിട്ടാൻ എന്ത് ചെയ്യണം? | Service answer: K-SMART Civil Registration, download online; a source is named. |
| 2 | വിവാഹം കഴിഞ്ഞിട്ട് രണ്ട് മാസമായി, രജിസ്റ്റർ ചെയ്യാൻ എത്ര രൂപ? | Worked out in code (no LLM): past 45 days → 120 + 100 fine = 220 rupees. |
| 3 | അതിന് എന്തൊക്കെ രേഖകൾ വേണം? | Follow-up stays on marriage registration (documents list), not "which service?". |
| 4 | ചാലക്കുടിയിൽ ഇന്ന് മഴയുണ്ടോ? | Weather for ചാലക്കുടി (തൃശ്ശൂർ ജില്ല), not Thiruvananthapuram. |
| 5 | കട്ടപ്പനയിൽ നാളെ തണുപ്പുണ്ടാകുമോ? | Tomorrow's weather, said to be for the ഇടുക്കി district centre (Kattappana has no coordinates). |
| 6 | ഇന്നത്തെ പ്രധാന ന്യൂസ് പറയൂ | News headlines (ന്യൂസ് was not recognised before 13 Nov). |
| 7 | മഴവെള്ള സംഭരണി ഇല്ലെങ്കിൽ വീടിന് പെർമിറ്റ് കിട്ടുമോ? | Building-permit answer or an honest "not sure" — not a weather report. |
| 8 | ഡിഗ്രി സർട്ടിഫിക്കറ്റ് കളഞ്ഞുപോയാൽ പഞ്ചായത്തിൽ നിന്ന് എന്തെങ്കിലും ചെയ്യാമോ? | Not a weather report; a decline or "which service?" is fine (not a panchayat service). |
| 9 | ഹരിതകർമ്മസേനയുടെ പരാതി എവിടെ കൊടുക്കണം? | Grievance (K-SMART public grievance) or a decline — no invented phone numbers. |
| 10 | ബിരിയാണി ഉണ്ടാക്കുന്നത് എങ്ങനെ? | Scope message (panchayat services, weather, news); no recipe. |

Also check: each reply under ~20 s after warm-up; the spoken reply matches the text; no English
plumbing ("UNSURE", "passage", "context") is ever read out.
