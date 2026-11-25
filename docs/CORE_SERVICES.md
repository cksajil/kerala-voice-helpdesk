# Core services for v1 (draft for the owner's approval, 14 Nov 2026)

Plan: a small number of services answered well, each with a hand-checked entry built from an official
source written for citizens (K-SMART help pages, LSGD pages, government orders, official guidelines),
not from charter tables. Every entry answers the same questions where they apply: **what to bring, fee,
time, where/how to apply, who is eligible**, with the source URL, page and fetch date.

Status against the 50 curated entries in `data/kb.jsonl` (14 Nov):
**covered** = an entry answers the usual questions; **partial** = some facets missing or failures seen;
**missing** = no curated entry. "Seen" = failed on a real or test question.

## A. Panchayat services to cover (30)

| # | Service | Status | Gap / evidence |
|---|---|---|---|
| 1 | Birth registration (on time, who reports, how) | partial | vehicle birth misread (held-out h05) |
| 2 | Death registration, death at home or in hospital (documents, who reports) | **missing** | held-out g01 answered with late-registration documents |
| 3 | Late birth/death registration (fees, documents) | covered | |
| 4 | Name in the birth register; corrections to birth/death certificates | partial | corrections not covered |
| 5 | Birth/death/marriage certificate copies on K-SMART | covered | |
| 6 | Marriage registration (documents, fee, time, compulsory, online/video KYC) | covered | follow-up after "compulsory" declined (live, 14 Nov) — retrieval bug, not content |
| 7 | Building permit: documents, fee bands, time | partial | fee band misread (h15); small-house rules |
| 8 | Building permit rules citizens ask about: rainwater storage, setbacks, small plots | **missing** | live 14 Nov: "permit in 30 days" for rainwater question |
| 9 | Occupancy certificate / completion | **missing** | held-out g02 |
| 10 | New house (building) number | **missing** | live 14 Nov, turns 18–20 |
| 11 | Building tax (property tax): rate, how to pay, appeal, exemption, demolished building | partial | demolished-house question (h18) |
| 12 | Ownership certificate; transfer of ownership in panchayat records | covered | |
| 13 | Ownership certificate for KSEB / water connection | **missing** | common real-world reason for the certificate |
| 14 | Residence certificate | covered | |
| 15 | Certificates from the panchayat president (identity, character, not-remarried, etc.) | covered | |
| 16 | Trade licence: new, documents, fee bands, renewal | covered | |
| 17 | Dog and other pet licence | covered (thin) | one entry |
| 18 | Well permit / well construction | **missing** | held-out g04 answered from charter text only |
| 19 | Welfare pensions: old age, widow, disability, agricultural worker, unmarried women — eligibility, amount | covered | car-ownership eligibility garbled (h19) |
| 20 | Pension application process, mustering / life certificate, income limit | partial | |
| 21 | Unemployment allowance | covered | |
| 22 | MGNREGS job card and work demand | **missing** | held-out g03 answered from charter text only |
| 23 | Haritha Karma Sena: user fee, collection, complaints | **missing** | held-out g05 |
| 24 | Waste / street-light / road complaints to the panchayat | partial | general grievance entry only |
| 25 | Grievances against the local body (K-SMART) and reply time | covered | |
| 26 | Right to information (RTI) | covered | |
| 27 | LIFE Mission housing (eligibility, how to apply) | **missing** | |
| 28 | K-SMART: what it is, registration, Akshaya / Kudumbashree help desk, service charges | covered | |
| 29 | Primary health centre services (medical / health certificates) | covered (thin) | |
| 30 | Panchayat office basics: which office, ward member, office hours | **missing** | |

## B. Questions to decline clearly (another department)

Decline with a pointer to the right office instead of answering from a panchayat passage:

- Ration card: new card, adding or correcting names, card type (Civil Supplies) — live 14 Nov answered wrongly
- Income, caste, community, nativity, legal-heir, solvency certificates (Village Office / Revenue)
- Electricity bills and connections (KSEB) — held-out o02 answered as property tax
- Water bills and connections (Kerala Water Authority)
- Land records, survey, mutation, land tax (Revenue)
- Aadhaar, voter ID changes (UIDAI / Election Commission) — voter list already declined

## C. Sources to look for, per service (Gemini with search grounding finds the pages; we fetch them)

K-SMART help and FAQ pages; LSGD department and Information Kerala Mission pages; government orders
for fees (current, dated); Kerala Panchayat Building Rules (permit, rainwater storage, occupancy,
building number); civil registration guidelines (birth, death, marriage); Social Security Pension
guidelines (welfare pensions); LIFE Mission; MGNREGS Kerala; Suchitwa Mission (Haritha Karma Sena).

## Owner decisions

1. Approve, drop or add services in table A (target 25–30).
2. Confirm the decline list in B.
3. Spot-check about 25 drafted entries (about an hour) before they go into the knowledge base.
