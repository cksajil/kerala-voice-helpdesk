"""ingest/pages/*.jsonl -> data/docs.jsonl (Tier 2: official documents, built, not hand-written).

    python -m ingest.build_docs

Table rows become one passage each ("<service>. അപേക്ഷിക്കേണ്ട വിധം: ... ഫീസ്: ...").
Prose pages are cut into ~500-character chunks at line breaks, starting a new chunk at
numbered headings. Chunks with low OCR confidence or little Malayalam are dropped.

The same service row in several charters is grouped; when two or more panchayats give the
same fee and time, the passage is marked verify=false and says "പൗരാവകാശ രേഖകൾ പ്രകാരം";
otherwise it names the one panchayat ("കടനാട് പഞ്ചായത്തിന്റെ...") because fees and
time limits can differ. Disagreements are listed in ingest/report.md for a human.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from helpdesk.intent import detect_services
from helpdesk.normalise import normalise

from .fetch import load_sources

HERE = Path(__file__).parent
PAGES = HERE / "pages"
OUT = HERE.parent / "data" / "docs.jsonl"
REPORT = HERE / "report.md"

MIN_CONF = 75
CHUNK = 500
_OLD_CHILLU = re.compile("([ണനരലളക])്‍")
_CHILLU = dict(zip("ണനരലളക", "ൺൻർൽൾൿ"))
_KSMART = re.compile(r"\bK\s*[-‐]?\s*s\s*mart\b", re.I)
_HEADING = re.compile(r"^\s*(\d{1,2}|[ivx]{1,4})[.)]\s+\S")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_LIST_RESTART = re.compile(r"(?<![\d.])1\s*[.)](?!\d)")
_PHONE = re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)|(?<!\d)0\d{2,4}[\s-]?\d{6,8}(?!\d)")


# Frequent Tesseract misreads in charter tables.
OCR_FIXES = [(re.compile(r"(?<![ഀ-ൿ])ജല(?![ഀ-ൿ])"), "ഇല്ല"), (re.compile(r"സാക്ഷ്യപ്രതം"), "സാക്ഷ്യപത്രം"),
             (re.compile(r"പ്രവർത്തി ദിവസ"), "പ്രവൃത്തി ദിവസ"), (re.compile(r"സയജന്യ"), "സൗജന്യ"),
             (re.compile(r"ഓഒഷധ"), "ഔഷധ"), (re.compile(r"രോഗ്രപതിരോധ"), "രോഗപ്രതിരോധ"),
             (re.compile(r"[/(]?(?<![ഀ-ൿ])ഗാമ(?=പഞ്ചായത്ത|സഭ)"), "ഗ്രാമ")]


def clean(s: str) -> str:
    """Display text: atomic chillu, no ZWJ/ZWNJ, 'K-SMART', single spaces. Case kept."""
    s = unicodedata.normalize("NFC", s or "")
    s = _OLD_CHILLU.sub(lambda m: _CHILLU[m.group(1)], s)
    s = s.replace("‍", "").replace("‌", "")
    s = _KSMART.sub("K-SMART", s)
    for pat, rep in OCR_FIXES:
        s = pat.sub(rep, s)
    s = re.sub(r"\s+([,.;:)])", r"\1", s)
    return re.sub(r"\s+", " ", s).strip(" ;,")


def ml_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    return sum("ഀ" <= c <= "ൿ" for c in letters) / len(letters) if letters else 0.0


def _service(title: str, context: str = "") -> str:
    """Service from the row's own title; the section heading only when the title names none
    (a section carried over from an earlier page tagged pension rows as building permits)."""
    for text in (title, context):
        found = detect_services(text) if text else []
        if found:
            return found[0]
    return "lsgd_other"


# Words of officer / appeal-authority tables and header cells. A "service" made of these
# ("ഒന്നാം അപ്പീല് അധികാരി", "ആകെ", "വർഷം") is not a service: 243 of 1,780 rows from 22
# charters were like this, 175 of them from one charter's officer tables.
_ROLE = ["അധികാരി", "ഉദ്യോഗസ്ഥ", "സെക്രട്ടറി", "അപ്പീല", "ഒന്നാം", "രണ്ടാം", "ചുമതല", "ഓഫീസർ", "ക്രമ",
         "നമ്പർ", "പേര്", "തസ്തിക", "ഫോൺ", "ജില്ലാ", "ചീഫ്", "ട്രൈബ്യൂണ", "ഡയറക്ടർ", "ഡെപ്യൂട്ടി",
         "രജിസ്ട്രാര", "അസിസ്റ്റന്റ"]


# Column headers of other charter layouts read as a service ("നല്കേണ്ട രേഖകൾ" x70, "അടക്കേണ്ട ഫീസ്",
# "സമയപരിധി"): a title made only of these words is a header.
_HEADER = ["നല്കേണ്ട", "നൽകേണ്ട", "നലകേണ്ട", "അടക്കേണ്ട", "അടയ്ക്കേണ്ട", "സമർപ്പിക്കേണ്ട", "ഹാജരാക്കേണ്ട",
           "രേഖ", "ഫീസ", "ഫീ", "സമയ", "കാലാവധി", "റിവിഷൻ", "അതോറിറ്റി", "ലഭ്യമാകുന്ന", "സേവന", "വിവര",
           "നിബന്ധന", "മുഖേന", "മാത്രം", "കെ.സ്മാർട്ട", "ksmart", "k-smart", "ഇ-ഫയലിംഗ", "അപേക്ഷ", "ദിവസ"]


def junk_title(title: str) -> bool:
    words = [w for w in normalise(title, strip_punct=True).split() if len(w) > 1]
    if len("".join(words)) < 6:
        return True
    if all(any(w.startswith(normalise(h)) for h in _HEADER) for w in words):
        return True
    role = sum(any(w.startswith(normalise(r)) for r in _ROLE) for w in words)
    return role >= max(1, len(words) / 2)


def row_doc(src: dict, page: int, i: int, row: dict) -> dict | None:
    title = clean(row["service"])
    if junk_title(title):
        return None
    section = clean(row.get("section", ""))
    title = re.sub(r"^[(]?\d{1,2}[.)]\s*", "", title)   # "2. സ്പെഷ്യാലിറ്റി വിഭാഗം" -> without the number
    # The section only helps tag the service: as a title prefix it was often a generic page
    # heading ("ഗ്രാമപഞ്ചായത്ത് സെക്രട്ടറി നൽകുന്ന സേവനങ്ങൾ").
    parts = [title + "."]
    labels = [("how", "അപേക്ഷിക്കേണ്ട വിധം"), ("conditions", "വേണ്ട രേഖകൾ, നിബന്ധനകൾ"),
              ("fee", "ഫീസ്"), ("time", "ലഭിക്കുന്ന സമയം"), ("officer", "ചുമതലയുള്ള ഉദ്യോഗസ്ഥൻ")]
    for key, label in labels:
        if row.get(key):
            parts.append(f"{label}: {clean(row[key])}.")
    facets = {"how": "process", "conditions": "documents", "fee": "amount", "time": "time"}
    return {
        "id": f"{src['slug']}-p{page}-r{i}", "tier": 2, "kind": "row",
        "service": _service(title, section),
        "title": title, "text": " ".join(parts), "questions": [],
        "facets": sorted({f for k, f in facets.items() if row.get(k)}),
        "lsgi": src["lsgi_ml"], "district": src.get("district", ""),
        "source": f"{src.get('url') or src['slug']} ({src['lsgi_en']} GP citizen charter {src.get('year', '')}, PDF p.{page})",
        "verify": True, "checked": "", "fee": clean(row.get("fee", "")), "time": clean(row.get("time", "")),
    }


def prose_chunks(text: str) -> list[str]:
    chunks, cur = [], []
    for line in (l.strip() for l in text.splitlines()):
        if not line:
            continue
        if cur and (_HEADING.match(line) or sum(map(len, cur)) + len(line) > CHUNK):
            chunks.append(" ".join(cur))
            cur = []
        cur.append(line)
    if cur:
        chunks.append(" ".join(cur))
    return [clean(c) for c in chunks]


def prose_docs(src: dict, page: int, text: str) -> list[dict]:
    out = []
    for i, c in enumerate(prose_chunks(text)):
        if len(c) < 80 or ml_ratio(c) < 0.6 or len(_PHONE.findall(c)) >= 2:
            continue                           # too short, mostly non-Malayalam, or a staff/phone list
        if len(_LIST_RESTART.findall(c)) >= 2:
            continue                           # a table read as prose: columns interleave ("1. ... 1. ... 2.")
        title = re.split(r"[.:;]", c, maxsplit=1)[0][:70].strip()
        out.append({
            "id": f"{src['slug']}-p{page}-c{i}", "tier": 2, "kind": "prose", "service": _service(c[:120], c),
            "title": title, "text": c, "questions": [], "facets": [],
            "lsgi": src["lsgi_ml"], "district": src.get("district", ""),
            "source": f"{src.get('url') or src['slug']} ({src['lsgi_en']} GP citizen charter {src.get('year', '')}, PDF p.{page})",
            "verify": True, "checked": "",
        })
    return out


def web_docs(slug: str, pg: dict) -> tuple[list[dict], int]:
    """Chunks of one crawled web page (ingest.web). The page title goes on every chunk so title
    matching finds them; English chunks are dropped (Gemma answers in Malayalam from the passage
    it reads, and mixed-language passages gave worse answers) and counted for the report."""
    out = []
    page_title = clean(pg.get("title", "")) or slug
    # Filter paragraphs before chunking: bilingual pages put English and Malayalam side by side.
    paras = [l for l in pg["text"].splitlines() if l.strip()]
    keep = [l for l in paras if ml_ratio(l) >= 0.6]
    english = len(paras) - len(keep)
    for i, c in enumerate(prose_chunks("\n".join(keep))):
        if len(c) < 80 or len(_PHONE.findall(c)) >= 2:
            continue
        title = page_title if i == 0 else f"{page_title}: {re.split(r'[.:;]', c, maxsplit=1)[0][:50].strip()}"
        out.append({
            "id": f"web-{slug}-p{pg['page']}-c{i}", "tier": 2, "kind": "web",
            "service": _service(page_title, c), "title": title, "text": c, "questions": [], "facets": [],
            "lsgi": "", "site": "LSGD", "district": "", "source": f"{pg['url']} ({page_title})",
            "modified": pg.get("modified", ""), "verify": True, "checked": "",
        })
    return out, english


def group_key(title: str) -> str:
    """Rough identity of a service across charters: its first three content words, stemmed."""
    words = [w for w in normalise(title, strip_punct=True).split() if len(w) > 2][:3]
    return " ".join(w[:5] for w in words)


def _same(a: str, b: str) -> bool:
    na, nb = _NUM.findall(a), _NUM.findall(b)
    return na == nb and (bool(na) or normalise(a)[:12] == normalise(b)[:12])


def main():
    sources = {s["slug"]: s for s in load_sources()}
    docs, skipped, english, junk = [], 0, 0, 0
    for pf in sorted(PAGES.glob("*.jsonl")):
        if pf.stem.startswith("web-"):
            for line in pf.open(encoding="utf-8"):
                got, en = web_docs(pf.stem[4:], json.loads(line))
                docs += got
                english += en
            continue
        src = sources.get(pf.stem) or {"slug": pf.stem, "lsgi_ml": pf.stem, "lsgi_en": pf.stem}
        for line in pf.open(encoding="utf-8"):
            pg = json.loads(line)
            if pg["conf"] < MIN_CONF:
                skipped += 1
                continue
            for i, row in enumerate(pg.get("rows", [])):
                d = row_doc(src, pg["page"], i, row)
                if d is None:
                    junk += 1
                else:
                    docs.append(d)
            if pg.get("text") and not pg.get("table_fallback"):   # a table we could not label: skip
                docs.extend(prose_docs(src, pg["page"], pg["text"]))

    # Cross-charter agreement on table rows.
    groups = defaultdict(list)
    for d in docs:
        if d["kind"] == "row":
            groups[group_key(d["title"])].append(d)
    agree = conflict = 0
    lines = ["# Charter cross-check", "", "Rows of the same service from different panchayats.", ""]
    for key, ds in sorted(groups.items()):
        lsgis = {d["lsgi"] for d in ds}
        if len(lsgis) < 2:
            continue
        base = ds[0]
        same = [d for d in ds if _same(d["fee"], base["fee"]) and _same(d["time"], base["time"])]
        if len({d["lsgi"] for d in same}) >= 2:
            agree += 1
            for d in same:
                d["verify"], d["agreed_by"] = False, sorted({x["lsgi"] for x in same})
        else:
            conflict += 1
            lines.append(f"## {base['title']}")
            lines += [f"- {d['lsgi']} (`{d['id']}`): ഫീസ് `{d['fee']}` · സമയം `{d['time']}`" for d in ds]
            lines.append("")
    for d in docs:
        d.pop("fee", None)
        d.pop("time", None)
    OUT.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs), encoding="utf-8")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    by_kind = defaultdict(int)
    for d in docs:
        by_kind[d["kind"]] += 1
    print(f"wrote {len(docs)} passages to {OUT} ({dict(by_kind)}); {skipped} low-confidence pages skipped, "
          f"{junk} officer/header rows dropped")
    if english:
        print(f"{english} English web paragraphs left out: crawl the Malayalam version of those pages")
    print(f"cross-charter: {agree} services agree, {conflict} differ (see {REPORT})")


if __name__ == "__main__":
    main()
