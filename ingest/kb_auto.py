"""Tier-1-schema knowledge-base entries built automatically from the citizen charters.

    python -m ingest.kb_auto            # data/kb_auto.jsonl + ingest/kb_auto_report.md (no network, no LLM)

Pipeline position (see ingest/README.md):
    fetch -> extract (OCR) -> build_docs -> canon (Gemini, cached) -> kb_auto

`ingest.canon` already groups the charter rows into canonical services and asks Gemini (free tier,
public charter text only, every answer cached in ingest/canon_cache.jsonl) for a clean title, a
summary, per-panchayat fee/time and citizen questions, dropping every number that is not in the rows.
This step turns the best-supported of those services into entries with the schema of the curated
data/kb.jsonl, so the help desk can treat them as Tier 1 (HELPDESK_KB_AUTO=1):

- support: a service must appear in the charters of at least MIN_LSGIS panchayats;
- facts: a fee or a time is stated as a fact only when at least two charters give the same value
  (compared after normalising: "ഫീസ് ഇല്ല" = "ഇല്ല", "അന്നേ ദിവസം" = "അതേ ദിവസം", numbers + unit);
  otherwise up to three panchayats' own values are listed with "check with your panchayat";
- dedupe: dropped when its title matches a curated entry's title (same service, title similarity
  >= DUP_SIM) and between auto entries the better-supported one is kept;
- verify: false only when every source is on an official government domain *and* every fee/time it
  states is agreed by two or more charters; anything else (single-charter values, model summary for a
  group the model called mixed, a non-government source) is verify: true;
- source: each member charter's URL with its PDF page; checked: the charters' fetch date
  (ingest/raw/manifest.json).
The curated entries are never changed.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

from helpdesk.normalise import normalise

HERE = Path(__file__).parent
ROOT = HERE.parent
KB, CANON, DOCS = ROOT / "data" / "kb.jsonl", ROOT / "data" / "canon.jsonl", ROOT / "data" / "docs.jsonl"
OUT, REPORT = ROOT / "data" / "kb_auto.jsonl", HERE / "kb_auto_report.md"
MANIFESTS = [HERE / "raw" / "manifest.json", HERE / "raw" / "web_manifest.json"]
MIN_LSGIS = 2
DUP_SIM = 0.75
OFFICIAL = (".gov.in", ".nic.in")          # lsgkerala.gov.in, cdn.web.lsgkerala.gov.in, kerala.gov.in ...
_ROW_SLUG = re.compile(r"^(.*)-p\d+-r\d+$")
_URL = re.compile(r"https?://\S+")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_ML = re.compile(r"[ഀ-ൿ]")


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()] if path.exists() else []


def official(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host.endswith(OFFICIAL) or host in ("gov.in", "kerala.gov.in")


# ----------------------------------------------------------------- fee / time normalisation

_NO_FEE = ("ഇല്ല", "ഇല്ലാ", "ഉല്ല", "സൗജന്യ", "ഫീസില്ല", "nil", "free")
_SAME_DAY = ("അതേ ദിവസ", "അതെ ദിവസ", "അന്നേ ദിവസ", "അന്നേദിവസ", "അന്ന് തന്നെ", "അന്നുതന്നെ", "അന്ന്തന്നെ",
             "same day", "ഉടൻ", "ഉടനെ", "ഉടനടി")
_UNITS = [("മണിക്കൂ", "മണിക്കൂർ"), ("ആഴ്ച", "ആഴ്ച"), ("മാസ", "മാസം"), ("വർഷ", "വർഷം"), ("ദിവസ", "ദിവസം")]


def fee_key(s: str) -> str | None:
    """'ഇല്ല' for no fee, 'rs:5' / 'rs:2,5' for amounts; None when the cell says nothing usable."""
    t = normalise(s).lower()
    nums = [n.replace(",", "") for n in _NUM.findall(t)]
    if nums:
        return "rs:" + ",".join(sorted(set(nums), key=float))
    if any(w in t for w in _NO_FEE):
        return "ഇല്ല"
    return None


def time_key(s: str) -> str | None:
    t = normalise(s).lower()
    if any(normalise(w) in t for w in _SAME_DAY):
        return "അതേ ദിവസം"
    nums = _NUM.findall(t)
    if not nums:
        return None
    unit = next((u for stem, u in _UNITS if normalise(stem) in t), "ദിവസം")
    work = "പ്രവൃത്തി " if ("പ്രവൃത്തി" in t or "പ്രവ്യത്തി" in t or "പ്രവര്ത്തി" in t) and unit == "ദിവസം" else ""
    return f"{nums[0].replace(',', '')} {work}{unit}"


def fee_text(key: str) -> str:
    return "ഫീസ് ഇല്ല" if key == "ഇല്ല" else "ഫീസ് " + " / ".join(f"{n} രൂപ" for n in key[3:].split(","))


def consensus(values: list[tuple[str, str]], keyf) -> tuple[str | None, int, list[tuple[str, str]]]:
    """values: (lsgi, raw cell). -> (agreed key or None, how many panchayats agree, usable values)."""
    usable, seen = [], set()
    for lsgi, raw in values:
        k = keyf(raw) if raw else None
        if k and lsgi not in seen:             # one vote per panchayat
            usable.append((lsgi, k))
            seen.add(lsgi)
    if not usable:
        return None, 0, []
    key, n = Counter(k for _, k in usable).most_common(1)[0]
    return (key, n, usable) if n >= 2 else (None, 0, usable)


# ----------------------------------------------------------------- dedupe

def _similar(a: str, b: str) -> float:
    from ingest.canon import title_similarity
    return title_similarity(a, b)


def _soft(x: str, y: str) -> float:
    """canon's word match (exact 1, inflected 0.8, stem 0.6), plus a shared stem of 4+ letters that
    covers 60% of the shorter word (ദത്തെടുക്കുന്ന ~ ദത്തെടുത്ത, വൈകിയുള്ള ~ വൈകി): 0.7."""
    from ingest.canon import _ms
    best = _ms(x, y)
    if best < 0.7:
        n = 0
        while n < min(len(x), len(y)) and x[n] == y[n]:
            n += 1
        if n >= 4 and n >= 0.6 * min(len(x), len(y)):
            best = 0.7
    return best


def coverage(title: str, target: str) -> float:
    """Share of the title's distinctive words found in target. Asymmetric on purpose:
    "കിണർ നിർമ്മാണ പെർമിറ്റ്" shares two words with the curated building-permit entry but its own
    distinctive word (കിണർ) is not there, so it is a different service."""
    from ingest.canon import _words
    wa, wb = _words(title), _words(target)
    if not wa or not wb:
        return 0.0
    return sum(max(_soft(x, y) for y in wb) for x in wa) / len(wa)


def best_match(entry: dict, curated: list[dict]) -> tuple[float, str | None]:
    """Best title/question coverage of the auto title by a curated entry. Same service: title or
    questions (questions carry synonyms: വികലാംഗ for ഭിന്നശേഷി). Any service: title only, since the
    charter's service tag can differ from the curated one (ownership certificate: certificate vs
    property_tax)."""
    best = (0.0, None)
    for c in curated:
        texts = [c["title"], *c.get("questions", [])] if c["service"] == entry["service"] else [c["title"]]
        best = max(best, (max(coverage(entry["title"], t) for t in texts), c["id"]), key=lambda x: x[0])
    return best


def duplicate_of(entry: dict, curated: list[dict]) -> str | None:
    """Dropped when a curated entry already answers this service: best_match >= DUP_SIM, or a half
    match (>= 0.5, same service) whose every distinctive title word occurs in the curated entry."""
    cov, cid = best_match(entry, curated)
    if cov >= DUP_SIM:
        return cid
    for c in curated:
        if c["service"] != entry["service"]:
            continue
        half = max(coverage(entry["title"], t) for t in [c["title"], *c.get("questions", [])])
        whole = " ".join([c["title"], *c.get("questions", []), c["text"]])
        if half >= 0.5 and coverage(entry["title"], whole) >= 0.99:
            return c["id"]
    return None


# ----------------------------------------------------------------- build

def provenance(row_ids: list[str], rows: dict, manifest: dict) -> tuple[list[str], str]:
    """Sources (URL + PDF page, as build_docs wrote them) and the latest fetch date."""
    sources, dates = [], []
    for rid in row_ids:
        r = rows.get(rid)
        if r and r.get("source"):
            sources.append(r["source"])
        m = _ROW_SLUG.match(rid)
        if m and m.group(1) in manifest:
            dates.append(manifest[m.group(1)].get("fetched", ""))
    return list(dict.fromkeys(sources)), max((d for d in dates if d), default="")


def build(canon: list[dict], curated: list[dict], rows: dict, cache: dict, manifest: dict) -> tuple[list[dict], dict]:
    stats = defaultdict(int)
    dropped: list[tuple[str, str, str]] = []
    cands = []
    for c in canon:
        lsgis = c.get("lsgis") or []
        if len(lsgis) < MIN_LSGIS:
            stats["one_charter"] += 1
            continue
        gid = c["id"].removeprefix("canon-")
        raw = cache.get(gid) or {}
        mixed = raw.get("same_service") in (False, "false", "no")
        variants = [v for v in raw.get("variants") or [] if isinstance(v, dict)]
        # canon.check_group already dropped numbers not in the rows; repeat it on what is used here
        from ingest.canon import numbers_ok
        member_text = {m: rows[m]["text"] for m in c.get("members", []) if m in rows}
        lsgi_of = {m: rows[m].get("lsgi", "") for m in member_text}
        fees, times = [], []
        for v in variants:
            rid = str(v.get("row_id", "")).strip()
            if rid not in member_text:
                continue
            for field, bucket in (("fee", fees), ("time", times)):
                val = str(v.get(field) or "").strip()
                if val and numbers_ok(val, member_text[rid]):
                    bucket.append((lsgi_of[rid], val))
        fee_k, fee_n, fee_vals = consensus(fees, fee_key)
        time_k, time_n, time_vals = consensus(times, time_key)
        summary = str(raw.get("summary_ml") or "").strip()
        if summary and not numbers_ok(summary, " ".join(member_text.values())):
            summary = ""
        if not summary:
            stats["no_summary"] += 1
            dropped.append((c["id"], c["title"], "no grounded summary"))
            continue
        parts = [c["title"] + ".", summary]
        facts_agreed = True
        if fee_k:
            parts.append(f"{fee_text(fee_k)} ({fee_n} പഞ്ചായത്തുകളുടെ പൗരാവകാശ രേഖ പ്രകാരം).")
        if time_k:
            parts.append(f"ലഭിക്കുന്ന സമയം: {time_k} ({time_n} പഞ്ചായത്തുകളുടെ പൗരാവകാശ രേഖ പ്രകാരം).")
        local = []
        if not fee_k and fee_vals:
            facts_agreed = False
            local += [f"{l}: {fee_text(k)}" for l, k in fee_vals[:3]]
        if not time_k and time_vals:
            facts_agreed = False
            local += [f"{l}: സമയം {k}" for l, k in time_vals[:3]]
        if local:
            parts.append("പഞ്ചായത്തുകൾ തമ്മിൽ വ്യത്യാസമുണ്ട് (" + "; ".join(local) + "). നിങ്ങളുടെ പഞ്ചായത്തിൽ ഉറപ്പാക്കുക.")
        text = " ".join(parts)
        if len(_ML.findall(text)) < 0.6 * len(re.sub(r"[\s\d.,:;()/-]", "", text)):
            stats["not_malayalam"] += 1
            dropped.append((c["id"], c["title"], "too little Malayalam"))
            continue
        sources, fetched = provenance(c.get("members", []), rows, manifest)
        urls = [u for s in sources for u in _URL.findall(s)]
        all_official = bool(urls) and all(official(u) for u in urls)
        entry = {
            "id": "auto-" + gid, "service": c["service"], "title": c["title"], "text": text,
            "questions": c.get("questions", []), "facets": c.get("facets", []),
            "source": "; ".join(sources[:4]) + (f"; +{len(sources) - 4} more" if len(sources) > 4 else ""),
            "verify": not (all_official and facts_agreed and not mixed),
            "checked": fetched, "fetched": fetched,
            "kind": "kb_auto", "lsgis": lsgis, "agreed_by": lsgis, "members": c.get("members", []),
            "origin": "ingest.kb_auto from ingest.canon (Gemini, cached) over OCR of citizen charters",
        }
        dup = duplicate_of(entry, curated)
        if dup:
            stats["duplicate_of_curated"] += 1
            dropped.append((entry["id"], entry["title"], f"duplicate of curated {dup}"))
            continue
        cands.append(entry)
    # between auto entries: better-supported first, then drop near-identical titles of the same service
    cands.sort(key=lambda e: (-len(e["lsgis"]), e["id"]))
    out: list[dict] = []
    for e in cands:
        twin = next((o for o in out if o["service"] == e["service"] and _similar(e["title"], o["title"]) >= 0.7), None)
        if twin:
            stats["duplicate_of_auto"] += 1
            dropped.append((e["id"], e["title"], f"near-duplicate of {twin['id']}"))
            continue
        out.append(e)
    stats["kept"] = len(out)
    near = [(e, *best_match(e, curated)) for e in out]
    near = sorted([n for n in near if n[1] >= 0.5], key=lambda n: -n[1])
    stats["possible_duplicates"] = len(near)
    return out, {"stats": dict(stats), "dropped": dropped, "near": near}


def report(entries: list[dict], info: dict, n_curated: int) -> str:
    by_svc = Counter(e["service"] for e in entries)
    ver = Counter(e["service"] for e in entries if e["verify"])
    lines = ["# Automatically built KB entries (`data/kb_auto.jsonl`)", "",
             f"Built by `python -m ingest.kb_auto` from `data/canon.jsonl` (cached Gemini output over OCR of the "
             f"citizen charters). Curated entries (`data/kb.jsonl`, {n_curated}) are unchanged.", "",
             f"- auto entries: **{len(entries)}**, verify=true: **{sum(e['verify'] for e in entries)}** "
             f"({100 * sum(e['verify'] for e in entries) / max(1, len(entries)):.0f}%)",
             f"- from official government domains only: {sum(all(official(u) for u in _URL.findall(e['source'])) for e in entries)}",
             "- " + ", ".join(f"{k} {v}" for k, v in sorted(info["stats"].items())), "",
             "| service | entries | verify=true |", "|---|---|---|"]
    lines += [f"| {s} | {n} | {ver[s]} |" for s, n in by_svc.most_common()]
    lines += ["", "## Entries", "", "| id | service | charters | verify | title |", "|---|---|---|---|---|"]
    lines += [f"| {e['id']} | {e['service']} | {len(e['lsgis'])} | {'yes' if e['verify'] else 'no'} | {e['title']} |"
              for e in entries]
    lines += ["", "## Possible duplicates kept (check by hand)", "",
              "Kept entries whose title half-matches a curated entry (0.5 <= coverage < DUP_SIM).", "",
              "| id | title | closest curated | coverage |", "|---|---|---|---|"]
    lines += [f"| {e['id']} | {e['title']} | {cid} | {cov:.2f} |" for e, cov, cid in info.get("near", [])]
    lines += ["", "## Not promoted", "", "| id | title | reason |", "|---|---|---|"]
    lines += [f"| {i} | {t} | {r} |" for i, t, r in info["dropped"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    from ingest.canon import load_cache
    curated = _load(KB)
    canon = _load(CANON)
    rows = {d["id"]: d for d in _load(DOCS)}
    manifest = {}
    for m in MANIFESTS:
        if m.exists():
            manifest.update(json.loads(m.read_text(encoding="utf-8")))
    entries, info = build(canon, curated, rows, load_cache(), manifest)
    OUT.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries), encoding="utf-8")
    REPORT.write_text(report(entries, info, len(curated)), encoding="utf-8")
    print(f"wrote {len(entries)} entries to {OUT.relative_to(ROOT)}; "
          f"verify=true {sum(e['verify'] for e in entries)}; " + ", ".join(f"{k} {v}" for k, v in sorted(info["stats"].items())))


if __name__ == "__main__":
    main()
