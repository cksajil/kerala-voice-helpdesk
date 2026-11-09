"""Canonical services + question variants from the charter rows, with an LLM (task B), and a
held-out question set to measure retrieval with (task A).

    GEMINI_API_KEY=... python -m ingest.canon                       # all groups, resumable
    python -m ingest.canon --model gemini-flash-lite-latest --batch 8 --max-requests 40
    python -m ingest.canon --dry-run                                 # groups + prompt size only

1. Group (no LLM): the 1,500 charter rows are the same few hundred services written 22 ways.
   Rows of one service tag whose titles share most of their words go into one group.
2. One request per batch of groups. For each group the model returns: is it a citizen service at
   all, a clean Malayalam title, synonyms, facts common to the rows, each panchayat's fee and time
   copied from its row, and 8 questions a citizen would ask. Tier-1 entries get questions only.
   One more request gives out-of-scope questions.
3. Checks in code, not trust: every number the model writes must occur in the group's rows,
   otherwise that fact is dropped; groups it calls "not a service" are dropped; replies that do
   not parse are retried once and then skipped.
4. Outputs
   data/canon.jsonl          one passage per canonical service (kind "canon"): title, summary,
                             per-panchayat fee/time, member row ids; 4 of the 8 questions are
                             indexed for retrieval (doc2query)
   eval/synth_questions.jsonl the other 4 questions per group (never indexed), 4 per Tier-1
                             entry, and the out-of-scope set: the held-out eval for eval/run_synth.py
   ingest/canon_cache.jsonl  raw model answers per group: a rerun skips groups already done

The model reads only public charter text. Free-tier Gemini requests may be used by Google to
improve its products, so never send user logs or audio through this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from helpdesk.intent import SERVICES
from helpdesk.retriever import _content_words, _match_score, load_kb

HERE = Path(__file__).parent
ROOT = HERE.parent
DOCS, KB = ROOT / "data" / "docs.jsonl", ROOT / "data" / "kb.jsonl"
CANON, SYNTH = ROOT / "data" / "canon.jsonl", ROOT / "eval" / "synth_questions.jsonl"
CACHE = HERE / "canon_cache.jsonl"
SERVICE_NAMES = sorted(set(SERVICES) | {"lsgd_other"})
FACETS = ["amount", "documents", "time", "validity", "compulsory", "eligibility", "process"]
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")

# ---------------------------------------------------------------- 1. grouping (no LLM)


@lru_cache(maxsize=None)
def _words(title: str) -> tuple[str, ...]:
    return tuple(_content_words(title))


@lru_cache(maxsize=None)
def _ms(a: str, b: str) -> float:
    return _match_score(a, b)


def title_similarity(a: str, b: str) -> float:
    """Symmetric share of distinctive title words that match (exact 1, inflected 0.8, stem 0.6)."""
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    ab = sum(max(_ms(x, y) for y in wb) for x in wa)
    ba = sum(max(_ms(y, x) for x in wa) for y in wb)
    return (ab + ba) / (len(wa) + len(wb))


def group_rows(rows: list[dict], threshold: float = 0.5) -> list[dict]:
    """Greedy clustering within each service tag; the shortest title starts a group. A row joins
    the group whose first title it matches best (>= threshold). OCR splits words, so the threshold
    is loose; the model then says when a group mixes two services."""
    groups: list[dict] = []
    by_service = defaultdict(list)
    for r in rows:
        by_service[r["service"]].append(r)
    for service, rs in sorted(by_service.items()):
        local: list[dict] = []
        for r in sorted(rs, key=lambda r: (len(r["title"]), r["id"])):
            scored = [(title_similarity(r["title"], g["title"]), i) for i, g in enumerate(local)]
            best = max(scored, default=(0.0, -1))
            if best[0] >= threshold:
                local[best[1]]["rows"].append(r)
            else:
                local.append({"title": r["title"], "service": service, "rows": [r]})
        groups += local
    for g in groups:
        g["id"] = "g" + hashlib.sha1("|".join(sorted(r["id"] for r in g["rows"])).encode()).hexdigest()[:10]
    return groups


def split_of(key: str) -> str:
    """Stable dev/test split for reporting; both halves are held out from the index."""
    return "test" if int(hashlib.sha1(key.encode()).hexdigest(), 16) % 3 == 0 else "dev"

# ---------------------------------------------------------------- 2. prompts

GROUP_PROMPT = """You are cleaning a knowledge base of Kerala panchayat citizen services, written in Malayalam.
Each GROUP below holds rows OCR-read from different panchayats' citizen charters that probably describe
the same service. OCR text has broken words and stray characters; read through them.

For EACH group return one JSON object with exactly these keys:
- "group_id": the group's id, copied.
- "is_service": false if the rows are not a service a citizen can apply for or ask about (table headers,
  officer or appeal lists, notes, staff names, garbled text); then leave the other fields empty.
- "same_service": false if the rows clearly describe two or more different services.
- "service": one of {services}.
- "title_ml": a short, clean, standard Malayalam name for the service (no panchayat names, no numbering).
- "synonyms_ml": 3 to 6 other ways people name this service in Malayalam (colloquial words, common English loanwords).
- "summary_ml": 1 to 3 Malayalam sentences with ONLY facts that the rows state and that hold for all of them
  (how to apply, documents, who is eligible). Do not add anything that is not in the rows.
- "variants": for each row, {{"row_id", "fee", "time"}}: the fee and the time limit copied exactly from that row
  ("" when the row does not say).
- "facets": which of {facets} the rows answer.
- "questions_ml": 8 different questions a citizen would SPEAK to a voice helpdesk about this service, in natural
  Malayalam: mix formal and colloquial, short and long, ask about fee, documents, time or eligibility only when
  the rows answer them, at least 2 without the service's official name, no panchayat names, no English script.

Return a JSON array of these objects, one per group, in the same order. No text outside the JSON.

GROUPS:
{groups}"""

TIER1_PROMPT = """Below are entries from a Malayalam knowledge base of Kerala panchayat services.
For EACH entry write 6 different questions a citizen would SPEAK to a voice helpdesk that this entry answers:
natural Malayalam, formal and colloquial, at least 2 without the entry's exact title words, no English script.
Return a JSON array of {{"id": ..., "questions_ml": [...]}} in the same order. No text outside the JSON.

ENTRIES:
{entries}"""

SCOPE_PROMPT = """Write 60 short questions in natural spoken Malayalam that people might ask a Kerala panchayat
voice helpdesk but that are NOT about panchayat or municipality services, weather or news. Mix: Kerala general
knowledge, travel and distances, food and recipes, other government departments (passport, Aadhaar, ration card,
KSEB electricity, police, driving licence, PSC, banks), health advice, chit-chat about the assistant, and a few
nonsense or half-heard sentences. Return a JSON array of strings. No text outside the JSON."""


def group_block(g: dict) -> str:
    lines = [f'<group id="{g["id"]}" service_tag="{g["service"]}">']
    for r in g["rows"][:12]:                                  # 12 panchayats are plenty to read a service
        lines.append(f'  <row id="{r["id"]}" panchayat="{r.get("lsgi", "")}">{r["text"][:700]}</row>')
    lines.append("</group>")
    return "\n".join(lines)

# ---------------------------------------------------------------- 3. model call and checks


def parse_json(text: str):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.S)
        return json.loads(m.group(0)) if m else None


class Gemini:
    """Thin wrapper: JSON mode where the model supports it, polite retries on 429/5xx, and a
    clean stop when the daily quota is used up (the run is resumable from the cache)."""

    def __init__(self, model: str, key: str | None = None, pause: float = 4.0):
        from google import genai
        self.genai = genai
        self.client = genai.Client(api_key=key or os.environ["GEMINI_API_KEY"])
        self.model, self.pause, self.requests = model, pause, 0

    def __call__(self, prompt: str) -> str:
        from google.genai import errors, types
        json_mode = not self.model.startswith("gemma")
        cfg = types.GenerateContentConfig(temperature=0.3, max_output_tokens=16384,
                                          **({"response_mime_type": "application/json"} if json_mode else {}))
        for attempt in range(5):
            try:
                time.sleep(self.pause)
                self.requests += 1
                return self.client.models.generate_content(model=self.model, contents=prompt, config=cfg).text
            except errors.APIError as e:
                msg = str(e)
                if e.code == 429 and re.search(r"per ?day|PerDay|daily", msg, re.I):
                    raise QuotaExhausted(msg) from e
                if e.code in (429, 500, 502, 503, 504):
                    wait = 30 * (2 ** attempt)
                    m = re.search(r"retry(?:Delay| in)\D*(\d+)", msg)
                    if m:
                        wait = int(m.group(1)) + 2
                    print(f"  {e.code}: waiting {wait}s", flush=True)
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError("model kept failing")


class QuotaExhausted(Exception):
    pass


def numbers_ok(text: str, source: str) -> bool:
    src = source.replace(",", "")
    return all(n.replace(",", "") in src for n in _NUM.findall(text or ""))


def check_group(ans: dict, g: dict) -> dict | None:
    """The model's answer for one group, or None. Facts with numbers not in the rows are dropped."""
    if not isinstance(ans, dict) or not ans.get("is_service") or not ans.get("title_ml"):
        return None
    rows = {r["id"]: r for r in g["rows"]}
    source = " ".join(r["text"] for r in g["rows"])
    summary = ans.get("summary_ml", "") if numbers_ok(ans.get("summary_ml", ""), source) else ""
    variants = []
    for v in ans.get("variants") or []:
        r = rows.get(v.get("row_id"))
        if r is None:
            continue
        fee = v.get("fee", "") if numbers_ok(v.get("fee", ""), r["text"]) else ""
        tm = v.get("time", "") if numbers_ok(v.get("time", ""), r["text"]) else ""
        if fee or tm:
            variants.append({"row_id": r["id"], "lsgi": r.get("lsgi", ""), "fee": fee, "time": tm})
    qs = [q.strip() for q in ans.get("questions_ml") or [] if isinstance(q, str) and 6 <= len(q.strip()) <= 200]
    qs = list(dict.fromkeys(q for q in qs if not re.search(r"[A-Za-z]{4,}", q)))
    service = ans.get("service") if ans.get("service") in SERVICE_NAMES else g["service"]
    return {"title": ans["title_ml"].strip(), "synonyms": [s for s in ans.get("synonyms_ml") or [] if isinstance(s, str)][:6],
            "summary": summary.strip(), "variants": variants, "questions": qs[:8], "service": service,
            "facets": [f for f in ans.get("facets") or [] if f in FACETS],
            "same_service": ans.get("same_service", True) is not False}


def canon_doc(g: dict, c: dict) -> dict:
    """One retrieval passage for a canonical service. Half of its questions are indexed."""
    lsgis = sorted({r.get("lsgi", "") for r in g["rows"] if r.get("lsgi")})
    parts = [c["title"] + "."]
    if c["summary"]:
        parts.append(c["summary"])
    else:                                                     # model's summary failed the number check
        parts.append(g["rows"][0]["text"].split(". ", 1)[-1])
    for v in c["variants"][:4]:
        bits = [f"ഫീസ്: {v['fee']}" if v["fee"] else "", f"സമയം: {v['time']}" if v["time"] else ""]
        parts.append(f"{v['lsgi']}: " + ", ".join(b for b in bits if b) + ".")
    if len(c["variants"]) > 4:
        parts.append("മറ്റ് പഞ്ചായത്തുകളിൽ വ്യത്യാസമുണ്ടാകാം.")
    indexed = c["questions"][::2]                             # 0, 2, 4, 6 -> index; 1, 3, 5, 7 -> eval
    return {"id": f"canon-{g['id']}", "tier": 2, "kind": "canon", "service": c["service"], "title": c["title"],
            "text": " ".join(parts), "questions": indexed + c["synonyms"], "facets": c["facets"] or
            sorted({f for r in g["rows"] for f in r.get("facets", [])}),
            "members": [r["id"] for r in g["rows"]], "lsgis": lsgis, "lsgi": lsgis[0] if len(lsgis) == 1 else "",
            "agreed_by": lsgis if len(lsgis) >= 2 else None, "verify": True,
            "source": "; ".join(dict.fromkeys(r["source"] for r in g["rows"][:3]))}

# ---------------------------------------------------------------- 4. run


def load_cache() -> dict:
    out = {}
    if CACHE.exists():
        for line in CACHE.open(encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                out[d["key"]] = d["answer"]
    return out


def run(call, batch: int = 8, max_requests: int | None = None, dry_run: bool = False) -> dict:
    docs = [json.loads(l) for l in DOCS.open(encoding="utf-8")]
    rows = [d for d in docs if d["kind"] == "row"]
    groups = group_rows(rows)
    tier1 = load_kb(KB)
    cache = load_cache()
    print(f"{len(rows)} rows -> {len(groups)} groups; {len(tier1)} Tier-1 entries; {len(cache)} cached answers")
    todo = [g for g in groups if g["id"] not in cache]
    jobs = [("groups", todo[i:i + batch]) for i in range(0, len(todo), batch)]
    t1_todo = [e for e in tier1 if f"t1:{e['id']}" not in cache]
    jobs += [("tier1", t1_todo[i:i + 15]) for i in range(0, len(t1_todo), 15)]
    if "scope" not in cache:
        jobs.append(("scope", None))
    if dry_run:
        sizes = [len(GROUP_PROMPT.format(services=SERVICE_NAMES, facets=FACETS,
                                         groups="\n".join(group_block(g) for g in b))) for k, b in jobs if k == "groups"]
        print(f"{len(jobs)} requests to make; group prompts {min(sizes, default=0)}-{max(sizes, default=0)} chars")
        return {"groups": len(groups), "requests": len(jobs)}
    done = 0
    with CACHE.open("a", encoding="utf-8") as f:
        for kind, b in jobs:
            if max_requests is not None and done >= max_requests:
                print(f"stopping after {done} requests (--max-requests)")
                break
            if kind == "groups":
                prompt = GROUP_PROMPT.format(services=SERVICE_NAMES, facets=FACETS,
                                             groups="\n".join(group_block(g) for g in b))
            elif kind == "tier1":
                prompt = TIER1_PROMPT.format(entries="\n".join(
                    f'<entry id="{e["id"]}">{e["title"]}: {e["text"][:500]}</entry>' for e in b))
            else:
                prompt = SCOPE_PROMPT
            try:
                out = None
                for _ in range(2):                            # one retry for unparseable JSON
                    out = parse_json(call(prompt))
                    if isinstance(out, list):
                        break
            except QuotaExhausted:
                print("daily quota used up: run again tomorrow, finished groups are cached")
                break
            done += 1
            if not isinstance(out, list):
                print(f"  {kind}: unparseable reply, skipped")
                continue
            if kind == "groups":
                by_id = {a.get("group_id"): a for a in out if isinstance(a, dict)}
                for g in b:
                    f.write(json.dumps({"key": g["id"], "answer": by_id.get(g["id"])}, ensure_ascii=False) + "\n")
            elif kind == "tier1":
                by_id = {a.get("id"): a for a in out if isinstance(a, dict)}
                for e in b:
                    f.write(json.dumps({"key": f"t1:{e['id']}", "answer": by_id.get(e["id"])}, ensure_ascii=False) + "\n")
            else:
                f.write(json.dumps({"key": "scope", "answer": out}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  request {done}/{len(jobs)} ({kind}) ok", flush=True)
    return write_outputs(groups, tier1)


def merge_same(checked: list[tuple[dict, dict]]) -> list[tuple[dict, dict]]:
    """Groups the title grouping kept apart (OCR splits words) but the model named the same: one
    service, one passage. Otherwise retrieval keeps one of two identical titles and the other's
    questions look like misses."""
    from helpdesk.normalise import normalise
    merged: dict = {}
    for g, c in checked:
        key = (c["service"], normalise(c["title"], strip_punct=True))
        if key not in merged:
            merged[key] = ({**g, "rows": list(g["rows"])}, {**c, "variants": list(c["variants"]),
                                                            "questions": list(c["questions"])})
            continue
        g0, c0 = merged[key]
        g0["rows"] += g["rows"]
        c0["variants"] += c["variants"]
        c0["questions"] += c["questions"]
        c0["synonyms"] = list(dict.fromkeys(c0["synonyms"] + c["synonyms"]))
        c0["summary"] = c0["summary"] or c["summary"]
    return list(merged.values())


def write_outputs(groups: list[dict], tier1: list[dict]) -> dict:
    cache = load_cache()
    canon, synth, stats, checked = [], [], defaultdict(int), []
    for g in groups:
        if g["id"] not in cache:
            stats["not_done"] += 1
            continue
        c = check_group(cache[g["id"]], g)
        if c is None:
            stats["not_a_service"] += 1
            continue
        if not c["same_service"]:
            stats["mixed_group"] += 1                          # kept, but flagged for a human
        raw = cache[g["id"]] or {}
        stats["facts_dropped"] += (bool(raw.get("summary_ml")) and not c["summary"]) + \
            sum(1 for v in raw.get("variants") or [] if v.get("fee") or v.get("time")) - len(c["variants"])
        checked.append((g, c))
    merged = merge_same(checked)
    stats["merged_away"] = len(checked) - len(merged)
    for g, c in merged:
        doc = canon_doc(g, c)
        canon.append(doc)
        for q in c["questions"][1::2]:
            synth.append({"q": q, "kind": "canon", "expect": [doc["id"], *doc["members"]],
                          "service": c["service"], "split": split_of(g["id"])})
        stats["canon"] += 1
    for e in tier1:
        a = cache.get(f"t1:{e['id']}") or {}
        for q in [q for q in a.get("questions_ml") or [] if isinstance(q, str)][:4]:
            synth.append({"q": q.strip(), "kind": "tier1", "expect": [e["id"]], "service": e["service"],
                          "split": split_of(e["id"])})
    for i, q in enumerate([q for q in cache.get("scope") or [] if isinstance(q, str)]):
        synth.append({"q": q.strip(), "kind": "out_of_scope", "expect": [], "service": "",
                      "split": "test" if i % 3 == 0 else "dev"})
    CANON.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in canon), encoding="utf-8")
    SYNTH.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in synth), encoding="utf-8")
    print(f"wrote {len(canon)} canonical services to {CANON} and {len(synth)} eval questions to {SYNTH}")
    print("  " + ", ".join(f"{k} {v}" for k, v in sorted(stats.items())))
    return {"canon": len(canon), "synth": len(synth), **stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.getenv("CANON_MODEL", "gemini-flash-lite-latest"))
    ap.add_argument("--batch", type=int, default=8, help="groups per request")
    ap.add_argument("--max-requests", type=int)
    ap.add_argument("--pause", type=float, default=4.0, help="seconds between requests")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--outputs-only", action="store_true", help="rebuild outputs from the cache, no requests")
    args = ap.parse_args()
    if args.outputs_only:
        docs = [json.loads(l) for l in DOCS.open(encoding="utf-8")]
        write_outputs(group_rows([d for d in docs if d["kind"] == "row"]), load_kb(KB))
        return
    call = None if args.dry_run else Gemini(args.model, pause=args.pause)
    run(call, batch=args.batch, max_requests=args.max_requests, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
