"""Fetch the official sources for the core services (ingest/core_sources.csv) and survey the LSGD
Panchayat Wiki (panchayatwiki.lsgkerala.gov.in). Runs on Kaggle: the work container cannot reach
lsgkerala.gov.in.

    python -m ingest.core_fetch                 # sources + wiki survey
    python -m ingest.core_fetch --wiki-pages 0  # sources only

Rules: robots.txt is checked for every URL (ingest.fetch.allowed, RFC 9309); one request every
DELAY seconds per run; every file keeps its URL, fetch date, HTTP status and SHA-256 in the manifest.
Output: ingest/raw/core/<sha12>.<ext> + manifest.json, and ingest/raw/core/wiki_titles.json (every page
title the wiki's MediaWiki API lists, with namespaces and site statistics), so the relevant pages
can be chosen before any bulk download.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import time
from pathlib import Path

import requests

from ingest.fetch import UA, allowed

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "raw" / "core"
WIKI = "https://panchayatwiki.lsgkerala.gov.in"
DELAY = 2.0


def get(url: str, **kw):
    time.sleep(DELAY)
    return requests.get(url, headers=UA, timeout=60, **kw)


def fetch_sources() -> list[dict]:
    rows = list(csv.DictReader((ROOT / "core_sources.csv").open(encoding="utf-8")))
    today = dt.date.today().isoformat()
    manifest = []
    for r in rows:
        rec = {**r, "fetched": today}
        if not allowed(r["url"]):
            rec["status"] = "robots_disallow"
        else:
            try:
                resp = get(r["url"])
                rec["status"] = resp.status_code
                if resp.ok:
                    body = resp.content
                    ext = "pdf" if body[:4] == b"%PDF" else "html"
                    sha = hashlib.sha256(body).hexdigest()
                    (OUT / f"{sha[:12]}.{ext}").write_bytes(body)
                    rec.update(file=f"{sha[:12]}.{ext}", sha256=sha, bytes=len(body),
                               content_type=resp.headers.get("content-type", ""))
            except requests.RequestException as e:
                rec["status"] = f"error: {type(e).__name__}"
        manifest.append(rec)
        print(rec["status"], rec.get("bytes", ""), r["url"][:100], flush=True)
    return manifest


def survey_wiki(max_pages: int) -> dict:
    """Site statistics and the list of page titles through the MediaWiki API (no page bodies)."""
    api = WIKI + "/api.php"
    out = {"api": api, "robots_api": allowed(api), "robots_root": allowed(WIKI + "/"), "fetched": dt.date.today().isoformat()}
    if not out["robots_api"]:
        out["note"] = "robots.txt disallows api.php; nothing listed"
        return out
    try:
        r = get(api, params={"action": "query", "meta": "siteinfo", "siprop": "statistics|namespaces", "format": "json"})
        info = r.json()["query"]
        out["statistics"] = info.get("statistics")
        out["namespaces"] = {k: v.get("*") or v.get("name", "") for k, v in info.get("namespaces", {}).items()}
    except Exception as e:
        out["error"] = f"siteinfo: {type(e).__name__}: {str(e)[:200]}"
        return out
    titles, cont = [], {}
    for ns in [int(k) for k in out["namespaces"] if int(k) >= 0]:
        cont = {}
        while len(titles) < max_pages:
            params = {"action": "query", "list": "allpages", "apnamespace": ns, "aplimit": 500, "format": "json", **cont}
            try:
                d = get(api, params=params).json()
            except Exception as e:
                out.setdefault("errors", []).append(f"allpages ns={ns}: {type(e).__name__}")
                break
            titles += [{"ns": ns, "title": p["title"], "pageid": p["pageid"]} for p in d["query"]["allpages"]]
            if "continue" not in d:
                break
            cont = d["continue"]
    out["n_titles"] = len(titles)
    out["titles"] = titles
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki-pages", type=int, default=50000, help="cap on titles listed (0 = skip the survey)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = fetch_sources()
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.wiki_pages:
        s = survey_wiki(args.wiki_pages)
        (OUT / "wiki_titles.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
        print("wiki:", {k: v for k, v in s.items() if k not in ("titles", "namespaces")})


if __name__ == "__main__":
    main()
