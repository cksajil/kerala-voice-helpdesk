"""Crawl chosen sections of LSGD websites into ingest/pages/web-<slug>.jsonl (Tier 2).

    python -m ingest.web                       # every section in ingest/web_sources.csv
    python -m ingest.web --only egov --max 20  # one section, at most 20 pages
    python -m ingest.web --links https://lsgkerala.gov.in/ml   # list a page's links, to pick sections

Each row of web_sources.csv is a start page and a path prefix; the crawl follows links on the
same host whose path starts with that prefix, breadth first, up to max_pages. It obeys robots.txt
(lsgkerala.gov.in allows content pages; it disallows search, login and elected members'
personal-detail pages), waits 2 s between requests, and keeps only the page's main text
(trafilatura). PDFs it finds are listed in ingest/web_pdfs.txt for you to add to sources.csv;
they are not downloaded here.

Each page's text is hashed; ingest/changes.md lists pages that are new or changed since the
last crawl, so a revised fee or rule shows up as a diff to review rather than going unnoticed.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlsplit

import requests

from .fetch import UA, allowed

HERE = Path(__file__).parent
PAGES = HERE / "pages"
STATE = HERE / "raw" / "web_manifest.json"
_HREF = re.compile(r"""href=["']([^"'#]+)["']""", re.I)
_SKIP_EXT = re.compile(r"\.(jpe?g|png|gif|svg|css|js|zip|docx?|xlsx?|pptx?|mp4|mp3)$", re.I)
_SKIP_PATH = re.compile(r"/(user|admin|search|node/add|comment|lbelection)/|[?&](page|sort|order)=\d{3,}")


def load_web_sources(path: Path = HERE / "web_sources.csv") -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("slug")]


def links(html: str, base: str) -> list[str]:
    out = []
    for h in _HREF.findall(html):
        u, _ = urldefrag(urljoin(base, h.strip()))
        if u.startswith("http"):
            out.append(u)
    return list(dict.fromkeys(out))


def main_text(html: str, url: str) -> tuple[str, str]:
    """(title, main text) of a page; boilerplate (menus, footers) removed."""
    import trafilatura
    text = trafilatura.extract(html, url=url, include_tables=True, include_comments=False,
                               favor_recall=True, output_format="txt") or ""
    meta = trafilatura.extract_metadata(html)
    title = (meta.title if meta and meta.title else "").split("|")[0].strip()
    return title, text


def crawl(src: dict, max_pages: int, get=requests.get, sleep=time.sleep) -> tuple[list[dict], list[str]]:
    start = src["url"]
    host = urlsplit(start).netloc
    prefix = src.get("path_prefix") or urlsplit(start).path
    queue, seen, pages, pdfs = deque([start]), {start}, [], []
    while queue and len(pages) < max_pages:
        url = queue.popleft()
        if not allowed(url):
            continue
        try:
            r = get(url, headers=UA, timeout=60)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  FAILED {url}: {e}")
            continue
        sleep(2)
        if "html" not in r.headers.get("Content-Type", "text/html"):
            continue
        title, text = main_text(r.text, url)
        if text.strip():
            pages.append({"page": len(pages) + 1, "kind": "prose", "text": text, "conf": 100.0,
                          "url": url, "title": title, "modified": r.headers.get("Last-Modified", ""),
                          "sha256": hashlib.sha256(text.encode()).hexdigest()})
            print(f"  {len(pages):3} {title[:60]} ({len(text)} chars)", flush=True)
        for u in links(r.text, r.url):
            if u.startswith("http://") and urlsplit(u).netloc == host:
                u = "https://" + u[7:]                 # same page; the e-governance page came twice
            p = urlsplit(u)
            if u.lower().endswith(".pdf"):
                pdfs.append(u)
            elif (p.netloc == host and p.path.startswith(prefix) and u not in seen
                  and not _SKIP_EXT.search(p.path) and not _SKIP_PATH.search(u)):
                seen.add(u)
                queue.append(u)
    return pages, list(dict.fromkeys(pdfs))


def diff(old: dict, pages: list[dict]) -> list[str]:
    """Lines for changes.md: new and changed pages, with numbers that appeared or vanished."""
    out = []
    for p in pages:
        prev = old.get(p["url"])
        if prev is None:
            out.append(f"- NEW {p['url']} — {p['title']}")
        elif prev["sha256"] != p["sha256"]:
            before, after = set(re.findall(r"\d[\d,.]*", prev.get("text", ""))), set(re.findall(r"\d[\d,.]*", p["text"]))
            nums = ""
            if before != after:
                nums = f" (numbers removed: {', '.join(sorted(before - after)[:8]) or '-'};" \
                       f" added: {', '.join(sorted(after - before)[:8]) or '-'})"
            out.append(f"- CHANGED {p['url']} — {p['title']}{nums}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated slugs from web_sources.csv")
    ap.add_argument("--max", type=int, help="pages per section (overrides max_pages)")
    ap.add_argument("--links", help="print the links on one page and exit")
    args = ap.parse_args()
    if args.links:
        if not allowed(args.links):
            raise SystemExit("robots.txt disallows that page")
        r = requests.get(args.links, headers=UA, timeout=60)
        for u in links(r.text, r.url):
            print(u)
        return
    PAGES.mkdir(exist_ok=True)
    STATE.parent.mkdir(exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    only = set(args.only.split(",")) if args.only else None
    changes, all_pdfs = [], []
    for src in load_web_sources():
        if only and src["slug"] not in only:
            continue
        print(f"{src['slug']}: {src['url']}")
        pages, pdfs = crawl(src, args.max or int(src.get("max_pages") or 30))
        old = state.get(src["slug"], {})
        changes += diff(old, pages)
        state[src["slug"]] = {p["url"]: {"sha256": p["sha256"], "text": p["text"], "title": p["title"],
                                         "fetched": dt.date.today().isoformat()} for p in pages}
        with (PAGES / f"web-{src['slug']}.jsonl").open("w", encoding="utf-8") as f:
            for p in pages:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        all_pdfs += pdfs
        print(f"{src['slug']}: {len(pages)} pages, {len(pdfs)} PDF links")
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1))
    (HERE / "web_pdfs.txt").write_text("\n".join(dict.fromkeys(all_pdfs)) + "\n")
    stamp = dt.date.today().isoformat()
    (HERE / "changes.md").write_text(f"# Website changes ({stamp})\n\n" + ("\n".join(changes) or "No changes.") + "\n")
    print(f"{len(changes)} new or changed pages -> {HERE / 'changes.md'}")


if __name__ == "__main__":
    main()
