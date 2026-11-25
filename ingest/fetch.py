"""Download the sources in ingest/sources.csv into ingest/raw/.

    python -m ingest.fetch            # everything not yet downloaded
    python -m ingest.fetch --force    # re-download (charters are revised every year)

A url ending in .pdf is downloaded as is. Any other url is a panchayat's "Citizen Charter"
page: the first PDF it links to on cdn.web.lsgkerala.gov.in is downloaded; a page with no
PDF is saved as HTML (some panchayats put the charter inline) and read as prose.
A 2-second pause between requests keeps this polite to the LSGD servers, and every URL is
checked against its site's robots.txt first: a disallowed URL is skipped and reported (download
that file by hand in a browser, or ask the site's owner, IKM, for the documents).
Each download is recorded in ingest/raw/manifest.json (url, sha256, date).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import requests

HERE = Path(__file__).parent
RAW = HERE / "raw"
UA = {"User-Agent": "kerala-voice-helpdesk/0.2 (citizen-charter indexer; github.com/cksajil/kerala-voice-helpdesk)"}
_PDF_HREF = re.compile(r"""href=["']([^"']+?\.pdf)["']""", re.I)


_ROBOTS: dict[str, RobotFileParser | None] = {}


def allowed(url: str) -> bool:
    """robots.txt says this user agent may fetch url. RFC 9309: a robots.txt that does not exist
    (4xx) allows everything; one that cannot be read (5xx, timeout, network error) disallows
    everything, so a site that is down or blocking us is not crawled by accident."""
    parts = urlsplit(url)
    root = f"{parts.scheme}://{parts.netloc}"
    if root not in _ROBOTS:
        rp = RobotFileParser()
        try:
            r = requests.get(root + "/robots.txt", headers=UA, timeout=20)
            if 200 <= r.status_code < 300:
                rp.parse(r.text.splitlines())
            elif 400 <= r.status_code < 500:
                rp.parse([])
            else:
                rp.disallow_all = True
        except requests.RequestException:
            rp.disallow_all = True
        _ROBOTS[root] = rp
    return _ROBOTS[root].can_fetch(UA["User-Agent"], url)


def load_sources(path: Path = HERE / "sources.csv") -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("slug")]


def pdf_links(page_html: str, base: str) -> list[str]:
    """PDF links on a charter page, charter-looking names first."""
    links = [urljoin(base, html.unescape(h)) for h in _PDF_HREF.findall(page_html)]
    links = list(dict.fromkeys(links))
    hint = re.compile(r"charter|pourav|paurav|%E0%B4%AA%E0%B5%8C|%E0%B4%AA%E0%B5%97|rekha|rakha", re.I)
    return sorted(links, key=lambda u: (not hint.search(u), links.index(u)))


def fetch_one(src: dict, manifest: dict, force: bool = False) -> str:
    slug, url = src["slug"], (src.get("url") or "").strip()
    if not url:
        return "no url (add the PDF by hand to ingest/raw/)"
    if not force and any(RAW.glob(f"{slug}.*")):
        return "already downloaded"
    if not allowed(url):
        return "SKIPPED: robots.txt disallows this url (download it by hand into ingest/raw/)"
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    final, body, ext = url, r.content, "pdf"
    if not url.lower().endswith(".pdf") and b"%PDF" not in r.content[:8]:
        links = pdf_links(r.text, r.url)
        links = [u for u in links if allowed(u)]
        if links:
            time.sleep(2)
            final = links[0]
            p = requests.get(final, headers=UA, timeout=120)
            p.raise_for_status()
            body = p.content
        else:
            ext = "html"
    sha = hashlib.sha256(body).hexdigest()
    prev = manifest.get(slug, {}).get("sha256")
    (RAW / f"{slug}.{ext}").write_bytes(body)
    manifest[slug] = {"url": final, "page": url, "sha256": sha,
                      "bytes": len(body), "fetched": dt.date.today().isoformat()}
    if prev and prev != sha:
        # A revised charter: drop its old OCR so ingest.extract redoes it, and say so.
        (HERE / "pages" / f"{slug}.jsonl").unlink(missing_ok=True)
        return f"CHANGED {slug}.{ext} ({len(body) // 1024} KB): will be re-extracted"
    if prev == sha:
        return "unchanged"
    return f"saved {slug}.{ext} ({len(body) // 1024} KB)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    RAW.mkdir(exist_ok=True)
    mpath = RAW / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    changed = []
    for src in load_sources():
        try:
            msg = fetch_one(src, manifest, args.force)
        except requests.RequestException as e:
            msg = f"FAILED: {e}"
        print(f"{src['slug']:24} {msg}", flush=True)
        if msg.startswith("CHANGED"):
            alias = src.get("kb_alias")
            changed.append(f"- {src['slug']}: {manifest[src['slug']]['url']}"
                           + (f" — re-check Tier-1 entries citing '{alias}' (python -m ingest.check_kb)" if alias else ""))
        mpath.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
        if not msg.startswith(("already", "no url", "SKIPPED")):
            time.sleep(2)
    (HERE / "charter_changes.md").write_text(
        f"# Charter changes ({dt.date.today().isoformat()})\n\n" + ("\n".join(changed) or "No changes.") + "\n")
    if changed:
        print(f"{len(changed)} charters changed -> {HERE / 'charter_changes.md'}")


if __name__ == "__main__":
    main()
