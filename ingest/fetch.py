"""Download the sources in ingest/sources.csv into ingest/raw/.

    python -m ingest.fetch            # everything not yet downloaded
    python -m ingest.fetch --force    # re-download (charters are revised every year)

A url ending in .pdf is downloaded as is. Any other url is a panchayat's "Citizen Charter"
page: the first PDF it links to on cdn.web.lsgkerala.gov.in is downloaded; a page with no
PDF is saved as HTML (some panchayats put the charter inline) and read as prose.
A 2-second pause between requests keeps this polite to the LSGD servers.
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
from urllib.parse import urljoin

import requests

HERE = Path(__file__).parent
RAW = HERE / "raw"
UA = {"User-Agent": "kerala-voice-helpdesk/0.2 (citizen-charter indexer; github.com/cksajil/kerala-voice-helpdesk)"}
_PDF_HREF = re.compile(r"""href=["']([^"']+?\.pdf)["']""", re.I)


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
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    final, body, ext = url, r.content, "pdf"
    if not url.lower().endswith(".pdf") and b"%PDF" not in r.content[:8]:
        links = pdf_links(r.text, r.url)
        if links:
            time.sleep(2)
            final = links[0]
            p = requests.get(final, headers=UA, timeout=120)
            p.raise_for_status()
            body = p.content
        else:
            ext = "html"
    (RAW / f"{slug}.{ext}").write_bytes(body)
    manifest[slug] = {"url": final, "page": url, "sha256": hashlib.sha256(body).hexdigest(),
                      "bytes": len(body), "fetched": dt.date.today().isoformat()}
    return f"saved {slug}.{ext} ({len(body) // 1024} KB)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    RAW.mkdir(exist_ok=True)
    mpath = RAW / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    for src in load_sources():
        try:
            msg = fetch_one(src, manifest, args.force)
        except requests.RequestException as e:
            msg = f"FAILED: {e}"
        print(f"{src['slug']:24} {msg}", flush=True)
        mpath.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
        if not msg.startswith(("already", "no url")):
            time.sleep(2)


if __name__ == "__main__":
    main()
