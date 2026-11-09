"""OCR every page of every downloaded PDF:  ingest/raw/<slug>.pdf -> ingest/pages/<slug>.jsonl

    python -m ingest.extract                 # all PDFs not yet extracted
    python -m ingest.extract --only kadanad  # one source (re-extracts)

About 3-5 s a page on a laptop; a 25-page charter takes 1-2 minutes. Page images are
kept in ingest/work/<slug>/ so a re-run only re-does the OCR.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import time
from pathlib import Path

import pdfplumber

from .ocr import page_content

HERE = Path(__file__).parent
RAW, PAGES, WORK = HERE / "raw", HERE / "pages", HERE / "work"


def extract(pdf: Path, out: Path) -> int:
    work = WORK / pdf.stem
    work.mkdir(parents=True, exist_ok=True)
    with pdfplumber.open(pdf) as doc:
        n = len(doc.pages)
    state: dict = {}
    t0 = time.perf_counter()
    with out.open("w", encoding="utf-8") as f:
        for p in range(1, n + 1):
            c = page_content(pdf, p, work, state)
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
            kind = f"{len(c['rows'])} rows" if c["kind"] == "table" else f"{len(c['text'])} chars"
            print(f"  {pdf.stem} p{p}/{n} {c['kind']:5} {kind:>11} conf {c['conf']}", flush=True)
    print(f"{pdf.stem}: {n} pages in {time.perf_counter() - t0:.0f}s")
    return n


def extract_html(src: Path, out: Path) -> int:
    """Charter published inline on the panchayat page: one prose 'page' of its text."""
    t = src.read_text(encoding="utf-8", errors="ignore")
    t = re.sub(r"(?is)<(script|style|nav|header|footer)[^>]*>.*?</\1>", " ", t)
    t = re.sub(r"(?i)<br\s*/?>|</p>|</li>|</tr>|</h\d>", "\n", t)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    t = "\n".join(re.sub(r"[ \t]+", " ", l).strip() for l in t.splitlines())
    t = re.sub(r"\n{2,}", "\n", t).strip()
    out.write_text(json.dumps({"page": 1, "kind": "prose", "text": t, "conf": 100.0}, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"{src.stem}: html, {len(t)} chars")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated slugs")
    args = ap.parse_args()
    PAGES.mkdir(exist_ok=True)
    only = set(args.only.split(",")) if args.only else None
    for src in sorted([*RAW.glob("*.pdf"), *RAW.glob("*.html")]):
        out = PAGES / f"{src.stem}.jsonl"
        if only is not None and src.stem not in only:
            continue
        if only is None and out.exists():
            continue
        (extract_html if src.suffix == ".html" else extract)(src, out)


if __name__ == "__main__":
    main()
