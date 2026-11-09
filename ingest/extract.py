"""OCR every page of every downloaded PDF:  ingest/raw/<slug>.pdf -> ingest/pages/<slug>.jsonl

    python -m ingest.extract                 # all PDFs not yet extracted
    python -m ingest.extract --only kadanad  # one source (re-extracts)
    python -m ingest.extract --jobs 4        # 4 PDFs at a time (Kaggle has 4 cores, Colab 2)
    python -m ingest.extract --budget 600    # stop starting new PDFs after 600 minutes

A PDF's pages are written to <slug>.jsonl.partial and renamed only when the PDF is finished,
so a run killed by a timeout never leaves a half-read charter that the next run would skip.

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
    tmp = out.with_suffix(".jsonl.partial")
    with tmp.open("w", encoding="utf-8") as f:
        for p in range(1, n + 1):
            c = page_content(pdf, p, work, state)
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
            kind = f"{len(c['rows'])} rows" if c["kind"] == "table" else f"{len(c['text'])} chars"
            print(f"  {pdf.stem} p{p}/{n} {c['kind']:5} {kind:>11} conf {c['conf']}", flush=True)
    tmp.replace(out)
    print(f"{pdf.stem}: {n} pages in {time.perf_counter() - t0:.0f}s", flush=True)
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


def _one(job: tuple[str, str]) -> str:
    src, out = Path(job[0]), Path(job[1])
    try:
        (extract_html if src.suffix == ".html" else extract)(src, out)
        return f"done {src.stem}"
    except Exception as e:                                  # one bad PDF must not stop the rest
        return f"FAILED {src.stem}: {e}"


def main():
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated slugs")
    ap.add_argument("--jobs", type=int, default=1, help="PDFs in parallel")
    ap.add_argument("--budget", type=float, help="minutes after which no new PDF is started")
    args = ap.parse_args()
    PAGES.mkdir(exist_ok=True)
    only = set(args.only.split(",")) if args.only else None
    jobs = []
    for src in sorted([*RAW.glob("*.pdf"), *RAW.glob("*.html")]):
        out = PAGES / f"{src.stem}.jsonl"
        if only is not None and src.stem not in only:
            continue
        if only is None and out.exists():
            continue
        jobs.append((str(src), str(out)))
    print(f"{len(jobs)} documents to extract, {args.jobs} at a time", flush=True)
    if args.jobs > 1:
        os.environ.setdefault("OMP_THREAD_LIMIT", "1")       # one core per tesseract process
    t0, deadline = time.time(), (time.time() + args.budget * 60) if args.budget else None
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        pending, futures = list(jobs), set()
        while pending or futures:
            while pending and len(futures) < args.jobs and (deadline is None or time.time() < deadline):
                futures.add(ex.submit(_one, pending.pop(0)))
            if not futures:
                break
            done = next(as_completed(futures))
            futures.remove(done)
            print(done.result(), flush=True)
    left = len(pending)
    print(f"finished in {(time.time() - t0) / 60:.0f} min" + (f"; {left} documents left for the next run" if left else ""))


if __name__ == "__main__":
    main()
