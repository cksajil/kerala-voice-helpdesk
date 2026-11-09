"""PDF page -> table rows or prose, by OCR.

Why OCR and not the PDF's text layer: charters use legacy ASCII fonts (ML-TT Revathi)
or Unicode fonts with broken glyph maps. The text layer of the Kadanad charter turned
"Ksmart അപേക്ഷ" into "ഗമൊൃ അേപേക്ഷ", which we then mis-read as "plain-paper application"
in four KB entries. Tesseract reads the rendered page, so the font does not matter.

Tables: pdfplumber finds the ruled cells (vector lines, font-independent); the page is
OCRed once in sparse-text mode (psm 11: psm 6 dropped whole table rows) and each word box
goes to the cell that contains its centre. Prose pages use psm 4 for reading order.

Needs: poppler (pdftoppm), tesseract with Malayalam data.
  Mac:   brew install poppler tesseract tesseract-lang
  Linux: apt install poppler-utils tesseract-ocr tesseract-ocr-mal
"""
from __future__ import annotations

import csv
import io
import re
import subprocess
from pathlib import Path

import pdfplumber

DPI = 300
LANG = "mal+eng"
MIN_CONF = 40          # drop OCR words below this confidence (stray marks, table rules)

# Column roles from header words. Order matters: the first matching role wins.
ROLES = [
    ("no", ["ക്രമ", "നമ്പർ", "നം."]),
    ("fee", ["ഫീസ", "ഫീ", "തുക", "നിരക്ക"]),
    ("time", ["ദിവസ", "സമയ", "കാലാവധി", "സമയപരിധി"]),
    ("how", ["അപേക്ഷിക്കുന്ന", "അപേക്ഷ സമർപ്പി", "എങ്ങനെ"]),
    ("officer", ["ഉദ്യോഗസ്ഥ", "ഓഫീസർ", "ചുമതല"]),
    ("conditions", ["നിബന്ധന", "രേഖ", "ഹാജരാക്ക", "ആവശ്യമായ", "അനുബന്ധ"]),
    ("service", ["സേവന", "വിവരങ്ങൾ", "ഇനം"]),
]
_JUNK = re.compile(r"^[^\wഀ-ൿ₹]+$")        # tokens with no letter/digit: | ] ~ $ ¥


def _norm(s: str) -> str:
    from helpdesk.normalise import normalise
    return normalise(s)


def render(pdf: Path, page: int, out_dir: Path) -> Path:
    stem = out_dir / f"p{page:04d}"
    png = stem.with_suffix(".png")
    if not png.exists():
        subprocess.run(["pdftoppm", "-r", str(DPI), "-png", "-singlefile", "-f", str(page), "-l", str(page),
                        str(pdf), str(stem)], check=True)
    return png


def ocr_words(png: Path, psm: int = 6) -> list[dict]:
    tsv = subprocess.run(["tesseract", str(png), "-", "-l", LANG, "--psm", str(psm), "tsv"],
                         capture_output=True, text=True, check=True).stdout
    words = []
    for r in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        t = (r.get("text") or "").strip()
        if r.get("level") != "5" or not t or _JUNK.match(t) or float(r.get("conf") or -1) < MIN_CONF:
            continue
        words.append({"t": t, "conf": float(r["conf"]), "h": int(r["height"]),
                      "x": int(r["left"]) + int(r["width"]) / 2, "y": int(r["top"]) + int(r["height"]) / 2,
                      "key": (int(r["block_num"]), int(r["par_num"]), int(r["line_num"]), int(r["word_num"]))})
    return words


def reading_order(words: list[dict]) -> str:
    """Words of one cell, top to bottom then left to right (sparse mode has no line ids)."""
    lines: list[list[dict]] = []
    for w in sorted(words, key=lambda w: w["y"]):
        if lines and abs(w["y"] - lines[-1][0]["y"]) <= 0.6 * max(w["h"], lines[-1][0]["h"], 1):
            lines[-1].append(w)
        else:
            lines.append([w])
    return " ".join(w["t"] for line in lines for w in sorted(line, key=lambda w: w["x"]))


def _roles(header: list[str]) -> list[str | None]:
    out = []
    for cell in header:
        c = _norm(cell)
        out.append(next((role for role, stems in ROLES if any(_norm(s) in c for s in stems)), None))
    return out


def _is_header(cells: list[str]) -> bool:
    """Header cells are short labels. Data rows also contain 'ഫീസ്', 'ദിവസം', 'അപേക്ഷ'
    (a long row on Kadanad p.3 replaced the real header and lost pages 4-10)."""
    filled = [c for c in cells if c.strip()]
    if not filled or max(len(c) for c in filled) > 45:
        return False
    roles = set(r for r in _roles(cells) if r)
    return len(roles) >= 3 and "service" in roles


def page_content(pdf: Path, page: int, work: Path, state: dict) -> dict:
    """{'page', 'kind': 'table'|'prose', 'rows'|'text', 'conf'}.
    state carries the header roles and the current section title across pages."""
    png = render(pdf, page, work)
    with pdfplumber.open(pdf) as doc:
        tables = [t for t in doc.pages[page - 1].find_tables() if len(t.rows) >= 2]
    words = ocr_words(png, psm=11 if tables else 4)
    conf = round(sum(w["conf"] for w in words) / len(words), 1) if words else 0.0
    s = DPI / 72
    rows, inside = [], set()
    for tb in tables:
        tx0, ty0, tx1, ty1 = (v * s for v in tb.bbox)
        inside |= {i for i, w in enumerate(words) if tx0 <= w["x"] <= tx1 and ty0 <= w["y"] <= ty1}
        # A heading printed just above the table ("പ്രാഥമികാരോഗ്യ കേന്ദ്രം") is the section.
        above = [w for w in words if ty0 - 0.6 * DPI <= w["y"] < ty0 and tx0 <= w["x"] <= tx1]
        if above:
            last_y = max(w["y"] for w in above)
            head = reading_order([w for w in above if abs(w["y"] - last_y) <= 0.6 * w["h"]])
            if 4 <= len(head) <= 70:
                state["section"] = head
        for row in tb.rows:
            boxes = [None if c is None else tuple(v * s for v in c) for c in row.cells]
            cells = []
            for b in boxes:
                if b is None:
                    cells.append("")
                    continue
                ws = [w for w in words if b[0] <= w["x"] <= b[2] and b[1] <= w["y"] <= b[3]]
                cells.append(reading_order(ws))
            filled = [c for c in cells if c.strip()]
            if not filled:
                continue
            if len(filled) == 1 and len(cells) >= 3:
                b = next(b for b, c in zip(boxes, cells) if c.strip())
                # A spanning row is a section title only when it is short: long spanning rows
                # are notes ("കുറിപ്പ് (1) 01.04.2011 മുതൽ ...") and went onto every row after them.
                if b is not None and (b[2] - b[0]) >= 0.6 * (tx1 - tx0) and len(filled[0]) <= 70:
                    state["section"] = filled[0]
                    continue
            if _is_header(cells):
                # Remember each header column's x-range: later pages often have a different
                # number of cells per row (merged cells), so map cells by position, not index.
                state["cols"] = [(b[0], b[2], r) for b, r in zip(boxes, _roles(cells)) if b is not None]
                continue
            cols = state.get("cols")
            if not cols:
                continue                                        # no header yet: cannot label columns
            rec: dict[str, str] = {}
            for b, text in zip(boxes, cells):
                if b is None or not text.strip():
                    continue
                role = max(cols, key=lambda c: min(b[2], c[1]) - max(b[0], c[0]))[2]
                if role and role != "no":
                    rec[role] = (rec.get(role, "") + " " + text).strip()
            if rec.get("service"):
                rec["section"] = state.get("section", "")
                rows.append(rec)
    if rows:
        rest = reading_order([w for i, w in enumerate(words) if i not in inside])
        return {"page": page, "kind": "table", "rows": rows, "conf": conf,
                "text": rest if len(rest) >= 120 else ""}
    if tables:                                   # tables without a usable header: read as prose
        words = ocr_words(png, psm=4)
    lines, last = [], None
    for w in sorted(words, key=lambda w: w["key"]):
        k = w["key"][:3]
        if k != last:
            lines.append([])
            last = k
        lines[-1].append(w["t"])
    return {"page": page, "kind": "prose", "text": "\n".join(" ".join(l) for l in lines), "conf": conf,
            "table_fallback": bool(tables)}
