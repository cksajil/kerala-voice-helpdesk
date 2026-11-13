"""Kerala towns for the weather tool: data/places_src.csv (name, Malayalam name, district) ->
data/places.jsonl with coordinates from the Open-Meteo geocoding API (a JSON API, not a crawl;
one request a second). Each row keeps the request URL and the fetch date.

    python -m ingest.places            # needs geocoding-api.open-meteo.com (run on Kaggle)

The English name is tried first, then the "alt" spellings. A result is used only if it is in Kerala
and within 60 km of its district's centre (tools.DISTRICTS). A town with no such result is kept with
lat/lon null: the weather tool then uses its district's centre and says so; nothing is guessed.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import time
from pathlib import Path
from urllib.parse import urlencode

import requests

from helpdesk.tools import DISTRICTS, UA

ROOT = Path(__file__).resolve().parent.parent
SRC, OUT = ROOT / "data" / "places_src.csv", ROOT / "data" / "places.jsonl"
API = "https://geocoding-api.open-meteo.com/v1/search"
MAX_KM = 60


def km(a, b, c, d):
    p = math.pi / 180
    h = math.sin((c - a) * p / 2) ** 2 + math.cos(a * p) * math.cos(c * p) * math.sin((d - b) * p / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


def main():
    rows, skipped = [], []
    today = dt.date.today().isoformat()
    for r in csv.DictReader(SRC.open(encoding="utf-8")):
        _, _, dlat, dlon = DISTRICTS[r["district"]]
        best, tried = None, []
        for name in [r["en"]] + [a for a in (r.get("alt") or "").split("|") if a]:
            url = f"{API}?{urlencode({'name': name, 'count': 10, 'language': 'en', 'countryCode': 'IN'})}"
            res = requests.get(url, headers=UA, timeout=20).json().get("results") or []
            time.sleep(1)
            cands = sorted((km(dlat, dlon, x["latitude"], x["longitude"]), x) for x in res if x.get("admin1") == "Kerala")
            tried.append((name, [(round(k), x.get("admin2")) for k, x in cands[:3]]))
            if cands and cands[0][0] <= MAX_KM:
                best = (cands[0][0], cands[0][1], url)
                break
        row = {"key": r["en"].lower().replace(" ", "_"), "en": r["en"], "ml": r["ml"], "district": r["district"],
               "stems": [s for s in r["stems"].split("|") if s], "lat": None, "lon": None,
               "km_from_district_centre": None, "source": url, "fetched": today}
        if best:
            k, x, url = best
            row.update(lat=round(x["latitude"], 4), lon=round(x["longitude"], 4), km_from_district_centre=round(k, 1), source=url)
        else:
            skipped.append((r["en"], tried))
        rows.append(row)
    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} places written, {len(rows) - len(skipped)} geocoded, {len(skipped)} on district centre")
    for s in skipped:
        print("  not geocoded:", s)


if __name__ == "__main__":
    main()
