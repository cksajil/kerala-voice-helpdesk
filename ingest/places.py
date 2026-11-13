"""Kerala towns for the weather tool: data/places_src.csv (name, Malayalam name, district) ->
data/places.jsonl with coordinates from the Open-Meteo geocoding API (a JSON API, not a crawl;
one request a second). Each row keeps the request URL and the fetch date.

    python -m ingest.places            # needs geocoding-api.open-meteo.com (run on Kaggle)

A result is used only if it is in Kerala and within 60 km of its district's centre (tools.DISTRICTS);
otherwise the town is left out and listed, never guessed.
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
        url = f"{API}?{urlencode({'name': r['en'], 'count': 10, 'language': 'en', 'countryCode': 'IN'})}"
        res = requests.get(url, headers=UA, timeout=20).json().get("results") or []
        _, _, dlat, dlon = DISTRICTS[r["district"]]
        cands = sorted((km(dlat, dlon, x["latitude"], x["longitude"]), x) for x in res if x.get("admin1") == "Kerala")
        if not cands or cands[0][0] > MAX_KM:
            skipped.append((r["en"], [(round(k), x.get("admin2")) for k, x in cands[:3]]))
        else:
            k, x = cands[0]
            rows.append({"key": r["en"].lower().replace(" ", "_"), "en": r["en"], "ml": r["ml"], "district": r["district"],
                         "stems": [s for s in r["stems"].split("|") if s], "lat": round(x["latitude"], 4),
                         "lon": round(x["longitude"], 4), "km_from_district_centre": round(k, 1),
                         "source": url, "fetched": today})
        time.sleep(1)
    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} places written, {len(skipped)} skipped")
    for s in skipped:
        print("  skipped:", s)


if __name__ == "__main__":
    main()
