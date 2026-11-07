"""'Kerala today' tools: weather (Open-Meteo) and news headlines (RSS).

Numbers are put into the reply by templates, not by the LLM, so a 2B model
cannot invent a temperature. The LLM only rephrases if you want it to.
Open-Meteo's free API is for non-commercial use; a paid client deployment
needs their commercial plan.
"""
from __future__ import annotations

import html
import re
import time

import feedparser
import requests

from .normalise import normalise

UA = {"User-Agent": "kerala-helpdesk/0.1 (portfolio demo)"}
TIMEOUT = 6

# district -> (Malayalam name, match stems incl. inflected forms, lat, lon)
DISTRICTS = {
    "thiruvananthapuram": ("തിരുവനന്തപുരം", ["തിരുവനന്തപുര", "trivandrum", "thiruvananthapuram"], 8.5241, 76.9366),
    "kollam": ("കൊല്ലം", ["കൊല്ല", "kollam"], 8.8932, 76.6141),
    "pathanamthitta": ("പത്തനംതിട്ട", ["പത്തനംതിട്ട", "pathanamthitta"], 9.2648, 76.7870),
    "alappuzha": ("ആലപ്പുഴ", ["ആലപ്പുഴ", "alappuzha", "alleppey"], 9.4981, 76.3388),
    "kottayam": ("കോട്ടയം", ["കോട്ടയ", "kottayam"], 9.5916, 76.5222),
    "idukki": ("ഇടുക്കി", ["ഇടുക്കി", "idukki"], 9.8500, 76.9400),
    "ernakulam": ("എറണാകുളം", ["എറണാകുള", "കൊച്ചി", "ernakulam", "kochi", "cochin"], 9.9312, 76.2673),
    "thrissur": ("തൃശ്ശൂർ", ["തൃശ്ശൂ", "തൃശൂ", "thrissur", "trichur"], 10.5276, 76.2144),
    "palakkad": ("പാലക്കാട്", ["പാലക്കാ", "palakkad", "palghat"], 10.7867, 76.6548),
    "malappuram": ("മലപ്പുറം", ["മലപ്പുറ", "malappuram"], 11.0510, 76.0711),
    "kozhikode": ("കോഴിക്കോട്", ["കോഴിക്കോ", "kozhikode", "calicut"], 11.2588, 75.7804),
    "wayanad": ("വയനാട്", ["വയനാ", "കൽപ്പറ്റ", "wayanad", "kalpetta"], 11.6085, 76.0830),
    "kannur": ("കണ്ണൂർ", ["കണ്ണൂ", "kannur", "cannanore"], 11.8745, 75.3704),
    "kasaragod": ("കാസർഗോഡ്", ["കാസർഗോ", "കാസർകോ", "kasaragod", "kasargod"], 12.4996, 74.9869),
}
DEFAULT_DISTRICT = "thiruvananthapuram"

# WMO weather codes -> short Malayalam description
WMO = {0: "തെളിഞ്ഞ ആകാശം", 1: "ഏറെക്കുറെ തെളിഞ്ഞ ആകാശം", 2: "ഭാഗികമായി മേഘാവൃതം", 3: "മേഘാവൃതം",
       45: "മൂടൽമഞ്ഞ്", 48: "മൂടൽമഞ്ഞ്", 51: "ചാറ്റൽമഴ", 53: "ചാറ്റൽമഴ", 55: "ശക്തമായ ചാറ്റൽമഴ",
       61: "നേരിയ മഴ", 63: "മഴ", 65: "ശക്തമായ മഴ", 80: "ഇടവിട്ട മഴ", 81: "ഇടവിട്ട മഴ",
       82: "ശക്തമായ ഇടവിട്ട മഴ", 95: "ഇടിമിന്നലോടു കൂടിയ മഴ", 96: "ഇടിമിന്നലോടു കൂടിയ മഴ",
       99: "ഇടിമിന്നലോടു കൂടിയ ശക്തമായ മഴ"}


def find_district(query: str) -> str | None:
    q = normalise(query)
    for key, (_, stems, _, _) in DISTRICTS.items():
        if any(normalise(s) in q for s in stems):
            return key
    return None


def weather(query: str) -> dict:
    key = find_district(query) or DEFAULT_DISTRICT
    name_ml, _, lat, lon = DISTRICTS[key]
    r = requests.get("https://api.open-meteo.com/v1/forecast", headers=UA, timeout=TIMEOUT, params={
        "latitude": lat, "longitude": lon, "timezone": "Asia/Kolkata", "forecast_days": 2,
        "current": "temperature_2m,relative_humidity_2m,weather_code",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum",
    })
    r.raise_for_status()
    d = r.json()
    cur, day = d["current"], d["daily"]
    text = (f"{name_ml}: ഇപ്പോൾ {round(cur['temperature_2m'])} ഡിഗ്രി സെൽഷ്യസ്, "
            f"{WMO.get(cur['weather_code'], 'കാലാവസ്ഥ ലഭ്യമല്ല')}. "
            f"ഇന്ന് കൂടിയ താപനില {round(day['temperature_2m_max'][0])}, കുറഞ്ഞത് {round(day['temperature_2m_min'][0])} ഡിഗ്രി. "
            f"മഴ സാധ്യത {day['precipitation_probability_max'][0]} ശതമാനം.")
    return {"district": key, "district_found": key == find_district(query), "text": text,
            "source": "Open-Meteo (open-meteo.com)", "raw": d}


NEWS_FEEDS = [
    ("Oneindia Malayalam", "https://malayalam.oneindia.com/rss/malayalam-news-fb.xml"),
    ("Google News (Malayalam)", "https://news.google.com/rss?hl=ml&gl=IN&ceid=IN:ml"),
]
_CACHE: dict[str, tuple[float, list[dict]]] = {}
CACHE_SECONDS = 600
_TAG = re.compile(r"<[^>]+>")


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub(" ", s or ""))).strip()


def news(query: str = "", n: int = 5) -> dict:
    """Headlines only (no article scraping): fresh, fast, and no copyright issue."""
    for name, url in NEWS_FEEDS:
        hit = _CACHE.get(url)
        if hit and time.time() - hit[0] < CACHE_SECONDS:
            items = hit[1]
        else:
            try:
                r = requests.get(url, headers=UA, timeout=TIMEOUT)
                r.raise_for_status()
                feed = feedparser.parse(r.content)
                items = [{"title": _clean(e.get("title")), "summary": _clean(e.get("summary"))[:200],
                          "link": e.get("link"), "published": e.get("published", "")}
                         for e in feed.entries]
                items = [i for i in items if i["title"]]
            except requests.RequestException:
                continue
            _CACHE[url] = (time.time(), items)
        if items:
            # Optional district filter: prefer headlines that mention the district.
            key = find_district(query)
            if key:
                stems = [normalise(s) for s in DISTRICTS[key][1]]
                local = [i for i in items if any(s in normalise(i["title"]) for s in stems)]
                items = local or items
            top = items[:n]
            text = "ഇന്നത്തെ പ്രധാന വാർത്തകൾ: " + " | ".join(i["title"] for i in top)
            return {"source": name, "items": top, "text": text}
    return {"source": None, "items": [], "text": ""}
