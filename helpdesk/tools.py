"""'Kerala today' tools: weather (Open-Meteo) and news headlines (RSS).

Numbers are put into the reply by templates, not by the LLM, so a 2B model
cannot invent a temperature. The LLM only rephrases if you want it to.
Open-Meteo's free API is for non-commercial use; a paid client deployment
needs their commercial plan.
"""
from __future__ import annotations

import html
import json
import re
import time
from pathlib import Path

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

# Places people ask weather for that are not district names -> (Malayalam name, stems, district, lat, lon).
# Checked before districts: "വാഗമണിലെ കാലാവസ്ഥ" got Thiruvananthapuram's weather (9 Nov log).
PLACES = {
    "vagamon": ("വാഗമൺ", ["വാഗമ", "vagamon"], "idukki", 9.686, 76.905),
    "munnar": ("മൂന്നാർ", ["മൂന്നാ", "munnar"], "idukki", 10.089, 77.060),
    "thekkady": ("തേക്കടി", ["തേക്കടി", "കുമളി", "thekkady", "kumily"], "idukki", 9.602, 77.161),
    "peermade": ("പീരുമേട്", ["പീരുമേ", "peermade"], "idukki", 9.577, 76.976),
    "ponmudi": ("പൊന്മുടി", ["പൊന്മുടി", "പൊൻമുടി", "ponmudi"], "thiruvananthapuram", 8.760, 77.116),
    "kovalam": ("കോവളം", ["കോവള", "kovalam"], "thiruvananthapuram", 8.400, 76.979),
    "varkala": ("വർക്കല", ["വർക്കല", "varkala"], "thiruvananthapuram", 8.734, 76.716),
    "kumarakom": ("കുമരകം", ["കുമരക", "kumarakom"], "kottayam", 9.617, 76.430),
    "athirappilly": ("അതിരപ്പിള്ളി", ["അതിരപ്പിള്ളി", "athirappilly"], "thrissur", 10.285, 76.570),
    "guruvayur": ("ഗുരുവായൂർ", ["ഗുരുവായൂ", "guruvayur"], "thrissur", 10.594, 76.041),
    "sabarimala": ("ശബരിമല", ["ശബരിമല", "sabarimala"], "pathanamthitta", 9.435, 77.081),
    "nelliyampathy": ("നെല്ലിയാമ്പതി", ["നെല്ലിയാമ്പതി", "nelliyampathy"], "palakkad", 10.535, 76.693),
    "bekal": ("ബേക്കൽ", ["ബേക്കൽ", "bekal"], "kasaragod", 12.392, 75.033),
    "thalassery": ("തലശ്ശേരി", ["തലശ്ശേരി", "തലശേരി", "thalassery"], "kannur", 11.749, 75.493),
    "bathery": ("സുൽത്താൻ ബത്തേരി", ["ബത്തേരി", "bathery"], "wayanad", 11.665, 76.263),
}


def _town_stem(name: str) -> str:
    """Match form of a town name: drop a final chillu, virama or anusvara, which change when the
    word is inflected (പുനലൂർ -> പുനലൂരിൽ, തളിപ്പറമ്പ് -> തളിപ്പറമ്പിൽ, വൈക്കം -> വൈക്കത്ത്)."""
    return re.sub(r"(?:[ൺൻർൽൾൿ]|[ംര]?്|ം)$", "", name)


def _load_towns() -> dict:
    """Municipalities and towns from data/places.jsonl (ingest/places.py: geocoded, with source URL)."""
    path = Path(__file__).resolve().parent.parent / "data" / "places.jsonl"
    out = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                stems = d["stems"] or [_town_stem(d["ml"])]
                lat, lon = d["lat"], d["lon"]
                if lat is None:                  # not geocoded: the district centre, said in the reply
                    _, _, lat, lon = DISTRICTS[d["district"]]
                out[d["key"]] = (d["ml"], stems + [d["en"].lower()], d["district"], lat, lon)
    return out


for _k, _v in _load_towns().items():
    PLACES.setdefault(_k, _v)          # the hand-written places above win

# A "<place>ിലെ" word we could not resolve: say so instead of quietly answering for the default.
_UNKNOWN_PLACE = re.compile(r"(?<![ഀ-ൿ])([ഀ-ൿ]{3,})(?:ിലെ|ത്തിലെ|യിലെ)(?![ഀ-ൿ])")
_NOT_PLACES = ("കേരള", "ജില്ല", "നാട്", "സ്ഥല", "പ്രദേശ", "ഇവിടെ", "അവിടെ", "നഗര")


def find_place(query: str):
    q = normalise(query)
    for key, (name, stems, district, lat, lon) in PLACES.items():
        if any(normalise(s) in q for s in stems):
            return name, district, lat, lon
    return None

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


_TOMORROW = re.compile(r"നാളെ|നാളത്തെ|tomorrow")


def weather(query: str) -> dict:
    place = find_place(query)
    if place:
        name, key, lat, lon = place
        name_ml = f"{name} ({DISTRICTS[key][0]} ജില്ല)"
        if (lat, lon) == DISTRICTS[key][2:]:          # town without coordinates: say whose weather it is
            name_ml = f"{DISTRICTS[key][0]} ജില്ലാ കേന്ദ്രം ({name} അല്ല)"
    else:
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
    if _TOMORROW.search(normalise(query)):
        # "നാളെ മഴ പെയ്യുമോ?" was answered with today's forecast (12 Nov log): read the second day.
        text = (f"{name_ml}: നാളെ കൂടിയ താപനില {round(day['temperature_2m_max'][1])}, "
                f"കുറഞ്ഞത് {round(day['temperature_2m_min'][1])} ഡിഗ്രി. "
                f"മഴ സാധ്യത {day['precipitation_probability_max'][1]} ശതമാനം.")
    else:
        text = (f"{name_ml}: ഇപ്പോൾ {round(cur['temperature_2m'])} ഡിഗ്രി സെൽഷ്യസ്, "
                f"{WMO.get(cur['weather_code'], 'കാലാവസ്ഥ ലഭ്യമല്ല')}. "
                f"ഇന്ന് കൂടിയ താപനില {round(day['temperature_2m_max'][0])}, കുറഞ്ഞത് {round(day['temperature_2m_min'][0])} ഡിഗ്രി. "
                f"മഴ സാധ്യത {day['precipitation_probability_max'][0]} ശതമാനം.")
    found = bool(place) or find_district(query) is not None
    if not found:
        m = _UNKNOWN_PLACE.search(normalise(query))
        if m and not any(m.group(1).startswith(normalise(x)) for x in _NOT_PLACES):
            text = f"ആ സ്ഥലം എന്റെ പട്ടികയിൽ ഇല്ല. ജില്ലയുടെ പേര് പറഞ്ഞാൽ കൃത്യമായി പറയാം. {text}"
    return {"district": key, "district_found": found, "text": text,
            "source": "Open-Meteo (open-meteo.com)", "raw": d}


NEWS_FEEDS = [
    ("Oneindia Malayalam", "https://malayalam.oneindia.com/rss/malayalam-news-fb.xml"),
    ("Google News (Kerala)", "https://news.google.com/rss/search?q=%E0%B4%95%E0%B5%87%E0%B4%B0%E0%B4%B3%E0%B4%82&hl=ml&gl=IN&ceid=IN:ml"),
    ("Google News (Malayalam)", "https://news.google.com/rss?hl=ml&gl=IN&ceid=IN:ml"),
]


def _district_feed(key: str) -> tuple[str, str]:
    """Google News search for the district's Malayalam name: the general feeds rarely
    carry local headlines ('തിരുവനന്തപുരം ജില്ലയിലെ പ്രത്യേക വാർത്തകൾ കിട്ടിയില്ല')."""
    from urllib.parse import quote
    name = DISTRICTS[key][0]
    return (f"Google News ({name})",
            f"https://news.google.com/rss/search?q={quote(name)}&hl=ml&gl=IN&ceid=IN:ml")
_CACHE: dict[str, tuple[float, list[dict]]] = {}
CACHE_SECONDS = 600
_TAG = re.compile(r"<[^>]+>")


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub(" ", s or ""))).strip()


def _headline(title: str) -> str:
    """Google News titles end with ' - Publisher'; drop it for speech (kept in items for the UI)."""
    head, sep, tail = title.rpartition(" - ")
    title = head if sep and len(tail) <= 40 else title
    # Video titles: "പെട്രോളില്‍ എഥനോള്‍ എത്രയുണ്ട്? | ഹലോ കേരളം | Latest News | 08 - 11 - 2026"
    return title.split(" | ")[0].strip()


def news(query: str = "", n: int = 3) -> dict:
    """Headlines only (no article scraping): fresh, fast, and no copyright issue.
    Three short headlines: five took ~28 s to speak in the voice demo."""
    key = find_district(query)
    feeds = ([_district_feed(key)] if key else []) + NEWS_FEEDS
    for name, url in feeds:
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
            # District filter: prefer headlines that mention the district; say so when none do.
            intro = "ഇന്നത്തെ പ്രധാന വാർത്തകൾ: "
            if key and url == feeds[0][1]:             # district search feed: all items are local
                intro = f"{DISTRICTS[key][0]} വാർത്തകൾ: "
            elif key:
                stems = [normalise(s) for s in DISTRICTS[key][1]]
                local = [i for i in items if any(s in normalise(i["title"]) for s in stems)]
                if local:
                    items, intro = local, f"{DISTRICTS[key][0]} വാർത്തകൾ: "
                else:
                    intro = f"{DISTRICTS[key][0]} ജില്ലയിലെ പ്രത്യേക വാർത്തകൾ കിട്ടിയില്ല. കേരളത്തിലെ പ്രധാന വാർത്തകൾ: "
            top = items[:n]
            text = intro + " ".join(_headline(i["title"]).rstrip(".;:,| ") + "." for i in top)
            return {"source": name, "items": top, "text": text}
    return {"source": None, "items": [], "text": ""}
