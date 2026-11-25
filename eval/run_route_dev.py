"""Routing check on eval/route_dev.jsonl (a tuned-on dev set: written to fix routing, not held out).

    python -m eval.run_route_dev          # offline: rule_route + place lookup, no LLM, no network

Per item: the router's rule route (None = no rule fired: the graph then asks "which service?" or
uses the open title match), and for weather the district the weather tool would use.
"""
import json
from pathlib import Path

from helpdesk import tools
from helpdesk.graph import rule_route


def main():
    items = [json.loads(l) for l in (Path(__file__).parent / "route_dev.jsonl").open(encoding="utf-8")]
    ok_route = ok_dist = n_dist = 0
    for it in items:
        r = rule_route(it["question"]) or "other"
        good = r == it["route"]
        ok_route += good
        line = f"{'ok ' if good else 'BAD'} {it['id']} {it['route']:11} -> {r:11}"
        if it.get("district"):
            place = tools.find_place(it["question"])
            d = place[1] if place else (tools.find_district(it["question"]) or tools.DEFAULT_DISTRICT)
            n_dist += 1
            ok_dist += d == it["district"]
            line += f" district {d}{'' if d == it['district'] else ' (want ' + it['district'] + ')'}"
        print(line, it["question"])
    print(f"route {ok_route}/{len(items)}   weather district {ok_dist}/{n_dist}")


if __name__ == "__main__":
    main()
