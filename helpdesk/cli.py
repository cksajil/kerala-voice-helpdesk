"""Ask the helpdesk from the terminal:  python -m helpdesk.cli "ട്രേഡ് ലൈസൻസ് പുതുക്കാൻ എന്ത് ചെയ്യണം?"
Without an argument it starts an interactive prompt."""
import json
import sys
import time

from .graph import build_graph


def main():
    g = build_graph()
    qs = sys.argv[1:] or iter(lambda: input("ചോദ്യം> "), "")
    for q in qs:
        t = time.perf_counter()
        s = g.invoke({"query": q})
        print(f"\n[{s['route']} via {s.get('route_by')}, {time.perf_counter() - t:.1f}s]")
        print(s["answer"])
        for src in s.get("sources", []):
            print("  -", json.dumps(src, ensure_ascii=False))


if __name__ == "__main__":
    main()
