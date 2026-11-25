"""Smoke test for a running help desk API (Docker or uvicorn). Python standard library only.

    python scripts/smoke_test.py                      # http://127.0.0.1:8000
    python scripts/smoke_test.py --url http://host:8000 --wait 1800

Waits for /health to report "ok" (the first Docker start downloads the 2.8 GB model and warms
the prompt cache), then asks five questions and checks the shape of each reply: a service answer
with a source, a fee worked out in code, a follow-up carried by the returned context, a decline
for an off-topic question, and a rejected empty question. No weather or news (they call outside
services). Exit code 0 when every check passes. Prints each reply's per-stage timings.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def call(url: str, body: dict | None = None, timeout: float = 600) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--wait", type=float, default=1800, help="seconds to wait for status ok")
    args = ap.parse_args()
    t0, last = time.time(), None
    while True:
        try:
            _, h = call(args.url + "/health", timeout=5)
            if h["status"] != last:
                print(f"[{time.time() - t0:6.0f}s] health: {h['status']} warm-up {h.get('warmup')}", flush=True)
                last = h["status"]
            if h["status"] == "ok":
                break
            if h["status"] == "error":
                print("FAIL start-up:", h.get("error"))
                return 1
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        if time.time() - t0 > args.wait:
            print("FAIL: not ready after", args.wait, "s")
            return 1
        time.sleep(5)
    print(json.dumps(h, ensure_ascii=False))

    failures = 0

    def check(name: str, ok: bool, reply: dict):
        nonlocal failures
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}: route={reply.get('route')} by={reply.get('route_by')} "
              f"refused={reply.get('refused')} timings={reply.get('timings')}")
        print("     ", (reply.get("answer") or json.dumps(reply, ensure_ascii=False))[:300])

    _, r = call(args.url + "/ask", {"question": "ജനന സർട്ടിഫിക്കറ്റ് എങ്ങനെ കിട്ടും?"})
    check("service answer", r.get("route") == "service_faq" and not r.get("refused") and bool(r.get("sources"))
          and bool(r.get("answer")), r)
    _, f = call(args.url + "/ask", {"question": "ജനനം രജിസ്റ്റർ ചെയ്യാൻ 40 ദിവസം വൈകിയാൽ എത്ര ഫീസ്?"})
    check("fee in code", f.get("tool") == "fee_calc" and "രൂപ" in f.get("answer", ""), f)
    _, a = call(args.url + "/ask", {"question": "വിവാഹ രജിസ്ട്രേഷന് എന്തൊക്കെ രേഖകൾ വേണം?"})
    _, b = call(args.url + "/ask", {"question": "എത്ര ദിവസം എടുക്കും?", "context": a.get("context")})
    check("follow-up via context", b.get("route") == "service_faq" and b.get("route_by") == "follow-up", b)
    _, o = call(args.url + "/ask", {"question": "ഒരു തമാശ പറയൂ"})
    check("off-topic declined", o.get("route") == "other", o)
    code, e = call(args.url + "/ask", {"question": ""})
    check("empty question rejected", code == 422, {"route": code})
    print("all checks passed" if not failures else f"{failures} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
