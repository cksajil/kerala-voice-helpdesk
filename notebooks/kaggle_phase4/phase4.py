"""Phase 4 smoke test on Kaggle CPU (private kernel). Kaggle has no Docker daemon, so this runs what the
image runs, without the container: the pinned requirements-api.txt, llama-cpp-python built with the
Dockerfile's portable CMAKE_ARGS, `uvicorn helpdesk.api:app` with the image's environment (2 threads,
in-process Gemma 4 E2B Q4_0, RAM prompt cache, warm-up "kb"), then scripts/smoke_test.py.
Afterwards it asks every curated passage's title twice, once warm (cache filled by the warm-up), to
show what the cache saves. Outputs: /kaggle/working/out.
"""
import glob, json, os, re, shutil, subprocess, sys, tarfile, time, urllib.request
from pathlib import Path

OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
REPO = Path("/kaggle/working/repo")
LOG = open(OUT / "run.log", "a", encoding="utf-8")


def sh(cmd, env=None, timeout=None):
    t = time.time()
    p = subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True,
                       env={**os.environ, **(env or {})}, timeout=timeout)
    out = p.stdout + p.stderr
    LOG.write(f"\n$ {cmd} [{time.time() - t:.0f}s, exit {p.returncode}]\n{out[-30000:]}\n"); LOG.flush()
    print(f"$ {cmd} -> exit {p.returncode} ({time.time() - t:.0f}s)", flush=True)
    return p.returncode, out


shutil.rmtree(REPO, ignore_errors=True)
tgz = glob.glob("/kaggle/input/**/repo.tar.gz", recursive=True)
if tgz:
    REPO.mkdir(); tarfile.open(tgz[0]).extractall(REPO)
else:
    shutil.copytree(Path(sorted(glob.glob("/kaggle/input/**/helpdesk/graph.py", recursive=True), key=len)[0]).parents[1], REPO)
(OUT / "REVISION").write_text((REPO / "REVISION").read_text())

docker = (REPO / "Dockerfile").read_text()
llama_ver = re.search(r"ARG LLAMA_CPP_PYTHON=(\S+)", docker).group(1)
cmake = re.search(r'ARG CMAKE_ARGS="([^"]+)"', docker).group(1)
sh(f"{sys.executable} -m pip install -q -r requirements-api.txt", timeout=3600)
sh(f"CMAKE_ARGS='{cmake}' {sys.executable} -m pip install -q llama-cpp-python=={llama_ver}", timeout=5400)
sh(f"{sys.executable} -m pip freeze > /kaggle/working/out/freeze.txt; lscpu | head -20 > /kaggle/working/out/cpu.txt")

env = {k: v for k, v in re.findall(r"^\s+(\w+)=(\S+?)(?: \\)?$", docker.split("ENV ", 1)[1].split("EXPOSE")[0], re.M)}
env["HF_HOME"] = "/kaggle/temp/hf"          # not under /kaggle/working: kept out of the outputs
print("image env:", env, flush=True)
srv_log = open(OUT / "uvicorn.log", "w")
t0 = time.time()
srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "helpdesk.api:app", "--port", "8000", "--workers", "1"],
                       cwd=REPO, env={**os.environ, **env}, stdout=srv_log, stderr=subprocess.STDOUT)
try:
    code, out = sh(f"{sys.executable} scripts/smoke_test.py --wait 3600", timeout=3 * 3600)
    (OUT / "smoke_test.txt").write_text(out)
    health = json.load(urllib.request.urlopen("http://127.0.0.1:8000/health"))
    health["start_to_ok_s"] = round(time.time() - t0)
    (OUT / "health.json").write_text(json.dumps(health, ensure_ascii=False, indent=1))

    # Warm vs cold: each curated title as a question. Warm = after the warm-up (the cache holds the
    # prompt prefix); cold = a fresh server with HELPDESK_WARMUP=none, same questions, same order.
    sys.path.insert(0, str(REPO))
    kb = [json.loads(l) for l in open(REPO / "data/kb.jsonl", encoding="utf-8") if l.strip()][:20]

    def ask_all(tag):
        rows = []
        for d in kb:
            req = urllib.request.Request("http://127.0.0.1:8000/ask", headers={"Content-Type": "application/json"},
                                         data=json.dumps({"question": d["title"] + " എങ്ങനെ?"}).encode())
            r = json.load(urllib.request.urlopen(req, timeout=900))
            rows.append({"id": d["id"], "route": r["route"], "src": [s.get("id") for s in r["sources"]][:1],
                         **{k: r["timings"].get(k) for k in ("answer", "total")}})
        (OUT / f"latency_{tag}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        return rows
    warm = ask_all("warm")
finally:
    srv.terminate(); srv.wait(60)

srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "helpdesk.api:app", "--port", "8000", "--workers", "1"],
                       cwd=REPO, env={**os.environ, **env, "HELPDESK_WARMUP": "none", "HELPDESK_CACHE_MB": "0"},
                       stdout=open(OUT / "uvicorn_cold.log", "w"), stderr=subprocess.STDOUT)
try:
    sh(f"{sys.executable} scripts/smoke_test.py --wait 1800 > /kaggle/working/out/smoke_test_cold.txt 2>&1", timeout=3600)
    cold = ask_all("cold")
finally:
    srv.terminate(); srv.wait(60)


def p(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return round(xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))], 2) if xs else None


summ = {m: {"n": len(rows), "answer_p50": p([r["answer"] for r in rows], .5), "answer_p90": p([r["answer"] for r in rows], .9),
            "total_p50": p([r["total"] for r in rows], .5), "total_p90": p([r["total"] for r in rows], .9)}
        for m, rows in (("warm", warm), ("cold", cold))}
(OUT / "latency_summary.json").write_text(json.dumps(summ, indent=1))
print(json.dumps(summ, indent=1))
