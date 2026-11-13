"""Geocode data/places_src.csv with Open-Meteo (ingest/places.py), then run the route dev set with the
towns loaded. Light CPU; runs on Kaggle because the container cannot reach the geocoding API."""
import glob, shutil, subprocess, sys, tarfile
from pathlib import Path

OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
REPO = Path("/kaggle/working/repo")
shutil.rmtree(REPO, ignore_errors=True)
tgz = glob.glob("/kaggle/input/**/repo.tar.gz", recursive=True)
if tgz:
    REPO.mkdir(); tarfile.open(tgz[0]).extractall(REPO)
else:
    shutil.copytree(Path(sorted(glob.glob("/kaggle/input/**/helpdesk/graph.py", recursive=True), key=len)[0]).parents[1], REPO)
shutil.copy(REPO / "REVISION", OUT)
for cmd in [f"{sys.executable} -m pip install -q rank_bm25 langgraph feedparser",
            f"{sys.executable} -m ingest.places", f"{sys.executable} -m eval.run_route_dev"]:
    p = subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True)
    print(f"$ {cmd} -> {p.returncode}\n{p.stdout[-20000:]}{p.stderr[-5000:]}", flush=True)
    (OUT / "log.txt").open("a").write(f"$ {cmd} -> {p.returncode}\n{p.stdout}{p.stderr}\n")
shutil.copy(REPO / "data/places.jsonl", OUT)
