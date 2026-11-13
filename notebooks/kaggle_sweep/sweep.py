"""HELPDESK_AUTO_BOOST sweep on tuned-on data only (synth dev split, pytest, 20-question set).
Kaggle CPU, no internet needed. Input: dataset sajilck/kvh-code. Output: /kaggle/working/sweep.json"""
import glob, json, os, shutil, subprocess, sys, tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path("/kaggle/working/repo")
shutil.rmtree(REPO, ignore_errors=True)
tgz = glob.glob("/kaggle/input/**/repo.tar.gz", recursive=True)
if tgz:
    REPO.mkdir(); tarfile.open(tgz[0]).extractall(REPO)
else:
    shutil.copytree(Path(sorted(glob.glob("/kaggle/input/**/helpdesk/graph.py", recursive=True), key=len)[0]).parents[1], REPO)
print(open(REPO / "REVISION").read())
subprocess.run(f"{sys.executable} -m pip install -q rank_bm25 langgraph feedparser pytest", shell=True, check=True)


def sh(cmd, env):
    p = subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True,
                       env={**os.environ, "HELPDESK_DENSE": "0", "HELPDESK_LLM": "none", **env})
    return p.stdout + p.stderr


def run(b):
    env = {"HELPDESK_AUTO_BOOST": str(b), "HELPDESK_CANON": "1", "HELPDESK_KB_AUTO": "1"}
    return {"boost": b,
            "synth": sh("python -m eval.run_synth --split dev --kb-auto --show 12", env)[-6000:],
            "pytest": sh("python -m pytest -q tests/ 2>&1 | tail -12", env)[-3000:],
            "run_eval": sh("python -m eval.run_eval | tail -4", env)}


with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(run, [1.0, 1.25, 1.5, 1.75, 2.0]))
Path("/kaggle/working/sweep.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
print("done")
