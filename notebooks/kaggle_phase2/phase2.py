"""Phase 2 on Kaggle CPU (private kernel, internet on, no secrets needed).

Input: dataset sajilck/kvh-code (repo.tar.gz = git archive of the branch).
1. robots.txt + link probe of ksmart.lsgkerala.gov.in and lsgkerala.gov.in (logged, never forced)
2. ingest.fetch --force: re-download every charter (robots.txt obeyed, 2 s apart); a changed charter
   loses its OCR and is re-extracted (Tesseract installed only then)
3. ingest.web: refresh the LSGD website sections
4. build_docs -> canon --outputs-only (cached Gemini answers only, no API calls) -> kb_auto
5. HELPDESK_AUTO_BOOST sweep on tuned-on data only: synth dev split, conversation tests, 20-question set.
   The held-out set (eval/heldout_v1.jsonl) is NOT run here.
Outputs in /kaggle/working/out.
"""
import glob, json, os, shutil, subprocess, sys, tarfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
REPO = Path("/kaggle/working/repo")
LOG = open(OUT / "run.log", "a", encoding="utf-8")


def sh(cmd, env=None, check=False, timeout=None, log=True):
    t = time.time()
    p = subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True,
                       env={**os.environ, **(env or {})}, timeout=timeout)
    out = p.stdout + p.stderr
    if log:
        LOG.write(f"\n$ {cmd}  [{time.time() - t:.0f}s, exit {p.returncode}]\n{out[-20000:]}\n"); LOG.flush()
    print(f"$ {cmd} -> exit {p.returncode} ({time.time() - t:.0f}s)", flush=True)
    if check and p.returncode:
        raise SystemExit(out[-3000:])
    return out


# ---- repo from the dataset
tgz = glob.glob("/kaggle/input/**/repo.tar.gz", recursive=True)[0]
shutil.rmtree(REPO, ignore_errors=True); REPO.mkdir()
tarfile.open(tgz).extractall(REPO)
print(open(REPO / "REVISION").read() if (REPO / "REVISION").exists() else "no REVISION file")
sh(f"{sys.executable} -m pip install -q -r requirements-ingest.txt", check=True)
os.environ.update(HELPDESK_DENSE="0", HELPDESK_LLM="none")

# ---- 1. probes (robots.txt decides; nothing is forced)
probe = {}
for site in ["https://ksmart.lsgkerala.gov.in", "https://lsgkerala.gov.in"]:
    try:
        import requests
        r = requests.get(site + "/robots.txt", timeout=20, headers={"User-Agent": "kerala-voice-helpdesk/0.2"})
        probe[site] = {"robots_status": r.status_code, "robots": r.text[:3000]}
    except Exception as e:
        probe[site] = {"error": repr(e)[:300]}
probe["ksmart_links"] = sh("python -m ingest.web --links https://ksmart.lsgkerala.gov.in/ 2>&1 | head -80")[-6000:]
(OUT / "probe.json").write_text(json.dumps(probe, ensure_ascii=False, indent=1))

# ---- 2-3. fetch and crawl
fetch_log = sh("python -m ingest.fetch --force", timeout=3600)
(OUT / "fetch.log").write_text(fetch_log)
changed = [l for l in fetch_log.splitlines() if "CHANGED" in l]
web_log = sh("python -m ingest.web --max 30", timeout=3600)
(OUT / "web.log").write_text(web_log)
if changed or any(not (REPO / "ingest/pages" / (p.stem + ".jsonl")).exists() for p in (REPO / "ingest/raw").glob("*.pdf")):
    sh("apt-get -qq update && apt-get -qq install -y tesseract-ocr tesseract-ocr-mal > /dev/null")
    sh("python -m ingest.extract --jobs 4 --budget 300", timeout=6 * 3600)

# ---- 4. rebuild
sh("python -m ingest.build_docs", check=True)
sh("python -m ingest.check_kb")
sh("python -m ingest.canon --outputs-only", check=True)
sh("python -m ingest.kb_auto", check=True)
tests = sh("python -m pytest -q tests/ 2>&1 | tail -3")

# ---- 5. boost sweep (tuned-on sets only), 4 in parallel
def run_boost(b):
    env = {"HELPDESK_AUTO_BOOST": str(b), "HELPDESK_CANON": "1", "HELPDESK_KB_AUTO": "1"}
    synth = sh("python -m eval.run_synth --split dev --kb-auto", env=env, timeout=7200, log=False)
    conv = sh("python -m pytest -q tests/test_helpdesk.py 2>&1 | tail -15", env=env, log=False)
    ev = sh("python -m eval.run_eval | tail -4", env=env, log=False)
    return {"boost": b, "synth": synth[-2500:], "pytest": conv[-3000:], "run_eval": ev}

with ThreadPoolExecutor(4) as ex:
    sweep = list(ex.map(run_boost, [1.0, 1.25, 1.5, 1.75, 2.0]))
(OUT / "sweep.json").write_text(json.dumps(sweep, ensure_ascii=False, indent=1))

# ---- save
for k in ["data/docs.jsonl", "data/canon.jsonl", "data/kb_auto.jsonl", "ingest/kb_auto_report.md",
          "ingest/raw/manifest.json", "ingest/raw/web_manifest.json", "ingest/report.md", "ingest/kb_check.md",
          "ingest/changes.md", "ingest/charter_changes.md", "ingest/web_pdfs.txt", "ingest/pages"]:
    src = REPO / k
    if src.is_dir():
        shutil.copytree(src, OUT / k, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.partial"))
    elif src.exists():
        (OUT / k).parent.mkdir(parents=True, exist_ok=True); shutil.copy(src, OUT / k)
(OUT / "summary.json").write_text(json.dumps({"changed_charters": changed, "tests": tests}, ensure_ascii=False, indent=1))
print("done; changed charters:", changed)
