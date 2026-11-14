"""Fetch the official core-service sources and survey the LSGD Panchayat Wiki (ingest/core_fetch.py).
Light CPU; on Kaggle because the work container cannot reach lsgkerala.gov.in. Outputs only the
fetched files, the manifest and the wiki title list (no build folders), under /kaggle/working/out."""
import glob, shutil, subprocess, sys, tarfile
from pathlib import Path

OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
REPO = Path("/kaggle/temp/repo")
shutil.rmtree(REPO, ignore_errors=True); REPO.mkdir(parents=True)
tgz = glob.glob("/kaggle/input/**/repo.tar.gz", recursive=True)
if tgz:
    tarfile.open(tgz[0]).extractall(REPO)
else:
    shutil.rmtree(REPO); shutil.copytree(Path(sorted(glob.glob("/kaggle/input/**/helpdesk/graph.py", recursive=True), key=len)[0]).parents[1], REPO)
shutil.copy(REPO / "REVISION", OUT)
p = subprocess.run(f"{sys.executable} -m ingest.core_fetch", shell=True, cwd=REPO, capture_output=True, text=True, timeout=3 * 3600)
(OUT / "log.txt").write_text(p.stdout[-50000:] + p.stderr[-20000:])
print(p.stdout[-5000:], p.stderr[-3000:])
shutil.copytree(REPO / "ingest/raw/core", OUT / "core", dirs_exist_ok=True)
