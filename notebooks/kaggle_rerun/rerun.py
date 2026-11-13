"""Held-out v1 re-run after the Phase 3 fixes (seen-failure numbers, not blind). Kaggle CPU (private kernel, internet on for model downloads, Open-Meteo and RSS).

Input: dataset sajilck/kvh-code. Every help-desk run uses 2 threads (llama.cpp and torch), close to the
Space's share; dense passage embeddings are built beforehand with 4 threads and timed separately.
1. eval.run_heldout on eval/heldout_v1.jsonl: old/expanded KB x BM25/BM25+dense, Gemma 4 E2B Q4_0 in-process
2. eval.bench_llama: llama-cpp-python vs llama-server (latest llama.cpp release) on identical prompts
Outputs: /kaggle/working/out (results CSV/JSON, logs, versions).
"""
import glob, json, os, shutil, subprocess, sys, tarfile, time, zipfile
from pathlib import Path

OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
REPO = Path("/kaggle/working/repo")
LOG = open(OUT / "run.log", "a", encoding="utf-8")


def sh(cmd, env=None, timeout=None):
    t = time.time()
    p = subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True,
                       env={**os.environ, **(env or {})}, timeout=timeout)
    out = p.stdout + p.stderr
    LOG.write(f"\n$ {cmd}  {env or ''} [{time.time() - t:.0f}s, exit {p.returncode}]\n{out[-30000:]}\n"); LOG.flush()
    print(f"$ {cmd} -> exit {p.returncode} ({time.time() - t:.0f}s)", flush=True)
    return p.returncode, out


shutil.rmtree(REPO, ignore_errors=True)
tgz = glob.glob("/kaggle/input/**/repo.tar.gz", recursive=True)
if tgz:
    REPO.mkdir(); tarfile.open(tgz[0]).extractall(REPO)
else:
    shutil.copytree(Path(sorted(glob.glob("/kaggle/input/**/helpdesk/graph.py", recursive=True), key=len)[0]).parents[1], REPO)
(OUT / "REVISION").write_text((REPO / "REVISION").read_text())
print((REPO / "REVISION").read_text())

sh(f"{sys.executable} -m pip install -q rank_bm25 langgraph feedparser requests huggingface_hub ", timeout=3600)
sh(f"CMAKE_ARGS='-DGGML_NATIVE=ON' {sys.executable} -m pip install -q llama-cpp-python==0.3.36", timeout=5400)
sh(f"{sys.executable} -m pip freeze | grep -iE 'llama|torch|sentence|transformers|langgraph|rank' > /kaggle/working/out/versions.txt")
sh("lscpu | head -20 > /kaggle/working/out/cpu.txt; free -g >> /kaggle/working/out/cpu.txt")

from huggingface_hub import hf_hub_download
gguf = hf_hub_download("ggml-org/gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-Q4_0.gguf")
base = {"GEMMA_GGUF": gguf, "N_THREADS": "2", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
        "HELPDESK_CANON": "1", "HELPDESK_CTX_K": "1", "TOKENIZERS_PARALLELISM": "false"}


# BM25 only (dense stays off), old and expanded KB, same settings as the blind run.
for kb in ("old", "expanded"):
    tag = f"rerun_{kb}_bm25"
    code, out = sh(f"{sys.executable} -m eval.run_heldout --kb {kb} --tag {tag}", env=base, timeout=4 * 3600)
    for f in glob.glob(str(REPO / f"eval/results/heldout_{tag}.*")):
        shutil.copy(f, OUT)
    (OUT / f"log_{tag}.txt").write_text(out[-60000:])
sh(f"{sys.executable} -m eval.run_route_dev > /kaggle/working/out/route_dev.txt 2>&1")
print("done")
