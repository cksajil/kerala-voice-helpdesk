"""Phase 3, benchmark only (the held-out runs are in kvh-phase3-heldout). Kaggle CPU (private kernel, internet on for model downloads, Open-Meteo and RSS).

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


# llama-server: latest llama.cpp release for Ubuntu x64 (CPU). Asset names changed over time
# (llama-bXXXX-bin-ubuntu-x64.zip, later .tar.gz): pick by pattern and log the candidates.
import requests, re as _re
rel = requests.get("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest", timeout=30).json()
names = [a["name"] for a in rel.get("assets", [])]
(OUT / "llama_release_assets.txt").write_text(rel.get("tag_name", "") + "\n" + "\n".join(names))
cands = [a for a in rel["assets"] if _re.search(r"ubuntu", a["name"]) and _re.search(r"x64|x86_64|amd64", a["name"])
         and not _re.search(r"vulkan|cuda|rocm|hip|sycl|openvino|musa|kompute|arm|aarch64|s390", a["name"], _re.I)
         and _re.search(r"\.(zip|tar\.gz|tgz)$", a["name"])]
asset = sorted(cands, key=lambda a: len(a["name"]))[0]
zpath = Path("/kaggle/working") / asset["name"]
zpath.write_bytes(requests.get(asset["browser_download_url"], timeout=600).content)
dest = Path("/kaggle/working/llamacpp"); dest.mkdir(exist_ok=True)
if asset["name"].endswith(".zip"):
    zipfile.ZipFile(zpath).extractall(dest)
else:
    tarfile.open(zpath).extractall(dest)
binary = next(p for p in dest.rglob("llama-server") if p.is_file())
binary.chmod(0o755)
(OUT / "llama_server_release.txt").write_text(rel.get("tag_name", "") + " " + asset["name"])
libdirs = sorted({str(p.parent) for p in dest.rglob("*.so*")} | {str(binary.parent)})
code, out = sh(f"{sys.executable} -m eval.bench_llama --server-bin {binary} --gguf {gguf} --threads 2",
               env={**base, "LD_LIBRARY_PATH": ":".join(libdirs)}, timeout=3 * 3600)
(OUT / "bench_log.txt").write_text(out[-30000:])
for f in glob.glob(str(REPO / "eval/results/bench_llama*.json")):
    shutil.copy(f, OUT)
print("done")
