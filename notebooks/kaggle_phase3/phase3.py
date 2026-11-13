"""Phase 3 on Kaggle CPU (private kernel, internet on for model downloads, Open-Meteo and RSS).

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

sh(f"{sys.executable} -m pip install -q rank_bm25 langgraph feedparser requests huggingface_hub sentence-transformers", timeout=3600)
sh(f"CMAKE_ARGS='-DGGML_NATIVE=ON' {sys.executable} -m pip install -q llama-cpp-python", timeout=5400)
sh(f"{sys.executable} -m pip freeze | grep -iE 'llama|torch|sentence|transformers|langgraph|rank' > /kaggle/working/out/versions.txt")
sh("lscpu | head -20 > /kaggle/working/out/cpu.txt; free -g >> /kaggle/working/out/cpu.txt")

from huggingface_hub import hf_hub_download
gguf = hf_hub_download("ggml-org/gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-Q4_0.gguf")
base = {"GEMMA_GGUF": gguf, "N_THREADS": "2", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
        "HELPDESK_CANON": "1", "HELPDESK_CTX_K": "1", "TOKENIZERS_PARALLELISM": "false"}

runs = [("old", False), ("expanded", False), ("old", True), ("expanded", True)]
summaries = {}
for kb, dense in runs:
    tag = f"{kb}_{'dense' if dense else 'bm25'}"
    env = dict(base)
    if dense:                     # build (and cache) the passage embeddings with 4 threads, timed apart
        code, out = sh(f"{sys.executable} -c \"import time,os;t=time.time();from helpdesk.retriever import Retriever;"
                       f"r=Retriever(dense=True);print('INDEX', len(r.docs), round(time.time()-t,1), r.encoder is not None)\"",
                       env={**env, "HELPDESK_KB_AUTO": "1" if kb == "expanded" else "0",
                            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}, timeout=4 * 3600)
        (OUT / f"index_{tag}.txt").write_text(out[-3000:])
    code, out = sh(f"{sys.executable} -m eval.run_heldout --kb {kb} {'--dense' if dense else ''} --tag {tag}",
                   env=env, timeout=4 * 3600)
    for f in glob.glob(str(REPO / f"eval/results/heldout_{tag}.*")):
        shutil.copy(f, OUT)
    (OUT / f"log_{tag}.txt").write_text(out[-60000:])

# llama-server built from the llama.cpp source vendored in the same llama-cpp-python release, with the
# same native CPU flags, so both back ends run the same llama.cpp code. (Release binaries were the
# first plan; the repo's "latest" release no longer carries them.)
import llama_cpp as _lc
ver = _lc.__version__
# The sdist straight from PyPI (pip download would prepare build metadata, which hung for 30 min).
import requests
meta = requests.get(f"https://pypi.org/pypi/llama-cpp-python/{ver}/json", timeout=60).json()
url = next(u["url"] for u in meta["urls"] if u["packagetype"] == "sdist")
Path("/kaggle/working/sdist").mkdir(exist_ok=True)
sdist = Path("/kaggle/working/sdist") / url.rsplit("/", 1)[1]
sdist.write_bytes(requests.get(url, timeout=600).content)
tarfile.open(sdist).extractall("/kaggle/working/sdist")
src = next(Path("/kaggle/working/sdist").glob("llama_cpp_python-*/vendor/llama.cpp"))
code, _ = sh(f"cmake -S {src} -B /kaggle/working/lcbuild -DGGML_NATIVE=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF "
             f"-DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build /kaggle/working/lcbuild --target llama-server -j 4",
             timeout=3600)
note = f"built from llama-cpp-python {ver} vendor/llama.cpp"
if code != 0 or not any(p.is_file() for p in Path("/kaggle/working/lcbuild").rglob("llama-server")):
    # The sdist may leave out the server sources: fall back to llama.cpp master, and say so.
    shutil.rmtree("/kaggle/working/lcbuild", ignore_errors=True)
    sh("git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /kaggle/working/llamacpp_src", timeout=1800)
    code, _ = sh("cmake -S /kaggle/working/llamacpp_src -B /kaggle/working/lcbuild -DGGML_NATIVE=ON -DLLAMA_CURL=OFF "
                 "-DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build /kaggle/working/lcbuild "
                 "--target llama-server -j 4", timeout=3600)
    _, head = sh("git -C /kaggle/working/llamacpp_src log -1 --format='%h %cs'")
    note = f"built from llama.cpp master {head.strip()} (NOT the version vendored in llama-cpp-python {ver})"
binary = next(p for p in Path("/kaggle/working/lcbuild").rglob("llama-server") if p.is_file())
(OUT / "llama_server_release.txt").write_text(note)
libdirs = [str(binary.parent)]
code, out = sh(f"{sys.executable} -m eval.bench_llama --server-bin {binary} --gguf {gguf} --threads 2",
               env={**base, "LD_LIBRARY_PATH": ":".join(libdirs)}, timeout=3 * 3600)
(OUT / "bench_log.txt").write_text(out[-30000:])
for f in glob.glob(str(REPO / "eval/results/bench_llama*.json")):
    shutil.copy(f, OUT)
print("done")
