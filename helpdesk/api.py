"""HTTP service for the help desk (Phase 4).

    uvicorn helpdesk.api:app --host 0.0.0.0 --port 8000

POST /ask     {"question": "...", "context": {...}}  -> answer, route, sources, timings, context
GET  /health  status "loading" | "warming" | "ok" | "error", warm-up progress, settings

The service is stateless: each reply carries a "context" object (the previous question, route,
services, tool and topic, exactly what helpdesk.cli carries between turns); send it back with the
next question to keep a follow-up ("അതിന് എത്ര ദിവസം?") on the same service.

LLM back ends (HELPDESK_LLM_BACKEND):
  inproc  (default) llama-cpp-python in this process, with a RAM prompt cache. The GGUF comes from
          GEMMA_GGUF or is downloaded once from Hugging Face (public repo, no token) into HF_HOME.
  http    an OpenAI-compatible server at HELPDESK_LLM_URL (llama-server, llama_cpp.server).
  none    no LLM: the best passage is read out (HELPDESK_LLM=none does the same).

Warm-up at start-up (HELPDESK_WARMUP): the model loads, one turn runs, and then the answer prompt
of every curated passage is prefilled into the RAM cache (one generated token each), so a question
answered from that passage skips most of its prefill. "kb" (default) warms data/kb.jsonl, "all"
also warms the kb_auto entries, "none" skips it. Warm-up runs in the background: /ask answers
straight away and /health shows progress. HELPDESK_CACHE_MB sets the cache size (LRU).
Only one turn uses the model at a time (llama.cpp contexts are not thread-safe).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger("helpdesk.api")

BACKEND = os.getenv("HELPDESK_LLM_BACKEND", "inproc")
THREADS = int(os.getenv("N_THREADS", "2"))
N_CTX = int(os.getenv("HELPDESK_N_CTX", "2048"))
CACHE_MB = int(os.getenv("HELPDESK_CACHE_MB", "2048"))
WARMUP = os.getenv("HELPDESK_WARMUP", "kb")
GGUF_REPO = os.getenv("GEMMA_REPO", "ggml-org/gemma-4-E2B-it-GGUF")
GGUF_FILE = os.getenv("GEMMA_FILE", "gemma-4-E2B-it-Q4_0.gguf")
MAX_QUESTION = 500
# The Space's settings unless the caller overrides them.
os.environ.setdefault("HELPDESK_CANON", "1")
os.environ.setdefault("HELPDESK_CTX_K", "1")
os.environ.setdefault("HELPDESK_DENSE", "0")


class Context(BaseModel):
    previous: str = ""
    previous_route: str = ""
    previous_services: list[str] = Field(default_factory=list)
    previous_tool: str = ""
    previous_topic: str = ""


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION)
    context: Context | None = None


class AskOut(BaseModel):
    answer: str
    route: str
    route_by: str | None = None
    refused: bool
    tool: str | None = None
    sources: list[dict]
    timings: dict[str, float]
    context: Context


class Service:
    def __init__(self):
        self.status = "loading"
        self.error = ""
        self.graph = None
        self.retriever = None
        self.model = None
        self.lock = threading.Lock()                 # one turn at a time on the model
        self.warm = {"done": 0, "total": 0, "seconds": 0.0}
        self.started = time.time()

    # --- start-up -------------------------------------------------------------------------
    def load(self):
        from . import llm
        if BACKEND == "none":
            os.environ["HELPDESK_LLM"] = "none"
        elif BACKEND == "inproc" and os.getenv("HELPDESK_LLM", "llama") != "none":
            from llama_cpp import Llama, LlamaRAMCache
            path = os.getenv("GEMMA_GGUF")
            if not path:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(GGUF_REPO, GGUF_FILE)
            # n_threads_batch (prefill) defaults to every core: pin it as on the Space.
            llama = Llama(model_path=path, n_ctx=N_CTX, n_threads=THREADS, n_threads_batch=THREADS, verbose=False)
            llama.set_cache(LlamaRAMCache(capacity_bytes=CACHE_MB << 20))
            llm.use_llama_cpp(llama)
            self.model = Path(path).name
        elif BACKEND == "http":
            self.model = llm.LLM_URL
        from .graph import build_graph
        from .retriever import Retriever
        self.retriever = Retriever(dense=os.getenv("HELPDESK_DENSE") == "1")
        self.graph = build_graph(self.retriever)

    def warm_up(self):
        from . import llm
        from .graph import ANSWER_PROMPT, answer_message, passage
        t0 = time.perf_counter()
        with self.lock:                              # first turn: loads weights into RAM, JIT paths
            self.graph.invoke({"query": "ജനന സർട്ടിഫിക്കറ്റ് എങ്ങനെ കിട്ടും?"})
        docs = []
        if llm.enabled() and BACKEND == "inproc" and WARMUP in ("kb", "all"):
            docs = [d for d in self.retriever.docs if d.get("tier") == 1
                    and (WARMUP == "all" or d.get("kind") != "kb_auto")]
        self.warm["total"] = len(docs)
        self.status = "warming"
        for d in docs:
            with self.lock:                          # a waiting /ask goes between two passages
                llm.chat(ANSWER_PROMPT, answer_message(passage(d), d["title"]), max_tokens=1, temperature=0)
            self.warm["done"] += 1
        self.warm["seconds"] = round(time.perf_counter() - t0, 1)
        self.status = "ok"
        log.info("warm-up done: %s", self.warm)

    def start(self):
        try:
            self.load()
            self.status = "warming"
        except Exception as e:                       # /health reports it; the process stays up
            self.status, self.error = "error", f"{type(e).__name__}: {e}"
            log.exception("start-up failed")
            return

        def bg():
            try:
                self.warm_up()
            except Exception as e:
                self.status, self.error = "error", f"warm-up: {type(e).__name__}: {e}"
                log.exception("warm-up failed")
        threading.Thread(target=bg, daemon=True, name="warmup").start()

    # --- requests -------------------------------------------------------------------------
    def ask(self, question: str, ctx: Context | None) -> dict:
        state, timings = {"query": question}, {}
        if ctx and ctx.previous:
            state.update(ctx.model_dump())
        with self.lock:
            t0 = t = time.perf_counter()
            for upd in self.graph.stream(dict(state), stream_mode="updates"):
                now = time.perf_counter()
                for node, delta in upd.items():
                    timings[node] = round(timings.get(node, 0) + now - t, 3)
                    state.update(delta or {})
                t = now
            timings["total"] = round(time.perf_counter() - t0, 3)
        nxt = Context(previous=question, previous_route=state.get("route") or "",
                      previous_services=state.get("services") or [], previous_tool=state.get("tool") or "",
                      previous_topic=state.get("topic") or "")
        return {"answer": state.get("answer", ""), "route": state.get("route") or "other",
                "route_by": state.get("route_by"), "refused": bool(state.get("refused")),
                "tool": state.get("tool"), "sources": state.get("sources") or [],
                "timings": timings, "context": nxt}


service = Service()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    # Load in a thread so the server answers /health ("loading") while the model loads.
    threading.Thread(target=service.start, daemon=True, name="load").start()
    yield


app = FastAPI(title="Kerala Voice Helpdesk", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    from . import llm
    body = {"status": service.status, "backend": BACKEND if llm.enabled() else "none",
            "model": service.model, "warmup": service.warm, "uptime_s": round(time.time() - service.started),
            "settings": {k: os.getenv(k) for k in ("HELPDESK_CANON", "HELPDESK_CTX_K", "HELPDESK_KB_AUTO",
                                                   "HELPDESK_DENSE", "HELPDESK_AUTO_BOOST")},
            "passages": len(service.retriever.docs) if service.retriever else 0}
    if service.error:
        body["error"] = service.error
    return body


# A plain def: FastAPI runs it in its thread pool, so a long LLM turn does not block /health.
@app.post("/ask", response_model=AskOut)
def ask(body: AskIn):
    if service.graph is None:
        raise HTTPException(503, detail=f"not ready: {service.status} {service.error}".strip())
    q = body.question.strip()
    if not q:
        raise HTTPException(422, detail="empty question")
    return service.ask(q, body.context)
