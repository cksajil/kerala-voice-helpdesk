"""LLM client with two backends.

1. HTTP (default): any OpenAI-compatible endpoint, e.g.
       python -m llama_cpp.server --model gemma-4-E2B-it-Q4_0.gguf --port 8080 --n_ctx 4096
2. In-process (Hugging Face Space): reuse the Llama object the Space already loads
       from helpdesk import llm; llm.use_llama_cpp(my_llama)
Set HELPDESK_LLM=none to run without an LLM (keyword router + extractive answers).
"""
from __future__ import annotations

import os

import requests

LLM_URL = os.getenv("HELPDESK_LLM_URL", "http://127.0.0.1:8080/v1/chat/completions")
LLM_MODEL = os.getenv("HELPDESK_LLM_MODEL", "gemma-4-e2b-it")
LLM_TIMEOUT = float(os.getenv("HELPDESK_LLM_TIMEOUT", "300"))  # slow CPUs need more than 120 s
_llama = None  # llama_cpp.Llama instance when running in-process


def use_llama_cpp(llama) -> None:
    global _llama
    _llama = llama


def enabled() -> bool:
    return os.getenv("HELPDESK_LLM", "llama") != "none"


def chat(system: str, user: str, max_tokens: int = 200, temperature: float = 0.2) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if _llama is not None:
        out = _llama.create_chat_completion(messages=messages, max_tokens=max_tokens,
                                            temperature=temperature)
        return out["choices"][0]["message"]["content"].strip()
    r = requests.post(LLM_URL, timeout=LLM_TIMEOUT, json={
        "model": LLM_MODEL, "temperature": temperature, "max_tokens": max_tokens, "messages": messages})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()
