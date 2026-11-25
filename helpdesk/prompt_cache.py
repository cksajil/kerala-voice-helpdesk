"""A RAM prompt cache for llama-cpp-python whose size limit holds.

llama-cpp-python's LlamaRAMCache (0.3.36) counts only the llama.cpp state bytes of each entry
(llama_cache.py: cache_size), but every entry also holds a copy of the logits of the cached prompt
(Llama.save_state: scores=self._scores.copy(), up to n_batch rows x n_vocab float32; Gemma's vocabulary
is 262k, so up to ~0.5 GB an entry). The limit is therefore not a limit: on the first Docker run
(Intel Mac, Docker Desktop with 4.1 GB, 25 Nov 2026) the process was killed during warm-up at about
the 10th-13th passage, whatever HELPDESK_CACHE_MB was.

The stored logits are never used to generate: Llama.load_state sets _requires_eval, and generate()
then evaluates the last prompt token again (reuse_prefix = longest_prefix - 1), which writes fresh
logits. So this cache keeps only the last row (load_state broadcasts it) and counts every array it
keeps. Only logprobs / logits_all need the full rows, and the help desk uses neither.
"""
from __future__ import annotations

from collections import OrderedDict


def entry_bytes(state) -> int:
    return int(state.llama_state_size) + state.scores.nbytes + state.input_ids.nbytes


def compact(state):
    if getattr(state, "scores", None) is not None and len(state.scores) > 1:
        state.scores = state.scores[-1:].copy()
    return state


def make_cache(capacity_bytes: int):
    """A LlamaRAMCache subclass (llama_cpp is imported here, so tests can use CompactCacheMixin)."""
    from llama_cpp import LlamaRAMCache

    class CompactRAMCache(CompactCacheMixin, LlamaRAMCache):
        pass

    return CompactRAMCache(capacity_bytes=capacity_bytes)


class CompactCacheMixin:
    cache_state: OrderedDict
    capacity_bytes: int

    @property
    def cache_size(self) -> int:
        return sum(entry_bytes(s) for s in self.cache_state.values())

    def __setitem__(self, key, value) -> None:
        key = tuple(key)
        self.cache_state.pop(key, None)
        self.cache_state[key] = compact(value)
        while self.cache_size > self.capacity_bytes and self.cache_state:
            self.cache_state.popitem(last=False)

    def stats(self) -> dict:
        return {"entries": len(self.cache_state), "mb": round(self.cache_size / 2**20, 1),
                "capacity_mb": round(self.capacity_bytes / 2**20)}
