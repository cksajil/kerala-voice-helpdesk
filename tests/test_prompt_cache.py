"""The prompt cache's size limit counts every array it keeps (helpdesk/prompt_cache.py)."""
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np

from helpdesk.prompt_cache import CompactCacheMixin, entry_bytes

VOCAB = 1000


class Cache(CompactCacheMixin):
    def __init__(self, capacity_bytes):
        self.capacity_bytes, self.cache_state = capacity_bytes, OrderedDict()


def state(n_tokens, kv=10_000):
    return SimpleNamespace(scores=np.ones((n_tokens, VOCAB), np.float32), input_ids=np.zeros(2048, np.intc),
                           n_tokens=n_tokens, llama_state=b"", llama_state_size=kv)


def test_keeps_only_the_last_logits_row_and_counts_it():
    c = Cache(10 << 20)
    c[(1, 2, 3)] = state(300)
    s = c.cache_state[(1, 2, 3)]
    assert s.scores.shape == (1, VOCAB)
    assert c.cache_size == 10_000 + VOCAB * 4 + 2048 * 4 == entry_bytes(s)


def test_limit_holds():
    one = entry_bytes(Cache(1 << 30).__setitem__((0,), state(300)) or state(1))
    c = Cache(3 * one)
    for i in range(10):
        c[(i,)] = state(300)
    assert len(c.cache_state) == 3 and c.cache_size <= c.capacity_bytes
    assert list(c.cache_state) == [(7,), (8,), (9,)]           # least recently set go first
    assert c.stats()["entries"] == 3


def test_one_row_broadcasts_like_llama_load_state():
    scores = np.zeros((512, VOCAB), np.float32)              # Llama.scores: n_batch rows
    kept = state(300).scores[-1:]
    scores[:300, :] = kept.copy()                            # what Llama.load_state does
    assert (scores[299] == 1).all()
