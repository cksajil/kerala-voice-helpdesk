"""The held-out set is frozen: any edit to eval/heldout_v1.jsonl fails here.

A new or corrected set is a new file (heldout_v2.jsonl) with its own hash, so numbers
measured on v1 stay comparable. Nothing here runs the help desk on the questions.
"""
import hashlib
from collections import Counter

from eval import check_heldout as H


def test_heldout_v1_unchanged():
    digest = hashlib.sha256(H.HELDOUT.read_bytes()).hexdigest()
    assert digest == H.SHA256, "eval/heldout_v1.jsonl changed; it is frozen (make a v2 instead)"


def test_heldout_labels_valid():
    assert H.label_errors(H.load()) == []


def test_heldout_composition():
    c = Counter(r["category"] for r in H.load())
    assert sum(c.values()) >= 40
    assert c["in_scope"] + c["kb_gap"] >= 25 and c["weather"] >= 5 and c["news"] >= 5
    assert c["out_of_scope"] >= 5


def test_heldout_not_copied_from_older_sets():
    worst = max(H.nearest(H.load()), key=lambda x: x[1])
    assert worst[1] < H.MAX_SIM, worst
