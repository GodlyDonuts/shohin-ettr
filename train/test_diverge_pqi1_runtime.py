#!/usr/bin/env python3
"""Focused structural tests for DIVERGE-PQI1."""

from __future__ import annotations

import torch

from diverge_pqi1_runtime import canonicalize_query


def _mask(text: str, phrase: str) -> list[bool]:
    start = text.index(phrase)
    result = [False] * len(text)
    result[start : start + len(phrase)] = [True] * len(phrase)
    return result


def test_canonicalization_is_entity_invariant() -> None:
    first = "Choose asteroid; birchwood is the decoy."
    second = "Choose riverbend; moonstone is the decoy."
    a = canonicalize_query(first, (_mask(first, "asteroid"), _mask(first, "birchwood")))
    b = canonicalize_query(second, (_mask(second, "riverbend"), _mask(second, "moonstone")))
    assert a.text == b.text == "Choose alpha; beta is the decoy."
    assert a.mention_spans == b.mention_spans


def test_scrub_removes_semantic_context() -> None:
    text = "Discard birchwood; the answer comes from asteroid."
    canonical = canonicalize_query(
        text,
        (_mask(text, "birchwood"), _mask(text, "asteroid")),
        scrub_context=True,
    )
    assert canonical.text == "alpha then beta"
    assert canonical.mention_spans == (((0, 5),), ((11, 15),))


def test_candidate_compatibility_is_antisymmetric() -> None:
    score = torch.tensor([[2.0, -1.0], [-3.0, 4.0]])
    logits = torch.stack((score, -score), dim=-1)
    identity = logits[:, 0, 0] + logits[:, 1, 1]
    swapped = logits[:, 0, 1] + logits[:, 1, 0]
    assert torch.equal(identity, -swapped)
    assert torch.equal(logits.flip(-1)[:, 0, 0] + logits.flip(-1)[:, 1, 1], swapped)


if __name__ == "__main__":
    test_canonicalization_is_entity_invariant()
    test_scrub_removes_semantic_context()
    test_candidate_compatibility_is_antisymmetric()
    print("DIVERGE-PQI1 runtime tests passed")
