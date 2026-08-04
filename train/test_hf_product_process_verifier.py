from collections import OrderedDict

import torch

from hf_product_process_verifier import (
    ProcessVerifierHead,
    build_balanced_pairs,
    identity_split,
    verifier_text,
)


def _group(identity: str, task: str, outcomes: tuple[bool, ...]):
    return [
        {
            "identity_sha256": identity,
            "task": task,
            "question": f"question {identity}",
            "completion": f"completion {index}",
            "sample_index": index,
            "correct": outcome,
        }
        for index, outcome in enumerate(outcomes)
    ]


def _identity_for_split(split: str, seed: int, prefix: str) -> str:
    for index in range(1000):
        identity = f"{prefix}-{index}"
        if identity_split(identity, seed) == split:
            return identity
    raise AssertionError("split identity not found")


def test_balanced_pairs_exclude_dev_and_final_identities() -> None:
    seed = 31
    train_math = _identity_for_split("train", seed, "math")
    train_science = _identity_for_split("train", seed, "science")
    dev = _identity_for_split("dev", seed, "dev")
    final = _identity_for_split("final", seed, "final")
    grouped = OrderedDict(
        (
            (train_math, _group(train_math, "math500", (True, False, False))),
            (train_science, _group(train_science, "bbh_logic", (False, True))),
            (dev, _group(dev, "math500", (True, False))),
            (final, _group(final, "bbh_logic", (True, False))),
        )
    )
    pairs = build_balanced_pairs(grouped, seed=seed, pairs_per_prompt=1)
    assert set(pairs) == {"bbh_logic", "math500"}
    identities = {pair[0] for rows in pairs.values() for pair in rows}
    assert identities == {train_math, train_science}


def test_identity_split_is_stable_and_total() -> None:
    assert identity_split("same", 7) == identity_split("same", 7)
    assert identity_split("same", 7) in {"train", "dev", "final"}


def test_verifier_prompt_contains_problem_and_candidate() -> None:
    rendered = verifier_text("What is 2+2?", "2+2=4")
    assert "What is 2+2?" in rendered
    assert "2+2=4" in rendered
    assert rendered.endswith("Verifier decision:")


def test_process_verifier_head_scores_each_row() -> None:
    head = ProcessVerifierHead(hidden_size=16, shape_size=4, width=8)
    scores = head(torch.randn(3, 16), torch.randn(3, 4))
    assert scores.shape == (3,)
    scores.sum().backward()
    assert all(parameter.grad is not None for parameter in head.parameters())
