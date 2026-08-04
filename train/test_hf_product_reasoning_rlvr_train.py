import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from hf_product_reasoning_rlvr_train import (
    PREFIX_CREDIT_REWARD,
    TERMINAL_ONLY_REWARD,
    ProductRLVRTrainError,
    _optimization_parameters,
    _reservoir_reward_rows,
    _shortest_verified_prefix_ids,
    _validate_resume_contract,
    policy_objective,
    standardized_group_advantages,
    verified_terminal_reward,
    verified_trajectory_reward,
)


def test_group_advantages_center_and_scale_mixed_rewards() -> None:
    advantages = standardized_group_advantages(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert float(advantages.mean()) == pytest.approx(0.0, abs=1e-7)
    assert float(advantages.square().mean()) == pytest.approx(1.0)
    assert advantages[0] > 0
    assert torch.all(advantages[1:] < 0)


def test_group_advantages_suppress_uniform_reward_groups() -> None:
    assert torch.equal(
        standardized_group_advantages(torch.ones(4)),
        torch.zeros(4),
    )
    assert torch.equal(
        standardized_group_advantages(torch.zeros(4)),
        torch.zeros(4),
    )


def test_group_advantages_reject_singleton() -> None:
    with pytest.raises(ProductRLVRTrainError, match="at least two"):
        standardized_group_advantages(torch.ones(1))


def test_policy_objective_raises_positive_reward_log_probability() -> None:
    positive_logp = torch.tensor(-2.0, requires_grad=True)
    policy_objective(positive_logp, torch.tensor(1.5)).backward()
    assert positive_logp.grad is not None
    assert float(positive_logp.grad) < 0


def test_policy_objective_lowers_negative_reward_log_probability() -> None:
    negative_logp = torch.tensor(-2.0, requires_grad=True)
    policy_objective(negative_logp, torch.tensor(-0.5)).backward()
    assert negative_logp.grad is not None
    assert float(negative_logp.grad) > 0


def test_lora_only_scope_excludes_other_checkpoint_parameters() -> None:
    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = torch.nn.Linear(2, 2)
            self.layer.lora_a = torch.nn.Linear(2, 1, bias=False)
            self.layer.lora_b = torch.nn.Linear(1, 2, bias=False)

    model = Toy()
    optimized, excluded = _optimization_parameters(model, "lora_only_update")
    assert sum(parameter.numel() for parameter in optimized) == 4
    assert sum(parameter.numel() for parameter in excluded) == 6


def test_all_trainable_scope_optimizes_full_contract() -> None:
    model = torch.nn.Linear(2, 2)
    optimized, excluded = _optimization_parameters(model, "all_trainable")
    assert sum(parameter.numel() for parameter in optimized) == 6
    assert excluded == []


def test_verified_reward_requires_correct_answer_and_termination() -> None:
    assert (
        verified_terminal_reward({"correct": True, "max_token_exhausted": False}) == 1.0
    )
    assert (
        verified_terminal_reward({"correct": True, "max_token_exhausted": True}) == 0.0
    )
    assert (
        verified_terminal_reward({"correct": False, "max_token_exhausted": False})
        == 0.0
    )


def test_prefix_credit_preserves_correct_exhausted_trajectory() -> None:
    candidate = {"correct": True, "max_token_exhausted": True}
    assert verified_trajectory_reward(candidate, TERMINAL_ONLY_REWARD) == 0.0
    assert verified_trajectory_reward(candidate, PREFIX_CREDIT_REWARD) == 0.5
    candidate["max_token_exhausted"] = False
    assert verified_trajectory_reward(candidate, PREFIX_CREDIT_REWARD) == 1.0


def test_prefix_credit_never_rewards_wrong_trajectory() -> None:
    for exhausted in (False, True):
        assert (
            verified_trajectory_reward(
                {"correct": False, "max_token_exhausted": exhausted},
                PREFIX_CREDIT_REWARD,
            )
            == 0.0
        )


def test_verified_prefix_discards_post_answer_loop() -> None:
    class CharacterTokenizer:
        @staticmethod
        def decode(ids: list[int], *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens
            return "".join(chr(value) for value in ids)

    response = "Work carefully. The answer is 42\nLoop forever."
    response_ids = [ord(value) for value in response]
    prefix_ids = _shortest_verified_prefix_ids(
        CharacterTokenizer(),
        {"task": "math500", "answer": r"\boxed{42}"},
        response_ids,
        stride=8,
    )
    prefix = CharacterTokenizer.decode(prefix_ids, skip_special_tokens=True)
    assert prefix.endswith("42")
    assert "Loop forever" not in prefix


def test_reward_reservoir_preserves_verifier_fields(tmp_path) -> None:
    path = tmp_path / "reward.jsonl"
    rows = [
        {
            "identity_sha256": f"id-{index}",
            "question": f"q-{index}",
            "answer": str(index),
            "task": "math500",
        }
        for index in range(4)
    ]
    encoded = b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode() for row in rows
    )
    path.write_bytes(encoded)

    selected, digest = _reservoir_reward_rows(path, limit=3, seed=17)

    assert len(selected) == 3
    assert all("identity_sha256" in row and "answer" in row for row in selected)
    assert digest == hashlib.sha256(encoded).hexdigest()


def test_resume_contract_requires_exact_global_cursor() -> None:
    args = SimpleNamespace(
        start_update=5,
        seed=31,
        data_seed=41,
        replay_data_seed=43,
        samples=4,
        groups_per_update=4,
        max_new_tokens=1536,
        replay_weight=0.25,
        schedule_total_updates=100,
        reward_contract=TERMINAL_ONLY_REWARD,
        parameter_scope="all_trainable",
    )
    metadata = {
        "rlvr_algorithm": "single_use_on_policy_group_normalized_reinforce_v1",
        "rlvr_reward": TERMINAL_ONLY_REWARD,
        "data_sha256": "reward",
        "rlvr_replay_data_sha256": "replay",
        "seed": 31,
        "data_seed": 41,
        "rlvr_replay_data_seed": 43,
        "rlvr_samples": 4,
        "rlvr_groups_per_update": 4,
        "rlvr_max_new_tokens": 1536,
        "rlvr_replay_weight": 0.25,
        "rlvr_schedule_total_updates": 100,
        "rlvr_parameter_scope": "all_trainable",
    }

    _validate_resume_contract(5, metadata, args, "reward", "replay")
    with pytest.raises(ProductRLVRTrainError, match="checkpoint_update"):
        _validate_resume_contract(4, metadata, args, "reward", "replay")


def test_zero_offset_rejects_accidental_rlvr_resume() -> None:
    args = SimpleNamespace(start_update=0)
    with pytest.raises(ProductRLVRTrainError, match="zero-offset"):
        _validate_resume_contract(
            5,
            {"rlvr_algorithm": "single_use_on_policy_group_normalized_reinforce_v1"},
            args,
            "reward",
            "replay",
        )
