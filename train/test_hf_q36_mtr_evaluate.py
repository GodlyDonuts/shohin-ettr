from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

import hf_q36_mtr_evaluate as module
from q36_mtr_roles import ROLE_CHECKPOINT_SCHEMA, TRAINABLE_PARAMETERS, role_contract
from shared_post_mlp_revision import trainable_state_sha256


class _Parameter:
    requires_grad = True
    dtype = __import__("torch").float32

    def __init__(self, count: int) -> None:
        self.count = count

    def numel(self) -> int:
        return self.count


class _Model:
    def __init__(self) -> None:
        self.text_model = SimpleNamespace(layers=[object() for _ in range(64)])

    def named_parameters(self):
        yield "blocks.0.adapter_a.weight", _Parameter(TRAINABLE_PARAMETERS // 2)
        yield "blocks.0.adapter_b.weight", _Parameter(TRAINABLE_PARAMETERS // 2)


def _metadata(role: str, arm: str) -> dict:
    model = _Model()
    names = [name for name, _ in model.named_parameters()]
    import hashlib

    metadata = {
        **role_contract(role),
        "update": 256,
        "trainable_parameter_name_sha256": hashlib.sha256(
            "\n".join(names).encode()
        ).hexdigest(),
        "controlled_layer_indices": list(range(48, 64)),
        "draft_control": "draft_unavailable" if arm == "draft_hidden" else "normal",
        "internal_draft_visible": role == "aligned",
        "draft_token_bytes_present": role != "owner",
        "draft_information_available": role == "aligned",
        "draft_attention_applied": role == "draft_hidden",
    }
    return metadata


@pytest.mark.parametrize(
    ("arm", "role", "fields"),
    [
        ("revision", "aligned", ["question"]),
        ("unchanged", "owner", ["question"]),
        ("self_refinement", "owner", ["source_prompt", "internal_draft.completion"]),
        ("draft_hidden", "draft_hidden", ["question"]),
    ],
)
def test_arm_role_and_visibility_contract(
    arm: str, role: str, fields: list[str]
) -> None:
    receipt = module.validate_adapter(_Model(), _metadata(role, arm), arm)
    assert receipt["role"] == role
    assert receipt["controlled_layer_indices"] == list(range(48, 64))
    assert module.model_visible_runtime_fields(arm) == fields


def test_wrong_checkpoint_role_fails_closed() -> None:
    with pytest.raises(module.Q36MTREvaluationError):
        module.validate_adapter(_Model(), _metadata("owner", "revision"), "revision")


def _checkpoint(path: Path) -> None:
    per_tensor = TRAINABLE_PARAMETERS // 32
    state = {}
    remaining = TRAINABLE_PARAMETERS
    for index in range(32):
        count = remaining if index == 31 else per_tensor
        suffix = "adapter_a.weight" if index % 2 == 0 else "adapter_b.weight"
        state[f"backbone.layers.{index // 2}.{suffix}"] = __import__("torch").zeros(
            count, dtype=__import__("torch").float32
        )
        remaining -= count
    metadata = {
        **role_contract("owner"),
        "optimizer_state_serialized": False,
        "checkpoint_trainable_only": True,
        "router_expert_checkpoint_tensors": 0,
        "serialization_restore_exact": True,
        "final_trainable_state_sha256": trainable_state_sha256(state),
    }
    __import__("torch").save(
        {
            "schema": ROLE_CHECKPOINT_SCHEMA,
            "update": 256,
            "trainable_state": state,
            "metadata": metadata,
        },
        path,
    )


def test_q36_checkpoint_loader_is_weights_only_and_trainable_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / "owner.pt"
    _checkpoint(path)
    payload = module.load_q36_checkpoint_payload(path)
    assert "optimizer" not in payload
    forged = __import__("torch").load(path, map_location="cpu", weights_only=True)
    forged["optimizer"] = {"state": {}}
    __import__("torch").save(forged, path)
    with pytest.raises(module.Q36MTREvaluationError):
        module.load_q36_checkpoint_payload(path)


def test_q36_prompt_accounting_never_adds_special_tokens() -> None:
    observed = []

    class Tokenizer:
        def __call__(self, rendered, **kwargs):
            observed.append(kwargs)
            return {"attention_mask": [[1, 1] for _ in rendered]}

    assert module.q36_nonpadding_prompt_tokens(Tokenizer(), ["rendered"]) == 2
    assert observed == [
        {
            "padding": True,
            "return_attention_mask": True,
            "add_special_tokens": False,
        }
    ]


def test_hidden_role_cannot_claim_draft_information_availability() -> None:
    metadata = _metadata("draft_hidden", "draft_hidden")
    metadata["draft_information_available"] = True
    with pytest.raises(module.Q36MTREvaluationError):
        module.validate_adapter(_Model(), metadata, "draft_hidden")


def test_static_wrappers_are_single_h100_and_no_dispatch() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = (root / "train" / "jobs" / "q36_mtr_evaluate.sbatch").read_text()
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in source
    assert "#SBATCH --no-requeue" in source
    assert "q36_require_authorization" in source
    assert "sbatch " not in source
    for name in (
        "q36_mtr_materialize.sbatch",
        "q36_mtr_merge_drafts.sbatch",
        "q36_mtr_merge_evaluation.sbatch",
    ):
        cpu = (root / "pipeline" / "jobs" / name).read_text()
        assert "--gres=" not in cpu
        assert "#SBATCH --no-requeue" in cpu
        assert "q36_require_authorization" in cpu
