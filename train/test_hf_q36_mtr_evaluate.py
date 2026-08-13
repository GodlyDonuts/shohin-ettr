from __future__ import annotations

from types import SimpleNamespace

import pytest

import hf_q36_mtr_evaluate as module
from q36_mtr_roles import TRAINABLE_PARAMETERS, role_contract


class _Parameter:
    requires_grad = True

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
