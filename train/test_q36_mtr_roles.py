from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hf_q36_mtr_train_role import (
    Q36MTRTrainingError,
    _validate_arguments,
    full_sequence_position_ids,
)
from q36_mtr_roles import (
    MODEL_REVISION,
    Q36MTRRoleError,
    ROLE_SPECS,
    TRAINABLE_PARAMETERS,
    role_contract,
    sequence_geometry_receipt,
    validate_contract,
    validate_matched_revision_geometry,
    validate_owner_warm_start,
)
from shared_post_mlp_revision import (
    SharedPostMLPError,
    SharedPostMLPProductModel,
)


def test_exact_role_states_are_distinct_and_equal_budget() -> None:
    owner = ROLE_SPECS["owner"]
    aligned = ROLE_SPECS["aligned"]
    hidden = ROLE_SPECS["draft_hidden"]
    assert owner.data_kind == "source_only"
    assert owner.warm_start_role is None
    assert aligned.warm_start_role == hidden.warm_start_role == "owner"
    assert aligned.updates == hidden.updates == 256
    assert aligned.max_rows == hidden.max_rows == 9_655
    assert aligned.max_sequence_length == hidden.max_sequence_length == 4_096
    assert aligned.seed == hidden.seed == 2026080815
    assert aligned.data_seed == hidden.data_seed == 2026080814
    assert aligned.draft_control == "normal"
    assert hidden.draft_control == "draft_unavailable"
    assert role_contract("aligned")["draft_token_bytes_present"] is True
    assert role_contract("aligned")["draft_information_available"] is True
    assert role_contract("draft_hidden")["draft_token_bytes_present"] is True
    assert role_contract("draft_hidden")["draft_information_available"] is False


def test_contract_pins_host_trainables_and_no_router() -> None:
    contract = role_contract("aligned")
    assert contract["model_revision"] == MODEL_REVISION
    assert contract["trainable_parameters"] == TRAINABLE_PARAMETERS
    assert contract["controlled_layers"] == 16
    assert contract["rank"] == contract["alpha"] == 18
    assert contract["router_expert_trainables"] == 0
    assert contract["external_proposer"] is False
    validate_contract(contract, "aligned")
    forged = copy.deepcopy(contract)
    forged["role_spec"]["updates"] = 257
    with pytest.raises(Q36MTRRoleError):
        validate_contract(forged, "aligned")


def test_owner_warm_start_is_exact_source_only_state() -> None:
    metadata = {
        **role_contract("owner"),
        "selected_rows": 100_000,
        "source_only_model_visible": True,
        "internal_draft_visible": False,
        "trainable_parameter_name_sha256": "a" * 64,
    }
    validate_owner_warm_start(
        metadata,
        checkpoint_update=256,
        trainable_parameters=TRAINABLE_PARAMETERS,
        trainable_parameter_name_sha256="a" * 64,
    )
    metadata["internal_draft_visible"] = True
    with pytest.raises(Q36MTRRoleError):
        validate_owner_warm_start(
            metadata,
            checkpoint_update=256,
            trainable_parameters=TRAINABLE_PARAMETERS,
            trainable_parameter_name_sha256="a" * 64,
        )


def test_hidden_control_preserves_tokens_and_full_positions() -> None:
    prompts = [[11, 12, 13, 14], [21, 22, 23]]
    responses = [[31, 32], [41, 42, 43]]
    draft_masks = [[1, 0, 0, 1], [1, 0, 1]]
    aligned = sequence_geometry_receipt(prompts, responses, draft_masks)
    hidden = sequence_geometry_receipt(prompts, responses, draft_masks)
    validate_matched_revision_geometry(aligned, hidden)
    assert hidden["draft_masked_tokens"] == 3
    attention = torch.tensor([[1, 0, 0, 1, 1, 1], [1, 0, 1, 1, 1, 0]])
    positions = full_sequence_position_ids(attention)
    assert positions.tolist() == [list(range(6)), list(range(6))]


class _OffsetTokenizer:
    def __call__(self, text: str, **_kwargs):
        return {
            "input_ids": [ord(character) % 127 for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def test_generation_mask_is_bound_to_exact_prompt_ids() -> None:
    model = SharedPostMLPProductModel.__new__(SharedPostMLPProductModel)
    torch.nn.Module.__init__(model)
    model.draft_control = "draft_unavailable"
    model.text_model = SimpleNamespace(embed_tokens=torch.nn.Embedding(127, 4))
    model._generation_prompt_attention = None
    model._generation_position_ids = None
    model._generation_prompt_ids = None
    prompt = (
        "Original problem:\nP\n\nInternal draft:\nDRAFT\n\nReturn answer"
        "\n\nOriginal problem:\nP"
    )
    tokenizer = _OffsetTokenizer()
    token_ids = tokenizer(prompt)["input_ids"]
    input_ids = torch.tensor([token_ids])
    attention = torch.ones_like(input_ids)
    model.prepare_generation_draft_attention(tokenizer, [prompt], input_ids, attention)
    _, hidden_attention = model.generation_embeddings(input_ids, attention)
    assert hidden_attention.shape == attention.shape
    assert int((hidden_attention == 0).sum()) == len("DRAFT")
    assert model.generation_position_ids().tolist() == [list(range(len(token_ids)))]
    substituted = input_ids.clone()
    substituted[0, 0] = (substituted[0, 0] + 1) % 127
    with pytest.raises(SharedPostMLPError):
        model.generation_embeddings(substituted, attention)


def test_matched_geometry_rejects_token_or_position_drift() -> None:
    aligned = sequence_geometry_receipt([[1, 2, 3]], [[4, 5]], [[1, 0, 1]])
    hidden = sequence_geometry_receipt([[1, 9, 3]], [[4, 5]], [[1, 0, 1]])
    with pytest.raises(Q36MTRRoleError):
        validate_matched_revision_geometry(aligned, hidden)
    hidden = copy.deepcopy(aligned)
    hidden["position_geometry_sha256"] = "0" * 64
    with pytest.raises(Q36MTRRoleError):
        validate_matched_revision_geometry(aligned, hidden)


def _arguments(role: str) -> SimpleNamespace:
    spec = ROLE_SPECS[role]
    return SimpleNamespace(
        role=role,
        model_revision=MODEL_REVISION,
        model_config_sha256=role_contract(role)["model_config_sha256"],
        quantization="nf4",
        updates=spec.updates,
        max_rows=spec.max_rows,
        max_sequence_length=spec.max_sequence_length,
        learning_rate=spec.learning_rate,
        gradient_accumulation=spec.gradient_accumulation,
        seed=spec.seed,
        data_seed=spec.data_seed,
        controlled_layers=16,
        rank=18,
        alpha=18.0,
        warm_start_checkpoint=(None if role == "owner" else Path("owner.pt")),
        batch_size=1,
        checkpoint_interval=spec.updates,
    )


@pytest.mark.parametrize("role", ["owner", "aligned", "draft_hidden"])
def test_cli_role_settings_are_frozen(role: str) -> None:
    args = _arguments(role)
    _validate_arguments(args)
    args.updates += 1
    with pytest.raises(Q36MTRTrainingError):
        _validate_arguments(args)


def test_role_wrappers_are_single_h100_no_requeue_and_authorized_only() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("q36_mtr_train_role.sbatch", "q36_mtr_generate_drafts.sbatch"):
        source = (root / "train" / "jobs" / name).read_text(encoding="utf-8")
        assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in source
        assert "#SBATCH --partition=normal" in source
        assert "#SBATCH --no-requeue" in source
        assert "q36_require_authorization" in source
        assert "--exclude=evc26,evc29,evc31,evc32,evc33,evc37,evc38,evc46" in source
        assert "sbatch " not in source
