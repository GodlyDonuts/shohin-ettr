from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hf_product_reasoning_train import reservoir_rows_with_sha256
from hf_q36_mtr_train_role import (
    Q36MTRTrainingError,
    _save_role_checkpoint,
    _validate_arguments,
    chunked_causal_cross_entropy,
    full_sequence_position_ids,
    tokenize_role_rows,
    training_consumption_receipt,
)
from build_pcf1_data import revision_prompt
from q36_mtr_roles import (
    CONTROLLED_LAYER_INDICES,
    LAYER_TYPES,
    MODEL_LAYERS,
    MODEL_REVISION,
    Q36MTRRoleError,
    ROLE_SPECS,
    TRAINABLE_PARAMETERS,
    expected_selected_rows,
    role_contract,
    sequence_geometry_receipt,
    validate_contract,
    validate_backbone_geometry,
    validate_backbone_moe_surface,
    validate_matched_revision_geometry,
    validate_owner_warm_start,
)
from shared_post_mlp_revision import (
    SharedPostMLPError,
    SharedPostMLPProductModel,
)
from ttr1_revision import tokenize_with_draft_mask


def test_exact_role_states_are_distinct_and_equal_budget() -> None:
    owner = ROLE_SPECS["owner"]
    aligned = ROLE_SPECS["aligned"]
    hidden = ROLE_SPECS["draft_hidden"]
    assert owner.data_kind == "source_only"
    assert owner.warm_start_role is None
    assert owner.max_rows == 100_000
    assert expected_selected_rows("owner") == 26_387
    assert expected_selected_rows("aligned") == 9_655
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
    assert contract["model_layers"] == MODEL_LAYERS == 40
    assert contract["controlled_layer_indices"] == list(CONTROLLED_LAYER_INDICES)
    assert contract["num_experts"] == 256
    assert contract["router_top_k"] == 8
    assert contract["model_loader"] == "causal"
    assert contract["causal_model_class"] == "Qwen3_5MoeForCausalLM"
    assert contract["rank"] == contract["alpha"] == 18
    assert contract["router_expert_trainables"] == 0
    assert contract["external_proposer"] is False
    validate_contract(contract, "aligned")
    forged = copy.deepcopy(contract)
    forged["role_spec"]["updates"] = 257
    with pytest.raises(Q36MTRRoleError):
        validate_contract(forged, "aligned")


def _exact_backbone():
    text_config = SimpleNamespace(
        model_type="qwen3_5_moe_text",
        hidden_size=2048,
        num_hidden_layers=40,
        num_experts=256,
        num_experts_per_tok=8,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        vocab_size=248_320,
        layer_types=list(LAYER_TYPES),
    )
    model_type = type("Qwen3_5MoeForCausalLM", (), {})
    model = model_type()
    model.config = SimpleNamespace(model_type="qwen3_5_moe", text_config=text_config)
    model.model = SimpleNamespace(
        layers=[
            SimpleNamespace(
                block_type=layer_type,
                mlp=SimpleNamespace(
                    gate=SimpleNamespace(top_k=8, num_experts=256, hidden_dim=2048),
                    experts=SimpleNamespace(
                        num_experts=256, hidden_dim=2048, intermediate_dim=512
                    ),
                    shared_expert=SimpleNamespace(
                        hidden_size=2048, intermediate_size=512
                    ),
                    shared_expert_gate=SimpleNamespace(
                        in_features=2048, out_features=1
                    ),
                ),
            )
            for layer_type in LAYER_TYPES
        ]
    )
    return model


def test_exact_q36_host_geometry_is_admitted_and_mutations_fail() -> None:
    exact = _exact_backbone()
    assert validate_backbone_geometry(exact) == list(CONTROLLED_LAYER_INDICES)
    resolved_causal = _exact_backbone()
    resolved_causal.config = resolved_causal.config.text_config
    assert validate_backbone_geometry(resolved_causal) == list(CONTROLLED_LAYER_INDICES)
    surface = validate_backbone_moe_surface(exact)
    assert surface["layers"] == 40
    assert surface["controlled_layer_indices"] == list(CONTROLLED_LAYER_INDICES)
    assert len(surface["native_router_expert_geometry_sha256"]) == 64
    for field, forged in (
        ("num_hidden_layers", 64),
        ("num_experts", 255),
        ("num_experts_per_tok", 4),
        ("moe_intermediate_size", 1024),
    ):
        changed = _exact_backbone()
        setattr(changed.config.text_config, field, forged)
        with pytest.raises(Q36MTRRoleError):
            validate_backbone_geometry(changed)

    wrong_layout = _exact_backbone()
    wrong_layout.config = SimpleNamespace(
        **{
            **vars(wrong_layout.config.text_config),
            "model_type": "qwen3_5_moe_other",
        }
    )
    with pytest.raises(Q36MTRRoleError):
        validate_backbone_geometry(wrong_layout)

    for field, forged in (
        ("top_k", 4),
        ("num_experts", 255),
        ("hidden_dim", 4096),
    ):
        changed = _exact_backbone()
        setattr(changed.model.layers[24].mlp.gate, field, forged)
        with pytest.raises(Q36MTRRoleError):
            validate_backbone_moe_surface(changed)


def test_role_checkpoint_contains_no_optimizer_or_native_moe_state(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    checkpoint = tmp_path / "role.pt"
    _save_role_checkpoint(checkpoint, model, 256, {"role": "owner"})
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert set(payload) == {"schema", "update", "trainable_state", "metadata"}
    assert "optimizer" not in payload
    assert set(payload["trainable_state"]) == {"weight"}


def test_owner_warm_start_is_exact_source_only_state() -> None:
    metadata = {
        **role_contract("owner"),
        "selected_rows": 26_387,
        "source_only_model_visible": True,
        "internal_draft_visible": False,
        "trainable_parameter_name_sha256": "a" * 64,
        "final_trainable_state_sha256": "b" * 64,
        "serialization_restore_exact": True,
    }
    validate_owner_warm_start(
        metadata,
        checkpoint_update=256,
        trainable_parameters=TRAINABLE_PARAMETERS,
        trainable_parameter_name_sha256="a" * 64,
        loaded_trainable_state_sha256="b" * 64,
    )
    metadata["internal_draft_visible"] = True
    with pytest.raises(Q36MTRRoleError):
        validate_owner_warm_start(
            metadata,
            checkpoint_update=256,
            trainable_parameters=TRAINABLE_PARAMETERS,
            trainable_parameter_name_sha256="a" * 64,
            loaded_trainable_state_sha256="b" * 64,
        )
    metadata["internal_draft_visible"] = False
    with pytest.raises(Q36MTRRoleError):
        validate_owner_warm_start(
            metadata,
            checkpoint_update=256,
            trainable_parameters=TRAINABLE_PARAMETERS,
            trainable_parameter_name_sha256="a" * 64,
            loaded_trainable_state_sha256="c" * 64,
        )


class _BudgetTokenizer:
    chat_template = None
    eos_token_id = 3

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [1] * (3 if text == "answer" else 5)


def test_owner_sequence_budget_is_pre_eos_like_surviving_dense_recipe() -> None:
    prompts, responses, masks, receipt = tokenize_role_rows(
        _BudgetTokenizer(),
        [{"question": "problem", "response": "answer"}],
        role="owner",
        max_sequence_length=8,
    )
    assert len(prompts[0]) + len(responses[0]) == 9
    assert masks == [[1] * 5]
    assert receipt["maximum_observed_tokens"] == 9
    assert receipt["maximum_sequence_length"] == 8
    assert receipt["eos_token_allowance"] == 1


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


def test_q36_task_agnostic_revision_prompt_has_exact_draft_mask() -> None:
    tokenizer = _OffsetTokenizer()
    prompt = revision_prompt("P", "DRAFT")
    token_ids, attention, span = tokenize_with_draft_mask(tokenizer, prompt)
    assert len(token_ids) == len(attention) == len(prompt)
    assert prompt[slice(*span)] == "DRAFT"
    assert int(attention.count(0)) == len("DRAFT")
    assert all(attention[index] == 0 for index in range(*span))


def test_matched_geometry_rejects_token_or_position_drift() -> None:
    aligned = sequence_geometry_receipt([[1, 2, 3]], [[4, 5]], [[1, 0, 1]])
    hidden = sequence_geometry_receipt([[1, 9, 3]], [[4, 5]], [[1, 0, 1]])
    with pytest.raises(Q36MTRRoleError):
        validate_matched_revision_geometry(aligned, hidden)
    hidden = copy.deepcopy(aligned)
    hidden["position_geometry_sha256"] = "0" * 64
    with pytest.raises(Q36MTRRoleError):
        validate_matched_revision_geometry(aligned, hidden)


def test_training_consumption_binds_effective_prefix_not_loaded_pool() -> None:
    examples = [([index], [index + 20], [1]) for index in range(5)]
    receipt = training_consumption_receipt(
        examples,
        updates=2,
        gradient_accumulation=2,
        batch_size=1,
    )
    assert receipt["dataset_presentations"] == 5
    assert receipt["microsteps"] == 4
    assert receipt["consumed_presentations"] == 4
    assert receipt["unique_consumed_presentations"] == 4
    assert receipt["complete_dataset_cycles"] == 0
    assert receipt["partial_cycle_presentations"] == 4
    assert (
        receipt["presentation_index_sha256"]
        == hashlib.sha256(b"0\n1\n2\n3\n").hexdigest()
    )


def test_owner_population_fix_preserves_reservoir_shuffle(tmp_path: Path) -> None:
    data = tmp_path / "b1.jsonl"
    data.write_text(
        "".join(
            f'{{"question":"q{index}","response":"r{index}"}}\n' for index in range(7)
        ),
        encoding="utf-8",
    )
    ceiling_rows, ceiling_sha = reservoir_rows_with_sha256(data, 100_000, 20260802)
    exact_rows, exact_sha = reservoir_rows_with_sha256(data, 26_387, 20260802)
    assert exact_rows == ceiling_rows
    assert exact_sha == ceiling_sha


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
        engineering_sequence_extension=False,
        engineering_loss_chunk_size=None,
    )


@pytest.mark.parametrize("role", ["owner", "aligned", "draft_hidden"])
def test_cli_role_settings_are_frozen(role: str) -> None:
    args = _arguments(role)
    _validate_arguments(args)
    args.updates += 1
    with pytest.raises(Q36MTRTrainingError):
        _validate_arguments(args)


def test_aligned_role_allows_only_exact_engineering_sequence_extension() -> None:
    args = _arguments("aligned")
    args.max_sequence_length = 4_224
    args.engineering_sequence_extension = True
    args.engineering_loss_chunk_size = 512
    _validate_arguments(args)
    args.max_sequence_length = 4_225
    with pytest.raises(Q36MTRTrainingError, match="sequence"):
        _validate_arguments(args)


def test_role_wrappers_are_single_h100_no_requeue_and_authorized_only() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("q36_mtr_train_role.sbatch", "q36_mtr_generate_drafts.sbatch"):
        source = (root / "train" / "jobs" / name).read_text(encoding="utf-8")
        assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in source
        assert "#SBATCH --partition=normal" in source
        assert "#SBATCH --no-requeue" in source
        assert "q36_require_authorization" in source
        assert (
            "--exclude=evc26,evc29,evc31,evc32,evc33,evc34,evc37,evc38,evc43,evc46,evc50"
            in source
        )
        assert "sbatch " not in source


def test_draft_wrapper_allows_the_observed_long_tail_to_finish() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "train" / "jobs" / "q36_mtr_generate_drafts.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --time=04:00:00" in source
    assert "#SBATCH --time=02:30:00" not in source


def test_role_wrapper_binds_the_exact_engineering_sequence_extension() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "train" / "jobs" / "q36_mtr_train_role.sbatch").read_text(
        encoding="utf-8"
    )
    assert '"$MAX_SEQUENCE_LENGTH" == "4224"' in source
    assert "--engineering-sequence-extension" in source
    assert "--engineering-loss-chunk-size 512" in source
    assert "TRAIN_SCRIPT_SHA256" in source
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in source
    assert "#SBATCH --time=04:00:00" in source


def test_chunked_causal_loss_matches_full_loss_and_gradients() -> None:
    torch.manual_seed(17)
    hidden_full = torch.randn(2, 11, 7, requires_grad=True)
    hidden_chunked = hidden_full.detach().clone().requires_grad_(True)
    head_full = torch.nn.Linear(7, 13, bias=False)
    head_chunked = copy.deepcopy(head_full)
    labels = torch.randint(0, 13, (2, 11))
    labels[0, :3] = -100

    logits = head_full(hidden_full)
    full_loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 13),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    chunked_loss = chunked_causal_cross_entropy(
        hidden_chunked, labels, head_chunked, chunk_size=3
    )
    assert torch.allclose(full_loss, chunked_loss, atol=1e-6, rtol=1e-6)
    full_loss.backward()
    chunked_loss.backward()
    assert torch.allclose(hidden_full.grad, hidden_chunked.grad, atol=2e-6, rtol=2e-6)
    assert torch.allclose(
        head_full.weight.grad, head_chunked.weight.grad, atol=2e-6, rtol=2e-6
    )


def test_synthesis_wrapper_supports_exact_cyclic_scaling_offsets() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "train" / "jobs" / "q36_mtr_synthesize_trajectories.sbatch"
    ).read_text(encoding="utf-8")
    assert '[[ "$rotation" =~ ^[012]$ ]]' in source
    assert "SYNTHESIS_SCRIPT_SHA256" in source
    assert '--rotation-offset "$rotation"' in source
