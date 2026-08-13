"""Frozen role and causal-visibility contract for Q36-MTR.

This module contains no scheduler or model-acquisition capability.  It is the
small, importable boundary shared by the prospective Q36 trainer, generator,
mechanics gate, and dry-run graph compiler.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
MODEL_CONFIG_SHA256 = "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
MODEL_MANIFEST_SHA256 = (
    "06c9d8d8419244f2d001cb351e164f356718d9d77138e898b13afee35856f56e"
)
ARCHITECTURE = "shohin-q36-mtr-shared-post-mlp-v1"
HIDDEN_SIZE = 2048
CONTROLLED_LAYERS = 16
RANK = 18
ALPHA = 18.0
TRAINABLE_PARAMETERS = 1_179_648
QUANTIZATION = "nf4"
COMPUTE_DTYPE = "bfloat16"
OWNER_UPDATES = 256
REVISION_UPDATES = 256
OWNER_MAX_ROWS = 100_000
REVISION_PRESENTATIONS = 9_655
OWNER_MAX_SEQUENCE_LENGTH = 1_024
REVISION_MAX_SEQUENCE_LENGTH = 4_096
OWNER_LEARNING_RATE = 2e-4
REVISION_LEARNING_RATE = 2e-5
OWNER_GRADIENT_ACCUMULATION = 16
REVISION_GRADIENT_ACCUMULATION = 8
OWNER_SEED = 2026080711
OWNER_DATA_SEED = 20260802
REVISION_SEED = 2026080815
REVISION_DATA_SEED = 2026080814
DRAFT_SEED = 2026080818
DRAFT_SHARDS = 16
DRAFT_IDENTITIES = 7_113
DRAFT_MAX_NEW_TOKENS = 768


class Q36MTRRoleError(RuntimeError):
    """The prospective Q36-MTR role contract differs."""


@dataclass(frozen=True)
class RoleSpec:
    name: str
    data_kind: str
    draft_control: str
    warm_start_role: str | None
    updates: int
    max_rows: int
    max_sequence_length: int
    learning_rate: float
    gradient_accumulation: int
    seed: int
    data_seed: int


ROLE_SPECS = {
    "owner": RoleSpec(
        name="owner",
        data_kind="source_only",
        draft_control="normal",
        warm_start_role=None,
        updates=OWNER_UPDATES,
        max_rows=OWNER_MAX_ROWS,
        max_sequence_length=OWNER_MAX_SEQUENCE_LENGTH,
        learning_rate=OWNER_LEARNING_RATE,
        gradient_accumulation=OWNER_GRADIENT_ACCUMULATION,
        seed=OWNER_SEED,
        data_seed=OWNER_DATA_SEED,
    ),
    "aligned": RoleSpec(
        name="aligned",
        data_kind="natural_trajectory_revision",
        draft_control="normal",
        warm_start_role="owner",
        updates=REVISION_UPDATES,
        max_rows=REVISION_PRESENTATIONS,
        max_sequence_length=REVISION_MAX_SEQUENCE_LENGTH,
        learning_rate=REVISION_LEARNING_RATE,
        gradient_accumulation=REVISION_GRADIENT_ACCUMULATION,
        seed=REVISION_SEED,
        data_seed=REVISION_DATA_SEED,
    ),
    "draft_hidden": RoleSpec(
        name="draft_hidden",
        data_kind="natural_trajectory_revision",
        draft_control="draft_unavailable",
        warm_start_role="owner",
        updates=REVISION_UPDATES,
        max_rows=REVISION_PRESENTATIONS,
        max_sequence_length=REVISION_MAX_SEQUENCE_LENGTH,
        learning_rate=REVISION_LEARNING_RATE,
        gradient_accumulation=REVISION_GRADIENT_ACCUMULATION,
        seed=REVISION_SEED,
        data_seed=REVISION_DATA_SEED,
    ),
}


def role_spec(role: str) -> RoleSpec:
    try:
        return ROLE_SPECS[role]
    except KeyError as error:
        raise Q36MTRRoleError(f"unknown Q36-MTR role: {role}") from error


def trainable_name_sha256(names: Iterable[str]) -> str:
    ordered = sorted(names)
    if not ordered or len(ordered) != len(set(ordered)):
        raise Q36MTRRoleError("Q36-MTR trainable parameter names differ")
    return hashlib.sha256("\n".join(ordered).encode()).hexdigest()


def role_contract(role: str) -> dict[str, Any]:
    spec = role_spec(role)
    return {
        "architecture": ARCHITECTURE,
        "role": role,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "hidden_size": HIDDEN_SIZE,
        "controlled_layers": CONTROLLED_LAYERS,
        "rank": RANK,
        "alpha": ALPHA,
        "trainable_parameters": TRAINABLE_PARAMETERS,
        "quantization": QUANTIZATION,
        "compute_dtype": COMPUTE_DTYPE,
        "role_spec": asdict(spec),
        "router_expert_trainables": 0,
        "token_geometry": "identical_aligned_and_draft_hidden",
        "position_geometry": "explicit_full_sequence_positions",
        "hidden_intervention": "draft_attention_only_masked_not_deleted",
        "external_proposer": False,
        "task_router": False,
    }


def validate_contract(payload: Mapping[str, Any], role: str) -> None:
    expected = role_contract(role)
    observed = {key: payload.get(key) for key in expected}
    if observed != expected:
        raise Q36MTRRoleError(
            "Q36-MTR role contract differs: "
            f"expected={json.dumps(expected, sort_keys=True)} "
            f"observed={json.dumps(observed, sort_keys=True)}"
        )


def validate_owner_warm_start(
    metadata: Mapping[str, Any],
    *,
    checkpoint_update: int,
    trainable_parameters: int,
    trainable_parameter_name_sha256: str,
) -> None:
    """Require both revisers to start from the exact source-only owner state."""

    validate_contract(metadata, "owner")
    expected = {
        "update": OWNER_UPDATES,
        "trainable_parameters": TRAINABLE_PARAMETERS,
        "trainable_parameter_name_sha256": trainable_parameter_name_sha256,
        "selected_rows": OWNER_MAX_ROWS,
        "source_only_model_visible": True,
        "internal_draft_visible": False,
    }
    observed = {
        "update": checkpoint_update,
        "trainable_parameters": trainable_parameters,
        "trainable_parameter_name_sha256": metadata.get(
            "trainable_parameter_name_sha256"
        ),
        "selected_rows": metadata.get("selected_rows"),
        "source_only_model_visible": metadata.get("source_only_model_visible"),
        "internal_draft_visible": metadata.get("internal_draft_visible"),
    }
    if observed != expected:
        raise Q36MTRRoleError(
            "Q36-MTR owner warm start differs: "
            f"expected={expected} observed={observed}"
        )


def sequence_geometry_receipt(
    prompt_rows: list[list[int]],
    response_rows: list[list[int]],
    draft_attention_rows: list[list[int]],
) -> dict[str, Any]:
    """Hash exact token/position geometry independently of visibility control."""

    if not prompt_rows or not (
        len(prompt_rows) == len(response_rows) == len(draft_attention_rows)
    ):
        raise Q36MTRRoleError("Q36-MTR sequence batch geometry differs")
    rows: list[dict[str, Any]] = []
    masked_tokens = 0
    for prompt, response, mask in zip(
        prompt_rows, response_rows, draft_attention_rows, strict=True
    ):
        if (
            not prompt
            or not response
            or len(prompt) != len(mask)
            or any(value not in (0, 1) for value in mask)
        ):
            raise Q36MTRRoleError("Q36-MTR sequence row geometry differs")
        masked = sum(value == 0 for value in mask)
        masked_tokens += masked
        rows.append(
            {
                "prompt": prompt,
                "response": response,
                # Positions are always based on the full unmasked sequence.
                "position_ids": list(range(len(prompt) + len(response))),
                "draft_mask": mask,
            }
        )
    token_preimage = b"".join(
        (
            json.dumps({"prompt": row["prompt"], "response": row["response"]}) + "\n"
        ).encode()
        for row in rows
    )
    position_preimage = b"".join(
        (json.dumps(row["position_ids"]) + "\n").encode() for row in rows
    )
    mask_preimage = b"".join(
        (json.dumps(row["draft_mask"]) + "\n").encode() for row in rows
    )
    return {
        "rows": len(rows),
        "prompt_tokens": sum(len(row["prompt"]) for row in rows),
        "response_tokens": sum(len(row["response"]) for row in rows),
        "draft_masked_tokens": masked_tokens,
        "token_geometry_sha256": hashlib.sha256(token_preimage).hexdigest(),
        "position_geometry_sha256": hashlib.sha256(position_preimage).hexdigest(),
        "draft_attention_sha256": hashlib.sha256(mask_preimage).hexdigest(),
    }


def validate_matched_revision_geometry(
    aligned: Mapping[str, Any], draft_hidden: Mapping[str, Any]
) -> None:
    """Prove the two revisers differ only in draft visibility."""

    stable = (
        "rows",
        "prompt_tokens",
        "response_tokens",
        "token_geometry_sha256",
        "position_geometry_sha256",
    )
    if any(aligned.get(key) != draft_hidden.get(key) for key in stable):
        raise Q36MTRRoleError("Q36-MTR aligned/hidden token geometry differs")
    if aligned.get("draft_masked_tokens") != draft_hidden.get("draft_masked_tokens"):
        raise Q36MTRRoleError("Q36-MTR aligned/hidden draft span differs")
    if (
        not isinstance(aligned.get("draft_masked_tokens"), int)
        or aligned["draft_masked_tokens"] <= 0
    ):
        raise Q36MTRRoleError("Q36-MTR draft span is empty")
