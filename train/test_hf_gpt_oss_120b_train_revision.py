from __future__ import annotations

import ast
from pathlib import Path

from hf_gpt_oss_120b_train_revision import (
    CONSUMED_PRESENTATIONS,
    DATA_PRESENTATIONS,
    GRADIENT_ACCUMULATION,
    MAX_SEQUENCE_LENGTH,
    UPDATES,
    consumed_identity_sha256,
)


def test_training_geometry_matches_the_cross_family_mixtral_point() -> None:
    assert DATA_PRESENTATIONS == 9_655
    assert UPDATES == 256
    assert GRADIENT_ACCUMULATION == 8
    assert CONSUMED_PRESENTATIONS == 2_048
    assert MAX_SEQUENCE_LENGTH == 4_096


def test_consumed_identity_digest_is_order_sensitive() -> None:
    rows = [{"identity_sha256": f"{index:064x}"} for index in range(DATA_PRESENTATIONS)]
    left = consumed_identity_sha256(rows)
    rows[0], rows[1] = rows[1], rows[0]
    assert consumed_identity_sha256(rows) != left


def test_training_uses_harmony_final_targets_and_bounded_logits() -> None:
    path = Path(__file__).with_name("hf_gpt_oss_120b_train_revision.py")
    source = path.read_text()
    tree = ast.parse(source)
    assert "tokenize_training_example" in source
    assert "logits_to_keep=len(response) + 1" in source
    assert 'device_map={"": 0}' in source
    assert 'native_router_expert_trainables": 0' in source
    assert any(isinstance(node, ast.Call) for node in ast.walk(tree))
