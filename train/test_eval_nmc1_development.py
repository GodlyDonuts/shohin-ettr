import pytest
import torch

from eval_nmc1_development import (
    NMC1EvaluationError,
    extract_program,
    load_checkpoint_receipt,
    source_shuffle,
)
from natural_microcode_program import execute_fraction


def test_extracts_program_from_completion() -> None:
    completion = "prefix\n<MICROCODE_V1>\nR0 P:4 P:5 A\nC:0\n</MICROCODE_V1>\nsuffix"
    assert execute_fraction(extract_program(completion)) == 9


def test_source_shuffle_preserves_depth_and_changes_identity() -> None:
    rows = [
        {"identity_sha256": f"{index:064x}", "register_depth": depth}
        for depth in (1, 2)
        for index in range(depth * 10, depth * 10 + 3)
    ]
    mapping = source_shuffle(rows)
    by_identity = {row["identity_sha256"]: row for row in rows}
    assert set(mapping) == set(by_identity)
    for identity, donor in mapping.items():
        assert identity != donor["identity_sha256"]
        assert by_identity[identity]["register_depth"] == donor["register_depth"]


def test_source_shuffle_rejects_singleton_depth() -> None:
    with pytest.raises(ValueError, match="singleton"):
        source_shuffle([{"identity_sha256": "a" * 64, "register_depth": 1}])


def test_checkpoint_update_is_read_from_top_level(tmp_path) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "schema": "shohin-hf-product-reasoning-checkpoint-v1",
            "update": 1024,
            "metadata": {"model_revision": "revision"},
        },
        path,
    )
    assert load_checkpoint_receipt(path) == (
        1024,
        {"model_revision": "revision"},
    )


def test_checkpoint_receipt_rejects_wrong_schema(tmp_path) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save({"schema": "wrong", "update": 1024, "metadata": {}}, path)
    with pytest.raises(NMC1EvaluationError, match="schema"):
        load_checkpoint_receipt(path)
