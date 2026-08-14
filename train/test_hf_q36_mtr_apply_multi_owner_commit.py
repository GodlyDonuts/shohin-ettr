from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import hf_q36_mtr_apply_multi_owner_commit as module
from hf_q36_mtr_apply_multi_owner_commit import (
    CANDIDATE_SCHEMA,
    DEVELOPMENT_ROWS,
    Q36MTRMultiOwnerError,
    SOURCE_SCHEMA,
    choose_owner,
    load_development_candidates,
    load_development_source,
    make_commit_head,
)


def _identity(index: int) -> str:
    return f"{index:064x}"


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _candidate_files(root: Path) -> list[Path]:
    paths = [root / f"shard_{index:02d}.jsonl" for index in range(16)]
    rows: list[list[dict[str, object]]] = [[] for _ in paths]
    for index in range(DEVELOPMENT_ROWS):
        rows[index % len(paths)].append(
            {
                "schema": CANDIDATE_SCHEMA,
                "identity_sha256": _identity(index),
                "split": "development",
                "task": ("math500", "bbh_logic", "mbpp")[index % 3],
                "completion": f"candidate {index}",
                "generated_tokens": 4,
                "max_token_exhausted": False,
            }
        )
    for path, shard in zip(paths, rows, strict=True):
        _write(path, shard)
    return paths


def test_choose_owner_is_argmax_with_stable_tie_break() -> None:
    assert choose_owner([0.1, 0.4, 0.2]) == 1
    assert choose_owner([0.4, 0.4, 0.2]) == 0
    with pytest.raises(Q36MTRMultiOwnerError):
        choose_owner([0.1, float("nan"), 0.2])


def test_load_development_candidates_requires_exact_sharded_coverage(
    tmp_path: Path,
) -> None:
    paths = _candidate_files(tmp_path)
    rows = load_development_candidates(paths)
    assert len(rows) == DEVELOPMENT_ROWS
    assert rows[_identity(7)]["completion"] == "candidate 7"
    with pytest.raises(Q36MTRMultiOwnerError):
        load_development_candidates(paths[:-1])


def test_load_development_source_is_label_free_and_exact(tmp_path: Path) -> None:
    path = tmp_path / "development.jsonl"
    rows = [
        {
            "schema": SOURCE_SCHEMA,
            "identity_sha256": _identity(index),
            "split": "development",
            "task": ("math500", "bbh_logic", "mbpp")[index % 3],
            "source_prompt": f"problem {index}",
        }
        for index in range(DEVELOPMENT_ROWS)
    ]
    _write(path, rows)
    assert len(load_development_source(path)) == DEVELOPMENT_ROWS
    rows[0]["answer"] = "leak"
    _write(tmp_path / "leaked.jsonl", rows)
    with pytest.raises(Q36MTRMultiOwnerError):
        load_development_source(tmp_path / "leaked.jsonl")


def test_restores_setwise_head_with_exact_adapter_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.nn.Module, "to", lambda self, *_args, **_kwargs: self)
    expected = module.SetwiseCommitHead(8, module.HEAD_WIDTH, module.SETWISE_PROJECTION)
    payload = {
        "schema": module.SETWISE_MODEL_SCHEMA,
        "metadata": {
            "model_revision": module.MODEL_REVISION,
            "head_width": module.HEAD_WIDTH,
            "projection": module.SETWISE_PROJECTION,
            "projection_contract": module.SETWISE_PROJECTION_CONTRACT,
            "permutation_equivariant": True,
            "backbone_frozen": True,
            "adapter_checkpoint_sha256": "a" * 64,
        },
        "head_state": expected.state_dict(),
    }
    restored, contract = make_commit_head(
        payload,
        head_type="setwise",
        hidden_size=8,
        adapter_checkpoint_sha256="a" * 64,
    )
    assert isinstance(restored, module.SetwiseCommitHead)
    assert contract == module.SETWISE_PROJECTION_CONTRACT
    with pytest.raises(Q36MTRMultiOwnerError, match="setwise head"):
        make_commit_head(
            payload,
            head_type="setwise",
            hidden_size=8,
            adapter_checkpoint_sha256="b" * 64,
        )
