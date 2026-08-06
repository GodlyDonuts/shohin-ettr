from __future__ import annotations

from argparse import Namespace

import pytest
import torch

from diverge_vcr1_data import CorrectionTokens
from eval_diverge_vcr1 import (
    DRAFT_SCHEMA,
    VCR1EvalError,
    _left_pad_corrections,
    _ordered_identity_sha256,
    _validate_draft_payload,
)


def _draft(identity: str, *, task: str = "math500") -> dict[str, object]:
    return {
        "task": task,
        "identity_sha256": identity,
        "task_prompt": "problem",
        "source_completion": "draft",
        "source_correct": False,
    }


def test_left_padding_preserves_segment_alignment() -> None:
    rows = [
        CorrectionTokens([1, 2], [], [True, False], [False, True]),
        CorrectionTokens(
            [3, 4, 5, 6],
            [],
            [True, True, False, False],
            [False, False, True, True],
        ),
    ]
    ids, active, question, draft = _left_pad_corrections(rows, 0, torch.device("cpu"))
    assert ids.tolist() == [[0, 0, 1, 2], [3, 4, 5, 6]]
    assert active.tolist() == [[0, 0, 1, 1], [1, 1, 1, 1]]
    assert question.tolist() == [
        [False, False, True, False],
        [True, True, False, False],
    ]
    assert draft.tolist() == [
        [False, False, False, True],
        [False, False, True, True],
    ]


def test_draft_payload_provenance_and_identity_are_bound() -> None:
    rows = [_draft("a" * 64), _draft("b" * 64)]
    payload = {
        "schema": DRAFT_SCHEMA,
        "status": "complete",
        "task": "math500",
        "model_revision": "revision",
        "source_checkpoint_sha256": "c" * 64,
        "count": 2,
        "rows": rows,
    }
    args = Namespace(
        task="math500",
        model_revision="revision",
        source_checkpoint_sha256="c" * 64,
    )
    assert _validate_draft_payload(payload, args) == rows
    assert _ordered_identity_sha256(rows) == _ordered_identity_sha256(rows)

    payload["rows"] = [rows[0], rows[0]]
    with pytest.raises(VCR1EvalError, match="not unique"):
        _validate_draft_payload(payload, args)


def test_draft_payload_rejects_source_drift() -> None:
    row = _draft("a" * 64)
    payload = {
        "schema": DRAFT_SCHEMA,
        "status": "complete",
        "task": "math500",
        "model_revision": "wrong",
        "source_checkpoint_sha256": "c" * 64,
        "count": 1,
        "rows": [row],
    }
    args = Namespace(
        task="math500",
        model_revision="revision",
        source_checkpoint_sha256="c" * 64,
    )
    with pytest.raises(VCR1EvalError, match="provenance"):
        _validate_draft_payload(payload, args)
