#!/usr/bin/env python3
"""Pre-score mechanics tests for DIVERGE-SVE1."""

from __future__ import annotations

from pathlib import Path

import torch

from diverge_eal1_runtime import execute_program
from diverge_sve1_data import (
    DEVELOPMENT_SEED,
    augment_evaluation_episode,
    build_training_record,
    validate_evaluation_episode,
    validate_training_record,
)
from diverge_sve1_runtime import (
    SpanlessValueEventTransducer,
    compile_event_laws,
    decode_initial_events,
    digit_scrub,
    tensorize_event_sources,
)
from eval_diverge_sve1 import _gold_evidence_events, _gold_initial_events


def main() -> None:
    left = build_training_record(17)
    right = build_training_record(18)
    validate_training_record(left)
    assert left == build_training_record(17)
    assert left["identity_sha256"] != right["identity_sha256"]
    assert len(left["evidence_event_targets"]) == 4
    assert len(left["initial_event_targets"]) == 2
    assert sorted(value // 97 for value in left["evidence_event_targets"]) == [
        0,
        1,
        2,
        3,
    ]
    assert decode_initial_events(left["initial_event_targets"])

    device = torch.device("cpu")
    model = SpanlessValueEventTransducer()
    for kind, key, expected in (
        ("evidence", "evidence_text", (2, 2)),
        ("initial", "initial_text", (1, 1)),
    ):
        tensors = tensorize_event_sources(
            [left, right], device, text_key=key, expected_occurrences=expected
        )
        logits = model(tensors[0], tensors[1], kind=kind)  # type: ignore[arg-type]
        assert logits.shape[:2] == tensors[0].shape
        assert bool(tensors[3].all())
        logits.mean().backward()
        model.zero_grad(set_to_none=True)
    scrubbed = digit_scrub(str(left["evidence_text"]))
    assert not any("0" <= value <= "9" for value in scrubbed)

    public, assessor = augment_evaluation_episode(0, seed=DEVELOPMENT_SEED)
    validate_evaluation_episode(public, assessor)
    events = _gold_evidence_events(
        [public],
        [assessor],
        table_key="register_table",
        canonical_key="canonical_registers",
        reverse_table=False,
    )
    compilation = compile_event_laws(
        {"aliases": public["aliases"], "evidence": public["evidence"]},
        events,
        owner_state_sha256="0" * 64,
        text_key="source_text",
        hash_key="source_sha256",
    )
    assert compilation.packet is not None
    initial_events = _gold_initial_events(
        [public],
        [assessor],
        table_key="register_table",
        canonical_key="canonical_registers",
        reverse_table=False,
    )
    for visible, hidden, encoded in zip(
        public["transfer"], assessor["transfer"], initial_events, strict=True
    ):
        prediction = execute_program(
            compilation.packet,
            {
                "depth": visible["depth"],
                "initial_state": list(decode_initial_events(encoded)),
                "symbols": [
                    public["aliases"][value] for value in hidden["symbol_indices"]
                ],
            },
        )
        canonical = tuple(assessor["canonical_registers"])
        table = tuple(public["register_table"])
        restored = [0, 0]
        for position, value in enumerate(prediction):
            restored[canonical.index(table[position])] = value
        assert tuple(restored) == tuple(hidden["terminal_state"])

    runtime = Path(__file__).with_name("diverge_sve1_runtime.py").read_text()
    assert "scan_integer_spans" not in runtime
    assert "int(text[" not in runtime
    print("DIVERGE-SVE1 tests passed")


if __name__ == "__main__":
    main()
