#!/usr/bin/env python3
"""Focused data and mechanics tests for DIVERGE-OPB1."""

from __future__ import annotations

from pathlib import Path

import torch

from diverge_nls1_runtime import NeuralLawSynthesizer
from diverge_opb1_data import (
    DEVELOPMENT_SEED,
    augment_evaluation_episode,
    build_training_record,
    rotate_aliases,
)
from diverge_opb1_runtime import (
    EvidenceOperationPointer,
    compile_pointer_event_laws,
    tensorize_operation_sources,
)


def main() -> None:
    rows = [build_training_record(index) for index in range(4)]
    model = EvidenceOperationPointer()
    tensors = tensorize_operation_sources(rows, torch.device("cpu"))
    logits = model(*tensors)
    assert logits.shape == (4, 8)
    targets = torch.tensor([row["operation_target"] for row in rows])
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None and torch.all(torch.isfinite(parameter.grad))
        for parameter in model.parameters()
    )

    rotated = [{**row, "aliases": list(rotate_aliases(row["aliases"]))} for row in rows]
    rotated_logits = model(*tensorize_operation_sources(rotated, torch.device("cpu")))
    assert torch.allclose(rotated_logits, torch.roll(logits.detach(), -1, dims=1))

    public, assessor = augment_evaluation_episode(0, seed=DEVELOPMENT_SEED)
    assert len(public["evidence"]) == len(assessor["operation_targets"]) == 24
    assert all("operation" not in item for item in public["evidence"])
    assert all(
        item["fully_renamed_source_text"] != item["source_text"]
        for item in public["evidence"]
    )

    synthetic_events = []
    operations = []
    commitments = []
    for operation in range(8):
        for demonstration in range(3):
            synthetic_events.append(
                tuple(
                    role * 97 + (operation + demonstration + role) % 97
                    for role in range(4)
                )
            )
            operations.append(operation)
            commitments.append(f"commitment-{operation}-{demonstration}")
    law_model = NeuralLawSynthesizer()
    compiled = compile_pointer_event_laws(
        public["aliases"],
        commitments,
        synthetic_events,
        operations,
        law_model,
        device=torch.device("cpu"),
        event_owner_sha256="event",
        pointer_owner_sha256="pointer",
        law_owner_sha256="law",
    )
    assert compiled.packet is not None
    broken = compile_pointer_event_laws(
        public["aliases"],
        commitments,
        synthetic_events,
        [0] * len(operations),
        law_model,
        device=torch.device("cpu"),
        event_owner_sha256="event",
        pointer_owner_sha256="pointer",
        law_owner_sha256="law",
    )
    assert broken.packet is None and broken.error == "operation_not_complete"

    runtime = Path(__file__).with_name("diverge_opb1_runtime.py").read_text()
    assert "re.search" not in runtime and "aliases.index" not in runtime
    print("DIVERGE-OPB1 tests passed")


if __name__ == "__main__":
    main()
