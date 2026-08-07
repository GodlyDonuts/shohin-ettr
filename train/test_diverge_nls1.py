#!/usr/bin/env python3
"""Focused data and mechanics tests for DIVERGE-NLS1."""

from __future__ import annotations

import torch

from diverge_eal2_data import build_evaluation_episode
from diverge_nls1_data import DEVELOPMENT_SEED, build_training_record
from diverge_nls1_runtime import (
    NeuralLawSynthesizer,
    compile_neural_laws,
    episode_demonstrations,
    tensorize_law_rows,
)


def main() -> None:
    rows = [build_training_record(index) for index in range(8)]
    values, mask, targets = tensorize_law_rows(rows, torch.device("cpu"))
    model = NeuralLawSynthesizer()
    logits = model(values, mask)
    assert logits.shape == (8, 2, 25)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
    loss.backward()
    assert all(
        parameter.grad is not None and torch.all(torch.isfinite(parameter.grad))
        for parameter in model.parameters()
    )

    public, assessor = build_evaluation_episode(0, seed=DEVELOPMENT_SEED)
    temporal = [
        tuple(int(value) // 2 for value in item["numeric_role_ids"])
        for item in assessor["evidence"]
    ]
    grouped = episode_demonstrations(public, temporal, text_key="source_text")
    assert len(grouped) == 8 and all(len(value) == 3 for value in grouped)
    for demonstrations in grouped:
        reversed_values = torch.tensor(
            [list(reversed(demonstrations))], dtype=torch.long
        )
        forward_values = torch.tensor([demonstrations], dtype=torch.long)
        complete_mask = torch.ones((1, 3), dtype=torch.bool)
        torch.testing.assert_close(
            model(forward_values, complete_mask),
            model(reversed_values, complete_mask),
            rtol=1e-6,
            atol=1e-6,
        )
    compilation = compile_neural_laws(
        public,
        temporal,
        model,
        device=torch.device("cpu"),
        reader_state_sha256="1" * 64,
    )
    assert compilation.packet is not None and compilation.error is None
    assert compilation.evidence_count == 24
    one = compile_neural_laws(
        public,
        temporal,
        model,
        device=torch.device("cpu"),
        reader_state_sha256="1" * 64,
        control="one_example",
    )
    assert one.packet is not None
    print("diverge NLS1 mechanics tests passed")


if __name__ == "__main__":
    main()
