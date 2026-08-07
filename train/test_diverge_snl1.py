#!/usr/bin/env python3
"""Mechanics tests for DIVERGE-SNL1 spanless neural-law composition."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from build_diverge_snl1_data import DEVELOPMENT_SEED
from diverge_eal1_runtime import EpisodeLawPacket
from diverge_mze1_runtime import ROW_CANDIDATES
from diverge_snl1_runtime import compile_neural_event_laws
from diverge_sve1_data import augment_evaluation_episode
from eval_diverge_snl1 import _law_score
from eval_diverge_sve1 import _gold_evidence_events


class _OracleLawModel(nn.Module):
    def __init__(self, matrices: list[list[list[int]]]) -> None:
        super().__init__()
        indices = [
            [ROW_CANDIDATES.index(tuple(row)) for row in matrix] for matrix in matrices
        ]
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        del mask
        logits = torch.full(
            (values.shape[0], 2, len(ROW_CANDIDATES)),
            -100.0,
            device=values.device,
        )
        return logits.scatter(2, self.indices.to(values.device).unsqueeze(-1), 100.0)


def main() -> None:
    swapped_packet = EpisodeLawPacket(
        aliases=("operation",),
        rows=(((4, 3), (2, 1)),),
        evidence_commitments=(),
        reader_state_sha256="0" * 64,
        commitment="",
    )
    swapped_score = _law_score(
        [swapped_packet],
        [{"register_table": ["second", "first"]}],
        [{"canonical_registers": ["first", "second"], "matrices": [[[1, 2], [3, 4]]]}],
        table_key="register_table",
        canonical_key="canonical_registers",
        reverse_table=False,
    )
    if swapped_score["exact_rate"] != 1.0 or swapped_score["row_rate"] != 1.0:
        raise RuntimeError("SNL1 law scorer did not conjugate into table basis")

    public, assessor = augment_evaluation_episode(0, seed=DEVELOPMENT_SEED)
    events = _gold_evidence_events(
        [public],
        [assessor],
        table_key="register_table",
        canonical_key="canonical_registers",
        reverse_table=False,
    )
    model = _OracleLawModel(assessor["matrices"])
    compilation = compile_neural_event_laws(
        {"aliases": public["aliases"], "evidence": public["evidence"]},
        events,
        model,  # type: ignore[arg-type]
        device=torch.device("cpu"),
        event_owner_sha256="1" * 64,
        model_owner_sha256="2" * 64,
        text_key="source_text",
        hash_key="source_sha256",
    )
    expected = tuple(
        tuple(tuple(int(value) for value in row) for row in matrix)
        for matrix in assessor["matrices"]
    )
    if compilation.packet is None or compilation.packet.rows != expected:
        raise RuntimeError("SNL1 oracle packet differs")
    incomplete = compile_neural_event_laws(
        {"aliases": public["aliases"], "evidence": public["evidence"]},
        [*events[:-1], ()],
        model,  # type: ignore[arg-type]
        device=torch.device("cpu"),
        event_owner_sha256="1" * 64,
        model_owner_sha256="2" * 64,
        text_key="source_text",
        hash_key="source_sha256",
    )
    if incomplete.packet is not None or incomplete.error != "event_not_complete":
        raise RuntimeError("SNL1 incomplete event did not fail closed")
    source = Path(__file__).with_name("diverge_snl1_runtime.py").read_text()
    if any(
        token in source
        for token in ("compile_event_laws", "scan_integer_spans", "set(range")
    ):
        raise RuntimeError("SNL1 runtime contains an exact support/value parser")
    print("DIVERGE-SNL1 tests passed")


if __name__ == "__main__":
    main()
