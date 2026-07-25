from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from source_deleted_variable_topology_board import generate_episode  # noqa: E402
from multifamily_raw_machine_compiler import (  # noqa: E402
    CompilerOutput,
    MultiFamilyCompilerError,
    QueryOutput,
    execute_query,
    seal_machine,
)
from run_variable_topology_renderer_curriculum import (  # noqa: E402
    _base_example,
    _collate,
)


@pytest.mark.parametrize("cardinality,action_count", ((4, 2), (8, 4)))
def test_oracle_roles_execute_incidence_collision(
    cardinality: int,
    action_count: int,
) -> None:
    row = generate_episode(
        seed=88,
        split="development",
        family="permutation",
        renderer=5,
        cell="joint",
        cardinality=cardinality,
        action_count=action_count,
    )
    example = _base_example(row)
    source, query, source_labels, query_labels = _collate(
        [example],
        device=torch.device("cpu"),
    )
    source_logits = torch.nn.functional.one_hot(
        source_labels.clamp_min(0),
        num_classes=3,
    ).to(torch.float32)
    source_logits[source_labels.eq(-100)] = 0
    query_logits = torch.nn.functional.one_hot(
        query_labels.clamp_min(0),
        num_classes=2,
    ).to(torch.float32)
    query_logits[query_labels.eq(-100)] = 0

    with pytest.raises(
        MultiFamilyCompilerError,
        match="incidence-ambiguous",
    ):
        seal_machine(
            source,
            CompilerOutput(source_logits),
            row=0,
            structural_key_classes=True,
        )

    machine = seal_machine(
        source,
        CompilerOutput(source_logits),
        row=0,
        structural_key_classes=True,
        incidence_ambiguous_fallback=True,
    )
    answer = execute_query(
        machine,
        query,
        QueryOutput(query_logits),
        row=0,
        structural_key_classes=True,
    )
    assert answer.decode("ascii") == row.supervisor.answer
