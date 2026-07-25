from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train"
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))

from source_deleted_multifamily_machine_board import generate_episode  # noqa: E402
from multifamily_raw_machine_compiler import (  # noqa: E402
    QUERY_ACTION,
    QUERY_START,
    ROLE_ACTION,
    ROLE_SOURCE,
    ROLE_TARGET,
    collate_queries,
    collate_sources,
    scan_query,
    scan_source,
)
from multifamily_machine_process_custody import run_process_custody  # noqa: E402


def test_distinct_source_and_query_processes_execute_exactly() -> None:
    episode = generate_episode(
        seed=100,
        split="development",
        family="bitwise_rotate_xor",
        renderer=3,
        cell="joint",
    )
    source = episode.candidate.source.encode("ascii")
    query = episode.candidate.query.encode("ascii")
    source_batch = collate_sources((scan_source(source),))
    query_batch = collate_queries((scan_query(query),))
    source_logits = torch.full((1, 48, 3, 3), -20.0)
    source_order = (ROLE_TARGET, ROLE_ACTION, ROLE_SOURCE)
    for record in range(int(source_batch.record_valid[0].sum())):
        for occurrence, role in enumerate(source_order):
            source_logits[0, record, occurrence, role] = 20.0
    query_logits = torch.full((1, 9, 2), -20.0)
    query_count = int(query_batch.occurrence_valid[0].sum())
    query_order = (QUERY_ACTION,) * (query_count - 1) + (QUERY_START,)
    for occurrence, role in enumerate(query_order):
        query_logits[0, occurrence, role] = 20.0

    receipt = run_process_custody(
        source=source,
        source_role_logits=source_logits.tolist(),
        query=query,
        query_role_logits=query_logits.tolist(),
    )
    assert receipt.producer.pid != receipt.consumer.pid
    assert receipt.producer.wire_sha256 == receipt.consumer.wire_sha256
    assert receipt.consumer.answer.decode("ascii") == episode.supervisor.answer
    assert receipt.producer.source_sha256 not in receipt.consumer.wire_sha256
