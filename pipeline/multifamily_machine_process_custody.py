"""Two-process source-deletion custody for the multi-family machine.

The producer receives source bytes and model-produced source-role logits,
emits one canonical sealed wire, and exits. Only after successful producer
exit may a distinct consumer receive the wire, late query, and model-produced
query-role logits. Neither worker receives supervisor labels or an exact
parser.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train"
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))

from multifamily_raw_machine_compiler import (  # noqa: E402
    CompilerOutput,
    QueryOutput,
    SealedAnonymousMachine,
    collate_queries,
    collate_sources,
    execute_query,
    scan_query,
    scan_source,
    seal_machine,
)


class MultiFamilyProcessError(RuntimeError):
    """Raised when source and query process custody is violated."""


@dataclass(frozen=True, slots=True)
class ProducerReceipt:
    pid: int
    source_sha256: str
    wire_sha256: str


@dataclass(frozen=True, slots=True)
class ConsumerReceipt:
    pid: int
    query_sha256: str
    wire_sha256: str
    answer: bytes


@dataclass(frozen=True, slots=True)
class ProcessCustodyReceipt:
    producer: ProducerReceipt
    consumer: ConsumerReceipt

    def __post_init__(self) -> None:
        if (
            self.producer.pid == self.consumer.pid
            or self.producer.wire_sha256 != self.consumer.wire_sha256
        ):
            raise MultiFamilyProcessError("process custody receipt differs")


def _producer_worker(
    connection,
    source: bytes,
    source_role_logits: Sequence,
) -> None:
    try:
        batch = collate_sources((scan_source(source),))
        logits = torch.tensor(source_role_logits, dtype=torch.float32)
        machine = seal_machine(
            batch,
            CompilerOutput(source_role_logits=logits),
            row=0,
        )
        wire = machine.deployed_wire()
        connection.send(
            {
                "pid": mp.current_process().pid,
                "source_sha256": sha256(source).hexdigest(),
                "wire": wire,
                "wire_sha256": sha256(wire).hexdigest(),
            }
        )
    except Exception as exc:  # pragma: no cover - forwarded fail-closed.
        connection.send({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def _consumer_worker(
    connection,
    wire: bytes,
    query: bytes,
    query_role_logits: Sequence,
) -> None:
    try:
        machine = SealedAnonymousMachine.from_deployed_wire(wire)
        batch = collate_queries((scan_query(query),))
        logits = torch.tensor(query_role_logits, dtype=torch.float32)
        answer = execute_query(
            machine,
            batch,
            QueryOutput(query_role_logits=logits),
            row=0,
        )
        connection.send(
            {
                "answer": answer,
                "pid": mp.current_process().pid,
                "query_sha256": sha256(query).hexdigest(),
                "wire_sha256": sha256(wire).hexdigest(),
            }
        )
    except Exception as exc:  # pragma: no cover - forwarded fail-closed.
        connection.send({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def _receive(connection, process: mp.Process, label: str) -> dict:
    if not connection.poll(30):
        process.terminate()
        process.join(10)
        raise MultiFamilyProcessError(f"{label} worker timed out")
    message = connection.recv()
    process.join(30)
    if process.exitcode != 0 or "error" in message:
        raise MultiFamilyProcessError(
            f"{label} worker failed: {message.get('error', process.exitcode)}"
        )
    return message


def compile_in_fresh_process(
    *,
    source: bytes,
    source_role_logits: Sequence,
) -> tuple[bytes, ProducerReceipt]:
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_producer_worker,
        args=(child, source, source_role_logits),
    )
    process.start()
    child.close()
    message = _receive(parent, process, "producer")
    parent.close()
    wire = message["wire"]
    return wire, ProducerReceipt(
        pid=int(message["pid"]),
        source_sha256=message["source_sha256"],
        wire_sha256=message["wire_sha256"],
    )


def execute_in_fresh_process(
    *,
    wire: bytes,
    query: bytes,
    query_role_logits: Sequence,
) -> ConsumerReceipt:
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_consumer_worker,
        args=(child, wire, query, query_role_logits),
    )
    process.start()
    child.close()
    message = _receive(parent, process, "consumer")
    parent.close()
    return ConsumerReceipt(
        pid=int(message["pid"]),
        query_sha256=message["query_sha256"],
        wire_sha256=message["wire_sha256"],
        answer=message["answer"],
    )


def run_process_custody(
    *,
    source: bytes,
    source_role_logits: Sequence,
    query: bytes,
    query_role_logits: Sequence,
) -> ProcessCustodyReceipt:
    wire, producer = compile_in_fresh_process(
        source=source,
        source_role_logits=source_role_logits,
    )
    consumer = execute_in_fresh_process(
        wire=wire,
        query=query,
        query_role_logits=query_role_logits,
    )
    return ProcessCustodyReceipt(producer=producer, consumer=consumer)


__all__ = [
    "ConsumerReceipt",
    "MultiFamilyProcessError",
    "ProcessCustodyReceipt",
    "ProducerReceipt",
    "compile_in_fresh_process",
    "execute_in_fresh_process",
    "run_process_custody",
]
