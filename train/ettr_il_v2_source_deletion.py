"""CPU-only operational source-deletion rehearsal for ETTR-IL-v2.

This module is a mechanics harness, not a model runner.  It executes two
otherwise identical ``WORLD -> COMMAND -> QUERY`` lanes:

* a clean lane physically removes each stage input package before launching
  the next stage; and
* a poison lane removes the same package, replaces its old path with a valid
  but contradictory source package, and proves the downstream artifact is
  unchanged.

Each stage runs in a fresh isolated Python process.  The child installs an
audit hook that denies every file read outside its exact closed-world input
manifest, every write outside its two declared output files, all network
access, and nested process creation.  The only semantic object crossing a
stage boundary is a fixed-shape categorical packet (or the final answer).
Raw source bytes, transformer residuals, hidden states, and KV caches are not
transferable fields or files.

The rehearsal reuses the strict CJ1, ``FileRecord``, and file-set-root custody
contracts plus ETTR's frozen packet dimensions and opcode enum.  It imports no
model, optimizer, trainer, checkpoint writer, or job launcher.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import builtins
import os
from pathlib import Path
import secrets
import shutil
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from ettr_il_v2_custody import (  # noqa: E402
    FileRecord,
    cj1_dumps,
    cj1_loads,
    file_set_root,
    sha256_bytes,
)
from ettr_il_v2_materialize import (  # noqa: E402
    MAX_EDGES,
    NUM_RELATIONS,
    NUM_SLOTS,
    NUM_TYPES,
    NUM_VALUE_CODES,
    Opcode,
)


PROTOCOL = "R12-ETTR-IL-v2"
REHEARSAL_SCHEMA = "r12-ettr-il-v2-source-deletion-rehearsal-v1"
MANIFEST_SCHEMA = "r12-ettr-il-v2-stage-closed-world-manifest-v1"
EXECUTION_RECEIPT_SCHEMA = "r12-ettr-il-v2-stage-execution-receipt-v1"
DELETION_RECEIPT_SCHEMA = "r12-ettr-il-v2-source-deletion-receipt-v1"
WORLD_SOURCE_SCHEMA = "r12-ettr-il-v2-world-source-package-v1"
COMMAND_SOURCE_SCHEMA = "r12-ettr-il-v2-command-source-package-v1"
QUERY_SOURCE_SCHEMA = "r12-ettr-il-v2-query-source-package-v1"
SEALED_PACKET_SCHEMA = "r12-ettr-il-v2-sealed-packet-v1"
TERMINAL_PACKET_SCHEMA = "r12-ettr-il-v2-terminal-packet-v1"
ANSWER_SCHEMA = "r12-ettr-il-v2-source-deletion-answer-v1"

STAGES = ("WORLD", "COMMAND", "QUERY")
_UPSTREAM_STAGE = {"COMMAND": "WORLD", "QUERY": "COMMAND"}
_ARTIFACT_NAME = {
    "WORLD": "sealed_packet.json",
    "COMMAND": "terminal_packet.json",
    "QUERY": "answer.json",
}
_INPUT_NAMES = {
    "WORLD": frozenset({"source.json"}),
    "COMMAND": frozenset(
        {
            "source.json",
            "upstream_packet.json",
            "upstream_receipt.json",
            "upstream_deletion_receipt.json",
        }
    ),
    "QUERY": frozenset(
        {
            "source.json",
            "upstream_packet.json",
            "upstream_receipt.json",
            "upstream_deletion_receipt.json",
        }
    ),
}
_FORBIDDEN_TRANSFER_KEY_PARTS = (
    "residual",
    "hidden_state",
    "kv_cache",
    "key_cache",
    "value_cache",
    "world_source",
    "command_source",
    "query_source",
    "source_payload",
    "source_bytes",
)
_SHA256_LENGTH = 64


class SourceDeletionError(RuntimeError):
    """The operational source-deletion contract failed closed."""


@dataclass(frozen=True, slots=True)
class SourceDeletionRun:
    """One preserved clean/poison rehearsal and its final receipt."""

    root: Path
    receipt_path: Path
    receipt_sha256: str
    receipt: Mapping[str, Any]


def _strict_object(
    value: object,
    fields: Sequence[str],
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise SourceDeletionError(f"{label} is not an object")
    expected = set(fields)
    if set(value) != expected:
        raise SourceDeletionError(f"{label} fields differ")
    return value


def _require_plain_int(
    value: object,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        raise SourceDeletionError(f"{label} integer differs")
    return value


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SourceDeletionError(f"{label} SHA-256 differs")
    return value


def _read_regular(path: Path, *, immutable: bool = True) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise SourceDeletionError(f"required file is unavailable: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SourceDeletionError(f"path is not a regular non-symlink: {path}")
    if immutable and metadata.st_mode & 0o222:
        raise SourceDeletionError(f"input file is mutable: {path}")
    return path.read_bytes()


def _read_cj1(path: Path, *, immutable: bool = True) -> tuple[bytes, Any]:
    payload = _read_regular(path, immutable=immutable)
    try:
        value = cj1_loads(payload)
    except Exception as exc:
        raise SourceDeletionError(f"CJ1 file is invalid: {path}") from exc
    return payload, value


def _write_once(path: Path, payload: bytes) -> str:
    if not isinstance(payload, bytes):
        raise SourceDeletionError("immutable payload is not literal bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise SourceDeletionError(f"short immutable write: {path}")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    path.chmod(0o444)
    return sha256_bytes(payload)


def _write_cj1_once(path: Path, value: object) -> str:
    return _write_once(path, cj1_dumps(value))


def _directory_names(path: Path) -> set[str]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise SourceDeletionError(f"closed-world directory is unavailable: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise SourceDeletionError(f"closed-world path is not a directory: {path}")
    names: set[str] = set()
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise SourceDeletionError(
                    f"closed-world entry is not a regular file: {entry.path}"
                )
            names.add(entry.name)
    return names


def _file_record(name: str, payload: bytes) -> FileRecord:
    return FileRecord.from_payload(
        path=name,
        payload=payload,
        row_count=1,
        media_type="application/json",
        confidentiality="candidate",
    )


def _manifest_payload(
    *,
    stage: str,
    payloads: Mapping[str, bytes],
    parent_receipt_sha256s: Sequence[str],
) -> bytes:
    if stage not in STAGES or set(payloads) != set(_INPUT_NAMES[stage]):
        raise SourceDeletionError("stage input payload set differs")
    records = tuple(
        _file_record(name, payloads[name]) for name in sorted(payloads)
    )
    parents = [
        _require_sha256(value, "parent receipt")
        for value in parent_receipt_sha256s
    ]
    expected_parent_count = 0 if stage == "WORLD" else 2
    if len(parents) != expected_parent_count or len(parents) != len(set(parents)):
        raise SourceDeletionError("stage parent receipt set differs")
    return cj1_dumps(
        {
            "closed_world": True,
            "file_set_root": file_set_root(records),
            "files": [record.to_object() for record in records],
            "parent_receipt_sha256s": parents,
            "protocol": PROTOCOL,
            "schema": MANIFEST_SCHEMA,
            "stage": stage,
        }
    )


def _prepare_stage_input(
    directory: Path,
    *,
    stage: str,
    payloads: Mapping[str, bytes],
    parent_receipt_sha256s: Sequence[str] = (),
) -> str:
    directory.mkdir(parents=True, exist_ok=False)
    for name in sorted(payloads):
        _write_once(directory / name, payloads[name])
    manifest = _manifest_payload(
        stage=stage,
        payloads=payloads,
        parent_receipt_sha256s=parent_receipt_sha256s,
    )
    digest = _write_once(directory / "manifest.json", manifest)
    directory.chmod(0o555)
    return digest


def _load_stage_input(
    directory: Path,
    stage: str,
) -> tuple[bytes, dict[str, Any], dict[str, bytes]]:
    expected_names = set(_INPUT_NAMES[stage]) | {"manifest.json"}
    if _directory_names(directory) != expected_names:
        raise SourceDeletionError("closed-world stage file set differs")
    manifest_payload, manifest_value = _read_cj1(directory / "manifest.json")
    manifest = _strict_object(
        manifest_value,
        (
            "closed_world",
            "file_set_root",
            "files",
            "parent_receipt_sha256s",
            "protocol",
            "schema",
            "stage",
        ),
        "stage manifest",
    )
    if (
        manifest["schema"] != MANIFEST_SCHEMA
        or manifest["protocol"] != PROTOCOL
        or manifest["stage"] != stage
        or manifest["closed_world"] is not True
        or not isinstance(manifest["files"], list)
        or not isinstance(manifest["parent_receipt_sha256s"], list)
    ):
        raise SourceDeletionError("stage manifest identity differs")
    records = tuple(
        FileRecord.from_object(value) for value in manifest["files"]
    )
    if tuple(record.path for record in records) != tuple(sorted(_INPUT_NAMES[stage])):
        raise SourceDeletionError("stage manifest file order differs")
    if file_set_root(records) != manifest["file_set_root"]:
        raise SourceDeletionError("stage manifest file-set root differs")
    payloads: dict[str, bytes] = {}
    for record in records:
        payload = _read_regular(directory / record.path)
        record.verify_payload(payload)
        payloads[record.path] = payload
    parent_hashes = tuple(
        _require_sha256(value, "manifest parent receipt")
        for value in manifest["parent_receipt_sha256s"]
    )
    if len(parent_hashes) != (0 if stage == "WORLD" else 2):
        raise SourceDeletionError("manifest parent receipt count differs")
    return manifest_payload, manifest, payloads


def _install_worker_audit(
    *,
    stage: str,
    input_directory: Path,
    output_directory: Path,
) -> None:
    input_root = os.path.realpath(input_directory)
    output_root = os.path.realpath(output_directory)
    allowed_reads = {
        os.path.realpath(input_directory / name)
        for name in set(_INPUT_NAMES[stage]) | {"manifest.json"}
    }
    allowed_writes = {
        os.path.realpath(output_directory / _ARTIFACT_NAME[stage]),
        os.path.realpath(output_directory / "receipt.json"),
    }
    allowed_scans = {input_root, output_root}

    def normalize(value: object) -> str | None:
        if isinstance(value, int):
            return None
        path = os.fspath(value)
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)
        return os.path.realpath(path)

    def audit(event: str, arguments: tuple[object, ...]) -> None:
        if event.startswith("socket.") or event in {
            "http.client.connect",
            "subprocess.Popen",
            "urllib.Request",
        }:
            raise PermissionError("worker network or process creation is forbidden")
        if event == "open" and arguments:
            path = normalize(arguments[0])
            if path is None:
                return
            mode = arguments[1] if len(arguments) > 1 else "r"
            flags = arguments[2] if len(arguments) > 2 else 0
            writing = (
                isinstance(mode, str)
                and any(marker in mode for marker in "wax+")
            ) or (
                isinstance(flags, int)
                and bool(
                    flags
                    & (
                        os.O_WRONLY
                        | os.O_RDWR
                        | os.O_CREAT
                        | os.O_TRUNC
                        | os.O_APPEND
                    )
                )
            )
            allowed = allowed_writes if writing else allowed_reads
            if path not in allowed:
                raise PermissionError("file access outside closed-world stage")
        if event in {"os.listdir", "os.scandir"} and arguments:
            path = normalize(arguments[0])
            if path not in allowed_scans:
                raise PermissionError("directory scan outside closed-world stage")

    sys.addaudithook(audit)


def _probe_forbidden_sources(paths: Sequence[Path]) -> list[str]:
    receipts: list[str] = []
    for path in paths:
        try:
            with builtins.open(path, "rb"):
                pass
        except PermissionError:
            receipts.append(sha256_bytes(os.fsencode(os.path.realpath(path))))
            continue
        except OSError as exc:
            raise SourceDeletionError(
                "forbidden source probe reached the operating system"
            ) from exc
        raise SourceDeletionError("worker read a forbidden upstream source")
    return receipts


def _packet_object(
    *,
    schema: str,
    values: Sequence[int],
    types: Sequence[int],
    active: Sequence[bool],
    root: Sequence[bool],
    relations: Sequence[Sequence[int]],
    committed: bool,
    halted: bool,
) -> dict[str, Any]:
    value = {
        "active": list(active),
        "committed": committed,
        "halted": halted,
        "relations": [list(edge) for edge in relations],
        "root": list(root),
        "schema": schema,
        "type_index": list(types),
        "value_code": list(values),
    }
    _validate_packet(value, schema=schema)
    return value


def _validate_packet(value: object, *, schema: str) -> dict[str, Any]:
    packet = _strict_object(
        value,
        (
            "active",
            "committed",
            "halted",
            "relations",
            "root",
            "schema",
            "type_index",
            "value_code",
        ),
        "sealed packet",
    )
    if (
        packet["schema"] != schema
        or not isinstance(packet["committed"], bool)
        or not isinstance(packet["halted"], bool)
    ):
        raise SourceDeletionError("sealed packet identity differs")
    arrays = ("active", "root", "type_index", "value_code")
    if any(
        not isinstance(packet[name], list)
        or len(packet[name]) != NUM_SLOTS
        for name in arrays
    ):
        raise SourceDeletionError("sealed packet width differs")
    if any(not isinstance(item, bool) for item in packet["active"]):
        raise SourceDeletionError("sealed packet active mask differs")
    if any(not isinstance(item, bool) for item in packet["root"]):
        raise SourceDeletionError("sealed packet root mask differs")
    if any(
        _require_plain_int(
            item,
            "packet type",
            minimum=0,
            maximum=NUM_TYPES - 1,
        )
        != item
        for item in packet["type_index"]
    ):
        raise SourceDeletionError("sealed packet types differ")
    if any(
        _require_plain_int(
            item,
            "packet value",
            minimum=0,
            maximum=NUM_VALUE_CODES - 1,
        )
        != item
        for item in packet["value_code"]
    ):
        raise SourceDeletionError("sealed packet values differ")
    for slot, is_active in enumerate(packet["active"]):
        if not is_active and (
            packet["type_index"][slot] != 0
            or packet["value_code"][slot] != 0
            or packet["root"][slot]
        ):
            raise SourceDeletionError("inactive packet slot carries state")
    root_slots = [
        slot for slot, is_root in enumerate(packet["root"]) if is_root
    ]
    if len(root_slots) > 1 or any(
        not packet["active"][slot] for slot in root_slots
    ):
        raise SourceDeletionError("sealed packet root differs")
    if not isinstance(packet["relations"], list):
        raise SourceDeletionError("sealed packet relation ledger differs")
    relations: list[tuple[int, int, int]] = []
    for edge in packet["relations"]:
        if not isinstance(edge, list) or len(edge) != 3:
            raise SourceDeletionError("sealed packet edge differs")
        relation = _require_plain_int(
            edge[0],
            "edge relation",
            minimum=0,
            maximum=NUM_RELATIONS - 1,
        )
        source = _require_plain_int(
            edge[1],
            "edge source",
            minimum=0,
            maximum=NUM_SLOTS - 1,
        )
        target = _require_plain_int(
            edge[2],
            "edge target",
            minimum=0,
            maximum=NUM_SLOTS - 1,
        )
        if not packet["active"][source] or not packet["active"][target]:
            raise SourceDeletionError("sealed packet edge endpoint is inactive")
        relations.append((relation, source, target))
    if (
        relations != sorted(relations)
        or len(relations) != len(set(relations))
        or len(relations) > MAX_EDGES
    ):
        raise SourceDeletionError("sealed packet relation order differs")
    return packet


def _compile_world(source: object) -> dict[str, Any]:
    world = _strict_object(
        source,
        ("cells", "relations", "root_slot", "schema", "source_sentinel"),
        "WORLD source package",
    )
    if (
        world["schema"] != WORLD_SOURCE_SCHEMA
        or not isinstance(world["source_sentinel"], str)
        or not world["source_sentinel"].startswith("WORLD-RAW-")
        or not isinstance(world["cells"], list)
        or not isinstance(world["relations"], list)
    ):
        raise SourceDeletionError("WORLD source package identity differs")
    values = [0] * NUM_SLOTS
    types = [0] * NUM_SLOTS
    active = [False] * NUM_SLOTS
    observed_slots: set[int] = set()
    for item in world["cells"]:
        cell = _strict_object(
            item,
            ("slot", "type_index", "value_code"),
            "WORLD source cell",
        )
        slot = _require_plain_int(
            cell["slot"],
            "WORLD cell slot",
            minimum=0,
            maximum=47,
        )
        type_index = _require_plain_int(
            cell["type_index"],
            "WORLD cell type",
            minimum=0,
            maximum=NUM_TYPES - 1,
        )
        value_code = _require_plain_int(
            cell["value_code"],
            "WORLD cell value",
            minimum=1,
            maximum=NUM_VALUE_CODES - 1,
        )
        if slot in observed_slots:
            raise SourceDeletionError("WORLD source repeats a slot")
        if (slot < 32 and type_index >= 4) or (
            slot >= 32 and type_index != 4
        ):
            raise SourceDeletionError("WORLD source cell projection differs")
        observed_slots.add(slot)
        active[slot] = True
        types[slot] = type_index
        values[slot] = value_code
    root_slot = _require_plain_int(
        world["root_slot"],
        "WORLD root",
        minimum=0,
        maximum=NUM_SLOTS - 1,
    )
    if root_slot not in observed_slots:
        raise SourceDeletionError("WORLD source root is inactive")
    root = [slot == root_slot for slot in range(NUM_SLOTS)]
    relations = sorted(world["relations"])
    return _packet_object(
        schema=SEALED_PACKET_SCHEMA,
        values=values,
        types=types,
        active=active,
        root=root,
        relations=relations,
        committed=False,
        halted=False,
    )


def _execute_command(packet_value: object, source: object) -> dict[str, Any]:
    packet = _validate_packet(packet_value, schema=SEALED_PACKET_SCHEMA)
    command = _strict_object(
        source,
        ("operations", "schema", "source_sentinel"),
        "COMMAND source package",
    )
    if (
        command["schema"] != COMMAND_SOURCE_SCHEMA
        or not isinstance(command["source_sentinel"], str)
        or not command["source_sentinel"].startswith("COMMAND-RAW-")
        or not isinstance(command["operations"], list)
        or not command["operations"]
    ):
        raise SourceDeletionError("COMMAND source package identity differs")
    values = list(packet["value_code"])
    types = list(packet["type_index"])
    active = list(packet["active"])
    root = list(packet["root"])
    relations = {tuple(edge) for edge in packet["relations"]}
    committed = packet["committed"]
    halted = packet["halted"]
    for position, operation_value in enumerate(command["operations"]):
        if committed or halted:
            raise SourceDeletionError("COMMAND mutates a terminal packet")
        if not isinstance(operation_value, dict) or "opcode" not in operation_value:
            raise SourceDeletionError("COMMAND operation differs")
        opcode_name = operation_value["opcode"]
        if not isinstance(opcode_name, str) or opcode_name not in Opcode.__members__:
            raise SourceDeletionError("COMMAND opcode differs")
        opcode = Opcode[opcode_name]
        before = (
            tuple(values),
            tuple(types),
            tuple(active),
            tuple(root),
            frozenset(relations),
            committed,
            halted,
        )
        if opcode is Opcode.WRITE:
            operation = _strict_object(
                operation_value,
                ("opcode", "source", "value_code"),
                "COMMAND WRITE",
            )
            slot = _require_plain_int(
                operation["source"],
                "COMMAND WRITE source",
                minimum=32,
                maximum=47,
            )
            value_code = _require_plain_int(
                operation["value_code"],
                "COMMAND WRITE value",
                minimum=1,
                maximum=NUM_VALUE_CODES - 1,
            )
            if not active[slot]:
                raise SourceDeletionError("COMMAND writes an inactive slot")
            values[slot] = value_code
        elif opcode in {Opcode.LINK, Opcode.UNLINK}:
            operation = _strict_object(
                operation_value,
                ("opcode", "relation", "source", "target"),
                f"COMMAND {opcode.name}",
            )
            relation = _require_plain_int(
                operation["relation"],
                "COMMAND edge relation",
                minimum=0,
                maximum=NUM_RELATIONS - 1,
            )
            source_slot = _require_plain_int(
                operation["source"],
                "COMMAND edge source",
                minimum=0,
                maximum=NUM_SLOTS - 1,
            )
            target_slot = _require_plain_int(
                operation["target"],
                "COMMAND edge target",
                minimum=0,
                maximum=NUM_SLOTS - 1,
            )
            if not active[source_slot] or not active[target_slot]:
                raise SourceDeletionError("COMMAND edge endpoint is inactive")
            edge = (relation, source_slot, target_slot)
            if opcode is Opcode.LINK:
                if edge in relations:
                    raise SourceDeletionError("COMMAND repeats an existing edge")
                relations.add(edge)
            else:
                if edge not in relations:
                    raise SourceDeletionError("COMMAND removes a missing edge")
                relations.remove(edge)
        elif opcode is Opcode.SET_ROOT:
            operation = _strict_object(
                operation_value,
                ("opcode", "source"),
                "COMMAND SET_ROOT",
            )
            slot = _require_plain_int(
                operation["source"],
                "COMMAND root source",
                minimum=0,
                maximum=NUM_SLOTS - 1,
            )
            if not active[slot]:
                raise SourceDeletionError("COMMAND roots an inactive slot")
            root = [index == slot for index in range(NUM_SLOTS)]
        elif opcode is Opcode.COMMIT:
            _strict_object(operation_value, ("opcode",), "COMMAND COMMIT")
            committed = True
        elif opcode is Opcode.HALT:
            _strict_object(operation_value, ("opcode",), "COMMAND HALT")
            halted = True
        elif opcode is Opcode.REJECT:
            _strict_object(operation_value, ("opcode",), "COMMAND REJECT")
            committed = True
            halted = True
        else:
            raise SourceDeletionError(
                f"COMMAND rehearsal does not admit {opcode.name}"
            )
        after = (
            tuple(values),
            tuple(types),
            tuple(active),
            tuple(root),
            frozenset(relations),
            committed,
            halted,
        )
        if before == after:
            raise SourceDeletionError(
                f"COMMAND operation {position} has no state effect"
            )
    if not committed and not halted:
        raise SourceDeletionError("COMMAND did not seal the terminal packet")
    return _packet_object(
        schema=TERMINAL_PACKET_SCHEMA,
        values=values,
        types=types,
        active=active,
        root=root,
        relations=sorted(relations),
        committed=committed,
        halted=halted,
    )


def _answer_query(packet_value: object, source: object) -> dict[str, Any]:
    packet = _validate_packet(packet_value, schema=TERMINAL_PACKET_SCHEMA)
    query = _strict_object(
        source,
        ("predicate", "schema", "source_sentinel"),
        "QUERY source package",
    )
    if (
        query["schema"] != QUERY_SOURCE_SCHEMA
        or not isinstance(query["source_sentinel"], str)
        or not query["source_sentinel"].startswith("QUERY-RAW-")
    ):
        raise SourceDeletionError("QUERY source package identity differs")
    predicate = _strict_object(
        query["predicate"],
        ("kind", "relation", "source", "target", "value_code", "value_slot"),
        "QUERY predicate",
    )
    if predicate["kind"] != "value_and_edge":
        raise SourceDeletionError("QUERY predicate kind differs")
    value_slot = _require_plain_int(
        predicate["value_slot"],
        "QUERY value slot",
        minimum=0,
        maximum=NUM_SLOTS - 1,
    )
    value_code = _require_plain_int(
        predicate["value_code"],
        "QUERY value code",
        minimum=0,
        maximum=NUM_VALUE_CODES - 1,
    )
    relation = _require_plain_int(
        predicate["relation"],
        "QUERY relation",
        minimum=0,
        maximum=NUM_RELATIONS - 1,
    )
    source_slot = _require_plain_int(
        predicate["source"],
        "QUERY edge source",
        minimum=0,
        maximum=NUM_SLOTS - 1,
    )
    target_slot = _require_plain_int(
        predicate["target"],
        "QUERY edge target",
        minimum=0,
        maximum=NUM_SLOTS - 1,
    )
    answer = (
        packet["committed"]
        and not packet["halted"]
        and packet["active"][value_slot]
        and packet["value_code"][value_slot] == value_code
        and [relation, source_slot, target_slot] in packet["relations"]
    )
    return {
        "answer": answer,
        "schema": ANSWER_SCHEMA,
        "terminal_packet_sha256": sha256_bytes(cj1_dumps(packet)),
    }


def _forbidden_transfer_keys_absent(value: object) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(part in lowered for part in _FORBIDDEN_TRANSFER_KEY_PARTS):
                return False
            if not _forbidden_transfer_keys_absent(child):
                return False
        return True
    if isinstance(value, list):
        return all(_forbidden_transfer_keys_absent(child) for child in value)
    return True


def _parse_execution_receipt(
    value: object,
    *,
    expected_stage: str,
) -> dict[str, Any]:
    receipt = _strict_object(
        value,
        (
            "denied_upstream_source_probe_path_sha256s",
            "forbidden_transfer_scan",
            "input_file_set_root",
            "input_manifest_sha256",
            "output_artifact",
            "output_artifact_sha256",
            "output_file_set_root",
            "parent_receipt_sha256s",
            "protocol",
            "schema",
            "source_package_sha256",
            "spawn_nonce",
            "stage",
            "worker_parent_pid",
            "worker_pid",
        ),
        "stage execution receipt",
    )
    if (
        receipt["schema"] != EXECUTION_RECEIPT_SCHEMA
        or receipt["protocol"] != PROTOCOL
        or receipt["stage"] != expected_stage
        or receipt["output_artifact"] != _ARTIFACT_NAME[expected_stage]
        or not isinstance(receipt["parent_receipt_sha256s"], list)
        or not isinstance(
            receipt["denied_upstream_source_probe_path_sha256s"], list
        )
        or not isinstance(receipt["forbidden_transfer_scan"], dict)
        or receipt["forbidden_transfer_scan"]
        != {
            "kv_cache_absent": True,
            "raw_upstream_source_absent": True,
            "residual_state_absent": True,
        }
    ):
        raise SourceDeletionError("stage execution receipt identity differs")
    for field in (
        "input_file_set_root",
        "input_manifest_sha256",
        "output_artifact_sha256",
        "output_file_set_root",
        "source_package_sha256",
    ):
        _require_sha256(receipt[field], f"stage receipt {field}")
    for value in receipt["parent_receipt_sha256s"]:
        _require_sha256(value, "stage receipt parent")
    for value in receipt["denied_upstream_source_probe_path_sha256s"]:
        _require_sha256(value, "denied probe path")
    if (
        not isinstance(receipt["spawn_nonce"], str)
        or len(receipt["spawn_nonce"]) != 32
        or any(character not in "0123456789abcdef" for character in receipt["spawn_nonce"])
    ):
        raise SourceDeletionError("stage receipt spawn nonce differs")
    _require_plain_int(
        receipt["worker_pid"],
        "worker PID",
        minimum=1,
        maximum=(1 << 31) - 1,
    )
    _require_plain_int(
        receipt["worker_parent_pid"],
        "worker parent PID",
        minimum=1,
        maximum=(1 << 31) - 1,
    )
    return receipt


def _parse_deletion_receipt(
    value: object,
    *,
    expected_stage: str,
) -> dict[str, Any]:
    receipt = _strict_object(
        value,
        (
            "deleted_input_file_set_root",
            "execution_receipt_sha256",
            "package_removed_before_successor",
            "poison_replacement_sha256",
            "protocol",
            "schema",
            "source_package_sha256",
            "stage",
        ),
        "source deletion receipt",
    )
    if (
        receipt["schema"] != DELETION_RECEIPT_SCHEMA
        or receipt["protocol"] != PROTOCOL
        or receipt["stage"] != expected_stage
        or receipt["package_removed_before_successor"] is not True
    ):
        raise SourceDeletionError("source deletion receipt identity differs")
    for field in (
        "deleted_input_file_set_root",
        "execution_receipt_sha256",
        "source_package_sha256",
    ):
        _require_sha256(receipt[field], f"deletion receipt {field}")
    replacement = receipt["poison_replacement_sha256"]
    if replacement is not None:
        _require_sha256(replacement, "poison replacement")
    return receipt


def _validate_parent_chain(
    *,
    stage: str,
    manifest: Mapping[str, Any],
    payloads: Mapping[str, bytes],
) -> None:
    parent_hashes = manifest["parent_receipt_sha256s"]
    if stage == "WORLD":
        if parent_hashes:
            raise SourceDeletionError("WORLD unexpectedly has a parent receipt")
        return
    expected_upstream = _UPSTREAM_STAGE[stage]
    receipt_payload = payloads["upstream_receipt.json"]
    deletion_payload = payloads["upstream_deletion_receipt.json"]
    if parent_hashes != [
        sha256_bytes(receipt_payload),
        sha256_bytes(deletion_payload),
    ]:
        raise SourceDeletionError("parent receipt binding differs")
    try:
        execution = _parse_execution_receipt(
            cj1_loads(receipt_payload),
            expected_stage=expected_upstream,
        )
        deletion = _parse_deletion_receipt(
            cj1_loads(deletion_payload),
            expected_stage=expected_upstream,
        )
    except Exception as exc:
        if isinstance(exc, SourceDeletionError):
            raise
        raise SourceDeletionError("upstream receipt is invalid") from exc
    if (
        deletion["execution_receipt_sha256"] != sha256_bytes(receipt_payload)
        or execution["output_artifact_sha256"]
        != sha256_bytes(payloads["upstream_packet.json"])
        or execution["source_package_sha256"]
        != deletion["source_package_sha256"]
    ):
        raise SourceDeletionError("upstream receipt chain differs")


def _worker(
    *,
    stage: str,
    input_directory: Path,
    output_directory: Path,
    spawn_nonce: str,
    forbidden_source_paths: Sequence[Path],
) -> None:
    if stage not in STAGES:
        raise SourceDeletionError("worker stage differs")
    output_directory.mkdir(parents=True, exist_ok=False)
    _install_worker_audit(
        stage=stage,
        input_directory=input_directory,
        output_directory=output_directory,
    )
    manifest_payload, manifest, payloads = _load_stage_input(
        input_directory,
        stage,
    )
    _validate_parent_chain(stage=stage, manifest=manifest, payloads=payloads)
    denied_probes = _probe_forbidden_sources(forbidden_source_paths)
    source_payload = payloads["source.json"]
    try:
        source_value = cj1_loads(source_payload)
        if stage == "WORLD":
            artifact = _compile_world(source_value)
        elif stage == "COMMAND":
            artifact = _execute_command(
                cj1_loads(payloads["upstream_packet.json"]),
                source_value,
            )
        else:
            artifact = _answer_query(
                cj1_loads(payloads["upstream_packet.json"]),
                source_value,
            )
    except Exception as exc:
        if isinstance(exc, SourceDeletionError):
            raise
        raise SourceDeletionError(f"{stage} worker input is invalid") from exc
    if not _forbidden_transfer_keys_absent(artifact):
        raise SourceDeletionError("stage artifact contains forbidden transfer keys")
    artifact_payload = cj1_dumps(artifact)
    if source_payload in artifact_payload:
        raise SourceDeletionError("stage artifact contains its raw source package")
    artifact_name = _ARTIFACT_NAME[stage]
    artifact_sha256 = _write_once(
        output_directory / artifact_name,
        artifact_payload,
    )
    artifact_record = _file_record(artifact_name, artifact_payload)
    receipt = {
        "denied_upstream_source_probe_path_sha256s": denied_probes,
        "forbidden_transfer_scan": {
            "kv_cache_absent": True,
            "raw_upstream_source_absent": True,
            "residual_state_absent": True,
        },
        "input_file_set_root": manifest["file_set_root"],
        "input_manifest_sha256": sha256_bytes(manifest_payload),
        "output_artifact": artifact_name,
        "output_artifact_sha256": artifact_sha256,
        "output_file_set_root": file_set_root((artifact_record,)),
        "parent_receipt_sha256s": manifest["parent_receipt_sha256s"],
        "protocol": PROTOCOL,
        "schema": EXECUTION_RECEIPT_SCHEMA,
        "source_package_sha256": sha256_bytes(source_payload),
        "spawn_nonce": spawn_nonce,
        "stage": stage,
        "worker_parent_pid": os.getppid(),
        "worker_pid": os.getpid(),
    }
    receipt_payload = cj1_dumps(receipt)
    if source_payload in receipt_payload:
        raise SourceDeletionError("stage receipt contains its raw source package")
    _write_once(output_directory / "receipt.json", receipt_payload)
    output_directory.chmod(0o555)


def _invoke_stage_worker(
    *,
    stage: str,
    input_directory: Path,
    output_directory: Path,
    spawn_nonce: str,
    forbidden_source_paths: Sequence[Path] = (),
) -> None:
    command = [
        sys.executable,
        "-I",
        "-S",
        "-B",
        str(Path(__file__).resolve()),
        "_worker",
        "--stage",
        stage,
        "--input-directory",
        str(input_directory.resolve()),
        "--output-directory",
        str(output_directory.resolve()),
        "--spawn-nonce",
        spawn_nonce,
    ]
    for path in forbidden_source_paths:
        command.extend(("--forbidden-source-path", str(path.resolve())))
    environment = {
        "HOME": str(output_directory.parent.resolve()),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        tail = detail[-1] if detail else "worker emitted no diagnostic"
        raise SourceDeletionError(f"{stage} worker failed closed: {tail}")


def _verify_stage_output(
    *,
    stage: str,
    output_directory: Path,
    input_manifest_sha256: str,
    source_payload: bytes,
    parent_receipt_sha256s: Sequence[str],
    spawn_nonce: str,
) -> tuple[bytes, bytes, dict[str, Any]]:
    if _directory_names(output_directory) != {
        _ARTIFACT_NAME[stage],
        "receipt.json",
    }:
        raise SourceDeletionError("worker output file set differs")
    artifact_payload, artifact = _read_cj1(
        output_directory / _ARTIFACT_NAME[stage]
    )
    receipt_payload, receipt_value = _read_cj1(
        output_directory / "receipt.json"
    )
    receipt = _parse_execution_receipt(
        receipt_value,
        expected_stage=stage,
    )
    if (
        receipt["input_manifest_sha256"] != input_manifest_sha256
        or receipt["source_package_sha256"] != sha256_bytes(source_payload)
        or receipt["parent_receipt_sha256s"] != list(parent_receipt_sha256s)
        or receipt["output_artifact_sha256"] != sha256_bytes(artifact_payload)
        or receipt["spawn_nonce"] != spawn_nonce
        or receipt["worker_parent_pid"] != os.getpid()
        or receipt["worker_pid"] == os.getpid()
        or source_payload in artifact_payload
        or source_payload in receipt_payload
        or not _forbidden_transfer_keys_absent(artifact)
    ):
        raise SourceDeletionError("worker output receipt binding differs")
    if stage == "WORLD":
        _validate_packet(artifact, schema=SEALED_PACKET_SCHEMA)
    elif stage == "COMMAND":
        _validate_packet(artifact, schema=TERMINAL_PACKET_SCHEMA)
    else:
        answer = _strict_object(
            artifact,
            ("answer", "schema", "terminal_packet_sha256"),
            "QUERY answer",
        )
        if (
            answer["schema"] != ANSWER_SCHEMA
            or not isinstance(answer["answer"], bool)
        ):
            raise SourceDeletionError("QUERY answer identity differs")
        _require_sha256(
            answer["terminal_packet_sha256"],
            "QUERY terminal packet",
        )
    return artifact_payload, receipt_payload, receipt


def _delete_stage_input(
    *,
    stage: str,
    input_directory: Path,
    input_file_set_root: str,
    source_payload: bytes,
    execution_receipt_payload: bytes,
    deletion_directory: Path,
    poison_payload: bytes | None,
) -> tuple[bytes, dict[str, Any]]:
    input_directory.chmod(0o755)
    shutil.rmtree(input_directory)
    if input_directory.exists():
        raise SourceDeletionError("stage source package survived deletion")
    replacement_sha256: str | None = None
    if poison_payload is not None:
        input_directory.mkdir(parents=True, exist_ok=False)
        replacement_sha256 = _write_once(
            input_directory / "source.json",
            poison_payload,
        )
        input_directory.chmod(0o555)
    receipt = {
        "deleted_input_file_set_root": _require_sha256(
            input_file_set_root,
            "deleted input file-set root",
        ),
        "execution_receipt_sha256": sha256_bytes(execution_receipt_payload),
        "package_removed_before_successor": True,
        "poison_replacement_sha256": replacement_sha256,
        "protocol": PROTOCOL,
        "schema": DELETION_RECEIPT_SCHEMA,
        "source_package_sha256": sha256_bytes(source_payload),
        "stage": stage,
    }
    deletion_directory.mkdir(parents=True, exist_ok=True)
    path = deletion_directory / f"{stage.lower()}.json"
    payload = cj1_dumps(receipt)
    _write_once(path, payload)
    _parse_deletion_receipt(receipt, expected_stage=stage)
    return payload, receipt


def _cleanup_replacement(path: Path) -> None:
    if path.exists():
        path.chmod(0o755)
        shutil.rmtree(path)
    if path.exists():
        raise SourceDeletionError("poison replacement survived cleanup")


def _source_packages(*, poison: bool) -> dict[str, bytes]:
    suffix = "POISON" if poison else "PRIMARY"
    if poison:
        world_cells = [
            {"slot": 0, "type_index": 0, "value_code": 2},
            {"slot": 32, "type_index": 4, "value_code": 40},
            {"slot": 33, "type_index": 4, "value_code": 41},
            {"slot": 34, "type_index": 4, "value_code": 72},
        ]
        command_value = 72
        query_value = 72
    else:
        world_cells = [
            {"slot": 0, "type_index": 0, "value_code": 1},
            {"slot": 32, "type_index": 4, "value_code": 33},
            {"slot": 33, "type_index": 4, "value_code": 34},
            {"slot": 34, "type_index": 4, "value_code": 70},
        ]
        command_value = 71
        query_value = 71
    return {
        "WORLD": cj1_dumps(
            {
                "cells": world_cells,
                "relations": [[0, 32, 33]],
                "root_slot": 0,
                "schema": WORLD_SOURCE_SCHEMA,
                "source_sentinel": f"WORLD-RAW-{suffix}-DO-NOT-TRANSFER",
            }
        ),
        "COMMAND": cj1_dumps(
            {
                "operations": [
                    {
                        "opcode": "WRITE",
                        "source": 34,
                        "value_code": command_value,
                    },
                    {
                        "opcode": "LINK",
                        "relation": 1,
                        "source": 33,
                        "target": 34,
                    },
                    {"opcode": "COMMIT"},
                ],
                "schema": COMMAND_SOURCE_SCHEMA,
                "source_sentinel": f"COMMAND-RAW-{suffix}-DO-NOT-TRANSFER",
            }
        ),
        "QUERY": cj1_dumps(
            {
                "predicate": {
                    "kind": "value_and_edge",
                    "relation": 1,
                    "source": 33,
                    "target": 34,
                    "value_code": query_value,
                    "value_slot": 34,
                },
                "schema": QUERY_SOURCE_SCHEMA,
                "source_sentinel": f"QUERY-RAW-{suffix}-DO-NOT-TRANSFER",
            }
        ),
    }


def _run_lane(directory: Path, *, install_poison: bool) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    inputs_root = directory / "inputs"
    outputs_root = directory / "outputs"
    deletions_root = directory / "deletions"
    inputs_root.mkdir()
    outputs_root.mkdir()
    source_packages = _source_packages(poison=False)
    poison_packages = _source_packages(poison=True)
    stage_artifacts: dict[str, bytes] = {}
    stage_receipts: dict[str, bytes] = {}
    deletion_receipts: dict[str, bytes] = {}
    stage_receipt_values: dict[str, dict[str, Any]] = {}
    deletion_values: dict[str, dict[str, Any]] = {}
    upstream_forbidden_paths: list[Path] = []

    for stage in STAGES:
        input_directory = inputs_root / stage.lower()
        output_directory = outputs_root / stage.lower()
        payloads: dict[str, bytes] = {"source.json": source_packages[stage]}
        parent_hashes: list[str] = []
        if stage != "WORLD":
            upstream = _UPSTREAM_STAGE[stage]
            payloads.update(
                {
                    "upstream_deletion_receipt.json": deletion_receipts[upstream],
                    "upstream_packet.json": stage_artifacts[upstream],
                    "upstream_receipt.json": stage_receipts[upstream],
                }
            )
            parent_hashes = [
                sha256_bytes(stage_receipts[upstream]),
                sha256_bytes(deletion_receipts[upstream]),
            ]
        manifest_sha256 = _prepare_stage_input(
            input_directory,
            stage=stage,
            payloads=payloads,
            parent_receipt_sha256s=parent_hashes,
        )
        spawn_nonce = secrets.token_hex(16)
        _invoke_stage_worker(
            stage=stage,
            input_directory=input_directory,
            output_directory=output_directory,
            spawn_nonce=spawn_nonce,
            forbidden_source_paths=tuple(upstream_forbidden_paths),
        )
        artifact_payload, execution_payload, execution = _verify_stage_output(
            stage=stage,
            output_directory=output_directory,
            input_manifest_sha256=manifest_sha256,
            source_payload=source_packages[stage],
            parent_receipt_sha256s=parent_hashes,
            spawn_nonce=spawn_nonce,
        )
        stage_artifacts[stage] = artifact_payload
        stage_receipts[stage] = execution_payload
        stage_receipt_values[stage] = execution
        poison_payload = poison_packages[stage] if install_poison else None
        deletion_payload, deletion = _delete_stage_input(
            stage=stage,
            input_directory=input_directory,
            input_file_set_root=execution["input_file_set_root"],
            source_payload=source_packages[stage],
            execution_receipt_payload=execution_payload,
            deletion_directory=deletions_root,
            poison_payload=poison_payload,
        )
        deletion_receipts[stage] = deletion_payload
        deletion_values[stage] = deletion
        upstream_forbidden_paths.append(input_directory / "source.json")

    answer_before_cleanup = sha256_bytes(stage_artifacts["QUERY"])
    source_sentinels = tuple(
        str(cj1_loads(payload)["source_sentinel"]).encode("ascii")
        for payload in (*source_packages.values(), *poison_packages.values())
    )
    carried_payloads = tuple(stage_artifacts.values()) + tuple(
        stage_receipts.values()
    )
    if any(
        source_payload in carried
        for source_payload in (
            *source_packages.values(),
            *poison_packages.values(),
        )
        for carried in carried_payloads
    ) or any(
        sentinel in carried
        for sentinel in source_sentinels
        for carried in carried_payloads
    ):
        raise SourceDeletionError("raw source content crossed a stage boundary")
    for path in (inputs_root / stage.lower() for stage in STAGES):
        _cleanup_replacement(path)
    if any((inputs_root / stage.lower()).exists() for stage in STAGES):
        raise SourceDeletionError("source package remains after lane completion")
    if sha256_bytes(stage_artifacts["QUERY"]) != answer_before_cleanup:
        raise SourceDeletionError("QUERY output changed after source replacement")
    process_launch_receipts = [
        sha256_bytes(
            cj1_dumps(
                {
                    "input_manifest_sha256": receipt["input_manifest_sha256"],
                    "spawn_nonce": receipt["spawn_nonce"],
                    "stage": stage,
                    "worker_pid": receipt["worker_pid"],
                }
            )
        )
        for stage, receipt in stage_receipt_values.items()
    ]
    if len(process_launch_receipts) != len(set(process_launch_receipts)):
        raise SourceDeletionError("stage process launch receipt is duplicated")
    return {
        "artifact_sha256s": {
            stage: sha256_bytes(stage_artifacts[stage]) for stage in STAGES
        },
        "deletion_receipt_sha256s": {
            stage: sha256_bytes(deletion_receipts[stage]) for stage in STAGES
        },
        "denied_upstream_source_probes": {
            stage: len(
                stage_receipt_values[stage][
                    "denied_upstream_source_probe_path_sha256s"
                ]
            )
            for stage in STAGES
        },
        "execution_receipt_sha256s": {
            stage: sha256_bytes(stage_receipts[stage]) for stage in STAGES
        },
        "package_removed_before_successor": {
            stage: deletion_values[stage]["package_removed_before_successor"]
            for stage in STAGES
        },
        "poison_replacement_sha256s": {
            stage: deletion_values[stage]["poison_replacement_sha256"]
            for stage in STAGES
        },
        "process_launch_receipts": process_launch_receipts,
        "source_package_sha256s": {
            stage: sha256_bytes(source_packages[stage]) for stage in STAGES
        },
        "transfer_scan": {
            "all_primary_source_payloads_absent": True,
            "all_primary_source_sentinels_absent": True,
            "all_replacement_payloads_absent": True,
            "all_replacement_sentinels_absent": True,
        },
        "worker_parent_pids": {
            stage: stage_receipt_values[stage]["worker_parent_pid"]
            for stage in STAGES
        },
        "worker_pids": {
            stage: stage_receipt_values[stage]["worker_pid"]
            for stage in STAGES
        },
    }


def run_source_deletion_rehearsal(root: Path) -> SourceDeletionRun:
    """Run clean and poison CPU lanes and publish one immutable receipt."""

    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    clean = _run_lane(root / "clean", install_poison=False)
    poisoned = _run_lane(root / "poisoned", install_poison=True)
    artifact_invariance = {
        stage: (
            clean["artifact_sha256s"][stage]
            == poisoned["artifact_sha256s"][stage]
        )
        for stage in STAGES
    }
    if not all(artifact_invariance.values()):
        raise SourceDeletionError("poison replacement changed a stage artifact")
    all_parent_pids = [
        lane["worker_parent_pids"][stage]
        for lane in (clean, poisoned)
        for stage in STAGES
    ]
    all_worker_pids = [
        lane["worker_pids"][stage]
        for lane in (clean, poisoned)
        for stage in STAGES
    ]
    all_launch_receipts = [
        value
        for lane in (clean, poisoned)
        for value in lane["process_launch_receipts"]
    ]
    if (
        any(parent != os.getpid() for parent in all_parent_pids)
        or any(worker == os.getpid() for worker in all_worker_pids)
        or len(all_launch_receipts) != 6
        or len(set(all_launch_receipts)) != 6
    ):
        raise SourceDeletionError("spawned process-boundary evidence differs")
    if clean["denied_upstream_source_probes"] != {
        "WORLD": 0,
        "COMMAND": 1,
        "QUERY": 2,
    } or poisoned["denied_upstream_source_probes"] != {
        "WORLD": 0,
        "COMMAND": 1,
        "QUERY": 2,
    }:
        raise SourceDeletionError("upstream source denied-probe counts differ")
    report = {
        "artifact_invariance_under_poison_replacement": artifact_invariance,
        "clean_lane": clean,
        "hard_process_boundaries": {
            "all_workers_differ_from_supervisor": True,
            "process_launch_receipts_unique": True,
            "spawned_stage_processes": 6,
        },
        "mode": "cpu_only_no_model_no_fit",
        "poisoned_lane": poisoned,
        "protocol": PROTOCOL,
        "schema": REHEARSAL_SCHEMA,
        "source_deletion_claim_boundary": (
            "physical_file_removal_and_process_isolation_not_memory_erasure"
        ),
        "status": "pass",
        "transfer_closure": {
            "digest_only_source_binding": True,
            "kv_cache_transfer_forbidden": True,
            "raw_upstream_source_transfer_forbidden": True,
            "residual_state_transfer_forbidden": True,
            "sealed_packet_is_only_interstage_semantic_artifact": True,
        },
    }
    receipt_path = root / "rehearsal_receipt.json"
    receipt_sha256 = _write_cj1_once(receipt_path, report)
    return SourceDeletionRun(
        root=root,
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha256,
        receipt=report,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("mode", choices=("_worker",))
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--spawn-nonce", required=True)
    parser.add_argument(
        "--forbidden-source-path",
        type=Path,
        action="append",
        default=[],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _worker(
        stage=args.stage,
        input_directory=args.input_directory.resolve(),
        output_directory=args.output_directory.resolve(),
        spawn_nonce=args.spawn_nonce,
        forbidden_source_paths=tuple(
            path.resolve() for path in args.forbidden_source_path
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANSWER_SCHEMA",
    "COMMAND_SOURCE_SCHEMA",
    "DELETION_RECEIPT_SCHEMA",
    "EXECUTION_RECEIPT_SCHEMA",
    "MANIFEST_SCHEMA",
    "PROTOCOL",
    "QUERY_SOURCE_SCHEMA",
    "REHEARSAL_SCHEMA",
    "SEALED_PACKET_SCHEMA",
    "SourceDeletionError",
    "SourceDeletionRun",
    "TERMINAL_PACKET_SCHEMA",
    "WORLD_SOURCE_SCHEMA",
    "run_source_deletion_rehearsal",
]
