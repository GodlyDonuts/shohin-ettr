import hashlib
import json
from pathlib import Path

import pytest

from opcode_program_registry import (
    AUDIT_SCHEMA,
    OpcodeProgramRegistryError,
    load_opcode_program_registry,
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("ascii")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _report(path: Path) -> str:
    programs = ((1, 6), (3, 4, 6))
    report = {
        "cross_split_coverage": {"opcode": {"development_instance_rate": 0.99}},
        "data_root": "/immutable/data",
        "schema": AUDIT_SCHEMA,
        "splits": {
            "development": {},
            "train": {
                "opcode_registry": [
                    {
                        "count": count,
                        "opcodes": list(program),
                        "sha256": _digest(program),
                    }
                    for count, program in zip((3, 2), programs, strict=True)
                ],
                "programs": {
                    "opcode": {"instances": 5, "unique": 2},
                },
            },
        },
        "status": "pass",
    }
    report["report_payload_sha256"] = _digest(report)
    payload = _canonical(report)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_load_opcode_program_registry_binds_programs_and_coverage(tmp_path) -> None:
    path = (tmp_path / "registry.json").resolve()
    digest = _report(path)
    registry = load_opcode_program_registry(
        path,
        expected_sha256=digest,
        max_steps=3,
        opcode_classes=9,
    )
    assert registry.classes == 2
    assert registry.programs == ((1, 6), (3, 4, 6))
    assert registry.development_instance_coverage == 0.99
    assert registry.file_sha256 == digest


def test_load_opcode_program_registry_rejects_mutation(tmp_path) -> None:
    path = (tmp_path / "registry.json").resolve()
    digest = _report(path)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(OpcodeProgramRegistryError, match="file hash"):
        load_opcode_program_registry(
            path,
            expected_sha256=digest,
            max_steps=3,
            opcode_classes=9,
        )
