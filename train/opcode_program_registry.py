"""Hash-bound loader for preregistered ETTR opcode-program skeletons."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re


AUDIT_SCHEMA = "r12-ettr-program-template-audit-v2"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class OpcodeProgramRegistryError(ValueError):
    """An opcode-program registry differs from its audited receipt."""


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


@dataclass(frozen=True, slots=True)
class OpcodeProgramRegistry:
    file_sha256: str
    payload_sha256: str
    development_instance_coverage: float
    programs: tuple[tuple[int, ...], ...]
    payload: bytes

    @property
    def classes(self) -> int:
        return len(self.programs)


def load_opcode_program_registry(
    path: Path,
    *,
    expected_sha256: str,
    max_steps: int,
    opcode_classes: int,
) -> OpcodeProgramRegistry:
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or _HEX64.fullmatch(expected_sha256) is None
        or not isinstance(max_steps, int)
        or max_steps < 1
        or not isinstance(opcode_classes, int)
        or opcode_classes < 2
    ):
        raise OpcodeProgramRegistryError("opcode registry arguments differ")
    payload = path.read_bytes()
    file_sha256 = hashlib.sha256(payload).hexdigest()
    if file_sha256 != expected_sha256:
        raise OpcodeProgramRegistryError("opcode registry file hash differs")
    try:
        report = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpcodeProgramRegistryError("opcode registry is malformed") from exc
    if not isinstance(report, dict) or _canonical(report) != payload:
        raise OpcodeProgramRegistryError("opcode registry is not canonical")
    claimed = report.pop("report_payload_sha256", None)
    if (
        report.get("schema") != AUDIT_SCHEMA
        or report.get("status") != "pass"
        or not isinstance(claimed, str)
        or _HEX64.fullmatch(claimed) is None
        or _digest(report) != claimed
    ):
        raise OpcodeProgramRegistryError("opcode registry audit receipt differs")
    try:
        train = report["splits"]["train"]
        summary = train["programs"]["opcode"]
        entries = train["opcode_registry"]
        coverage = report["cross_split_coverage"]["opcode"]
        expected_unique = summary["unique"]
        expected_instances = summary["instances"]
        development_instance_coverage = coverage["development_instance_rate"]
    except (KeyError, TypeError) as exc:
        raise OpcodeProgramRegistryError("opcode registry geometry differs") from exc
    if (
        not isinstance(entries, list)
        or not isinstance(expected_unique, int)
        or expected_unique < 2
        or len(entries) != expected_unique
        or not isinstance(expected_instances, int)
        or expected_instances < expected_unique
        or not isinstance(development_instance_coverage, (int, float))
        or isinstance(development_instance_coverage, bool)
        or not 0.0 <= development_instance_coverage <= 1.0
    ):
        raise OpcodeProgramRegistryError("opcode registry geometry differs")

    programs = []
    observed_instances = 0
    observed_digests = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "count",
            "opcodes",
            "sha256",
        }:
            raise OpcodeProgramRegistryError("opcode registry entry differs")
        count = entry["count"]
        opcodes = entry["opcodes"]
        digest = entry["sha256"]
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 1
            or not isinstance(opcodes, list)
            or not 1 <= len(opcodes) <= max_steps
            or any(type(opcode) is not int for opcode in opcodes)
            or any(not 0 <= opcode < opcode_classes for opcode in opcodes)
            or not isinstance(digest, str)
            or _HEX64.fullmatch(digest) is None
            or _digest(tuple(opcodes)) != digest
            or digest in observed_digests
        ):
            raise OpcodeProgramRegistryError("opcode registry entry differs")
        observed_instances += count
        observed_digests.add(digest)
        programs.append(tuple(opcodes))
    if observed_instances != expected_instances or len(set(programs)) != len(programs):
        raise OpcodeProgramRegistryError("opcode registry population differs")
    return OpcodeProgramRegistry(
        file_sha256=file_sha256,
        payload_sha256=claimed,
        development_instance_coverage=float(development_instance_coverage),
        programs=tuple(programs),
        payload=payload,
    )
