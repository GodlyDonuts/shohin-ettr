"""CPU-only three-process custody mechanics for SSQAC.

This module is deliberately a mechanics harness, not a reasoning evaluator.  It
uses three serial, isolated Python subprocess programs:

1. a compiler turns a trivial raw row-operation source into a sealed tensor;
2. a candidate executes only the sealed tensor with a primitive ALU; and
3. an assessor, created only after candidate exit, checks the mechanical result.

The candidate program is passed with ``python -c``.  It is therefore not given
this combined harness, the compiler program, or the assessor program as an
input file.  The harness proves delivered-file closure and guarded file reads;
it does not claim a hostile-kernel security boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCHEMA = "ssqac-three-process-custody-mechanics-v1"
COMPILER_INPUT_SCHEMA = "ssqac-compiler-input-manifest-v1"
CANDIDATE_INPUT_SCHEMA = "ssqac-candidate-closed-world-manifest-v1"
ASSESSOR_INPUT_SCHEMA = "ssqac-assessor-closed-world-manifest-v1"

_CANDIDATE_INPUT_FILES = frozenset(
    {"manifest.json", "primitive_runtime.json", "sealed_algebra.json"}
)
_CANDIDATE_DATA_FILES = ("primitive_runtime.json", "sealed_algebra.json")
_CANDIDATE_OUTPUT_FILES = frozenset({"candidate_result.json"})
_ASSESSOR_INPUT_FILES = frozenset(
    {"candidate_result.json", "expected_result.json", "manifest.json"}
)
_ASSESSOR_OUTPUT_FILES = frozenset({"assessment.json"})
_COMPILER_INPUT_FILES = frozenset({"manifest.json", "raw_source.json"})
_COMPILER_OUTPUT_FILES = frozenset(
    {
        "compiler_receipt.json",
        "expected_result.json",
        "primitive_runtime.json",
        "sealed_algebra.json",
    }
)

_FORBIDDEN_NAME_MARKERS = (
    "raw_source",
    "board",
    "generator",
    "completion",
    "gold",
    "verifier",
    "query",
    "answer",
    "assessor",
)
_FORBIDDEN_CONTENT_MARKERS = tuple(
    marker.encode("ascii")
    for marker in (
        '"raw_source"',
        '"board_generator"',
        '"completion_enumeration"',
        '"gold_verifier"',
        '"query_answer"',
        '"assessor_input"',
        '"assessor_files"',
        "source-only-custody-sentinel",
    )
)


class CustodyError(RuntimeError):
    """Raised when any custody or process condition fails closed."""


@dataclass(frozen=True)
class CustodyRun:
    """Paths and digest for one verified mechanics run."""

    root: Path
    receipt_path: Path
    receipt_sha256: str
    receipt: Mapping[str, Any]


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise CustodyError("value is not canonical ASCII JSON") from exc
    return encoded + b"\n"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_once(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise CustodyError(f"short immutable write: {path}")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    path.chmod(0o444)
    return _sha256_bytes(payload)


def _write_json_once(path: Path, value: Any) -> str:
    return _write_once(path, _canonical_json_bytes(value))


def _read_regular(path: Path, *, canonical_json: bool = False) -> bytes:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise CustodyError(f"required file is absent: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise CustodyError(f"path is not a regular non-symlink file: {path}")
    if metadata.st_mode & 0o222:
        raise CustodyError(f"immutable file has write bits: {path}")
    payload = path.read_bytes()
    if canonical_json:
        try:
            decoded = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CustodyError(f"malformed JSON artifact: {path}") from exc
        if payload != _canonical_json_bytes(decoded):
            raise CustodyError(f"JSON artifact is not canonical: {path}")
    return payload


def _file_reference(path: Path, *, relative_to: Path) -> dict[str, Any]:
    payload = _read_regular(path, canonical_json=path.suffix == ".json")
    metadata = path.lstat()
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": _sha256_bytes(payload),
        "bytes": len(payload),
        "mode": stat.S_IMODE(metadata.st_mode),
    }


def _assert_directory_files(directory: Path, expected: set[str] | frozenset[str]) -> None:
    try:
        metadata = directory.lstat()
    except FileNotFoundError as exc:
        raise CustodyError(f"closed-world directory is absent: {directory}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise CustodyError(f"closed-world path is not a directory: {directory}")
    observed: set[str] = set()
    for entry in os.scandir(directory):
        if entry.is_symlink():
            raise CustodyError(f"symlink is forbidden in closed world: {entry.path}")
        if not entry.is_file(follow_symlinks=False):
            raise CustodyError(f"non-file entry is forbidden in closed world: {entry.path}")
        observed.add(entry.name)
    if observed != set(expected):
        raise CustodyError(
            f"closed-world file set differs for {directory}: "
            f"expected={sorted(expected)} observed={sorted(observed)}"
        )


def _verify_manifest(
    directory: Path,
    *,
    schema: str,
    payload_files: set[str] | frozenset[str],
) -> dict[str, Any]:
    _assert_directory_files(directory, set(payload_files) | {"manifest.json"})
    manifest_payload = _read_regular(directory / "manifest.json", canonical_json=True)
    manifest = json.loads(manifest_payload)
    if set(manifest) != {"closed_world", "files", "schema"}:
        raise CustodyError("closed-world manifest keys differ")
    if manifest["schema"] != schema or manifest["closed_world"] is not True:
        raise CustodyError("closed-world manifest contract differs")
    if not isinstance(manifest["files"], dict) or set(manifest["files"]) != set(
        payload_files
    ):
        raise CustodyError("closed-world manifest payload set differs")
    for name in sorted(payload_files):
        reference = manifest["files"][name]
        if set(reference) != {"bytes", "sha256"}:
            raise CustodyError(f"manifest reference keys differ: {name}")
        payload = _read_regular(directory / name, canonical_json=True)
        if (
            reference["sha256"] != _sha256_bytes(payload)
            or reference["bytes"] != len(payload)
        ):
            raise CustodyError(f"manifest hash or size differs: {name}")
    return manifest


def _manifest_for(directory: Path, *, schema: str, payload_files: Sequence[str]) -> dict:
    references: dict[str, dict[str, Any]] = {}
    for name in sorted(payload_files):
        payload = _read_regular(directory / name, canonical_json=True)
        references[name] = {
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
        }
    return {"schema": schema, "closed_world": True, "files": references}


def _scan_candidate_inputs(directory: Path, *, raw_source: bytes) -> dict[str, Any]:
    _assert_directory_files(directory, _CANDIDATE_INPUT_FILES)
    scans: list[dict[str, Any]] = []
    raw_hash = _sha256_bytes(raw_source)
    for name in sorted(_CANDIDATE_INPUT_FILES):
        lowered_name = name.lower()
        if any(marker in lowered_name for marker in _FORBIDDEN_NAME_MARKERS):
            raise CustodyError(f"candidate input has forbidden filename: {name}")
        payload = _read_regular(directory / name, canonical_json=True)
        lowered = payload.lower()
        if any(marker in lowered for marker in _FORBIDDEN_CONTENT_MARKERS):
            raise CustodyError(f"candidate input has forbidden content: {name}")
        if _sha256_bytes(payload) == raw_hash or raw_source in payload:
            raise CustodyError(f"candidate input contains raw compiler source: {name}")
        scans.append(
            {
                "path": name,
                "sha256": _sha256_bytes(payload),
                "forbidden_name_markers_absent": True,
                "forbidden_content_markers_absent": True,
                "raw_source_absent": True,
            }
        )
    return {
        "files": scans,
        "forbidden_filenames_absent": True,
        "forbidden_content_absent": True,
        "raw_source_bytes_absent": True,
    }


def _lock_directory(directory: Path) -> None:
    directory.chmod(0o555)


_CHILD_COMMON = r'''
import hashlib
import json
import os
import stat
import sys
import time

def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n")

def digest(payload):
    return hashlib.sha256(payload).hexdigest()

ROOT = os.path.realpath(os.getcwd())
ALLOWED_READS = set()
ALLOWED_WRITES = set()
ALLOWED_SCANS = {ROOT}
NETWORK_GUARD = True

def normalized(value):
    if isinstance(value, int):
        return None
    path = os.fspath(value)
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    return os.path.realpath(path)

def audit(event, args):
    if event.startswith("socket.") or event in {
        "urllib.Request", "http.client.connect", "ftplib.connect"
    }:
        raise PermissionError("network access is forbidden")
    if event == "open" and args:
        path = normalized(args[0])
        if path is None:
            return
        mode = args[1] if len(args) > 1 else "r"
        flags = args[2] if len(args) > 2 else 0
        writing = (
            isinstance(mode, str) and any(flag in mode for flag in "wax+")
        ) or (isinstance(mode, int) and bool(
            mode & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        )) or (isinstance(flags, int) and bool(
            flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        ))
        allowed = ALLOWED_WRITES if writing else ALLOWED_READS
        if path not in allowed:
            raise PermissionError("file access outside the closed world")
    if event in {"os.listdir", "os.scandir"} and args:
        path = normalized(args[0])
        if path not in ALLOWED_SCANS:
            raise PermissionError("directory scan outside the closed world")

sys.addaudithook(audit)

def regular(path):
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("non-regular input")
    if metadata.st_mode & 0o222:
        raise RuntimeError("writable immutable input")
    return metadata

def read(path):
    regular(path)
    with open(path, "rb") as handle:
        return handle.read()

def read_json(path):
    payload = read(path)
    value = json.loads(payload)
    if payload != canonical(value):
        raise RuntimeError("non-canonical JSON")
    return value, payload

def write_once(path, value):
    payload = canonical(value)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise RuntimeError("short output write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o444)
    return {"sha256": digest(payload), "bytes": len(payload)}

def exact_files(directory, expected):
    names = set()
    for entry in os.scandir(directory):
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            raise RuntimeError("closed-world entry is not a regular file")
        names.add(entry.name)
    if names != set(expected):
        raise RuntimeError("closed-world file set differs")

def verify_manifest(directory, schema, payload_names):
    exact_files(directory, set(payload_names) | {"manifest.json"})
    manifest, manifest_bytes = read_json(os.path.join(directory, "manifest.json"))
    if set(manifest) != {"closed_world", "files", "schema"}:
        raise RuntimeError("manifest keys differ")
    if manifest["schema"] != schema or manifest["closed_world"] is not True:
        raise RuntimeError("manifest contract differs")
    if set(manifest["files"]) != set(payload_names):
        raise RuntimeError("manifest file set differs")
    for name in payload_names:
        payload = read(os.path.join(directory, name))
        reference = manifest["files"][name]
        if set(reference) != {"bytes", "sha256"}:
            raise RuntimeError("manifest reference keys differ")
        if reference["bytes"] != len(payload) or reference["sha256"] != digest(payload):
            raise RuntimeError("manifest reference differs")
    return manifest, manifest_bytes

def primitive_execute(tensor, instructions, field):
    rows = [[int(value) % field for value in row] for row in tensor]
    executed = 0
    halted = False
    for instruction in instructions:
        opcode = instruction["opcode"]
        if halted:
            raise RuntimeError("instruction follows HALT")
        if opcode == "HALT":
            halted = True
            continue
        if opcode != "AXPY" or set(instruction) != {
            "destination", "opcode", "scale", "source"
        }:
            raise RuntimeError("non-primitive instruction")
        destination = int(instruction["destination"])
        source = int(instruction["source"])
        scale = int(instruction["scale"]) % field
        if not (0 <= destination < len(rows) and 0 <= source < len(rows)):
            raise RuntimeError("row operand out of range")
        if len(rows[destination]) != len(rows[source]):
            raise RuntimeError("ragged tensor")
        rows[destination] = [
            (left + scale * right) % field
            for left, right in zip(rows[destination], rows[source])
        ]
        executed += 1
    if not halted:
        raise RuntimeError("program did not HALT")
    return {"tensor": rows, "executed_primitives": executed, "halted": True}

if os.environ.get("SSQAC_TEST_FAIL") == "1":
    raise RuntimeError("injected subprocess failure")
'''


_COMPILER_PROGRAM = _CHILD_COMMON + r'''
started_ns = time.time_ns()
input_dir = os.path.join(ROOT, "input")
output_dir = os.path.join(ROOT, "output")
ALLOWED_SCANS.update({input_dir, output_dir})
ALLOWED_READS.update({
    os.path.join(input_dir, "manifest.json"),
    os.path.join(input_dir, "raw_source.json"),
})
for name in {
    "compiler_receipt.json", "expected_result.json",
    "primitive_runtime.json", "sealed_algebra.json"
}:
    ALLOWED_WRITES.add(os.path.join(output_dir, name))

manifest, manifest_bytes = verify_manifest(
    input_dir, "ssqac-compiler-input-manifest-v1", {"raw_source.json"}
)
exact_files(output_dir, set())
source, source_bytes = read_json(os.path.join(input_dir, "raw_source.json"))
if set(source) != {"field", "instructions", "mechanics_only", "sentinel", "tensor"}:
    raise RuntimeError("raw source keys differ")
if source["mechanics_only"] is not True:
    raise RuntimeError("source makes a reasoning claim")
if source["sentinel"] != "source-only-custody-sentinel":
    raise RuntimeError("source sentinel differs")
field = int(source["field"])
if field != 257:
    raise RuntimeError("unexpected mechanics field")
expected = primitive_execute(source["tensor"], source["instructions"], field)
sealed = {
    "schema": "ssqac-sealed-algebra-mechanics-v1",
    "field": field,
    "tensor": source["tensor"],
    "instructions": source["instructions"],
    "mechanics_only": True,
}
runtime = {
    "schema": "ssqac-primitive-runtime-v1",
    "field": field,
    "allowed_opcodes": ["AXPY", "HALT"],
    "branching": False,
    "host_solver": False,
}
sealed_ref = write_once(os.path.join(output_dir, "sealed_algebra.json"), sealed)
runtime_ref = write_once(os.path.join(output_dir, "primitive_runtime.json"), runtime)
expected_ref = write_once(os.path.join(output_dir, "expected_result.json"), expected)
receipt = {
    "schema": "ssqac-compiler-receipt-v1",
    "input_manifest_sha256": digest(manifest_bytes),
    "raw_source_sha256": digest(source_bytes),
    "outputs": {
        "expected_result.json": expected_ref,
        "primitive_runtime.json": runtime_ref,
        "sealed_algebra.json": sealed_ref,
    },
    "mechanics_only": True,
    "network_guard": NETWORK_GUARD,
}
write_once(os.path.join(output_dir, "compiler_receipt.json"), receipt)
exact_files(output_dir, {
    "compiler_receipt.json", "expected_result.json",
    "primitive_runtime.json", "sealed_algebra.json"
})
event = {
    "role": "compiler", "pid": os.getpid(), "started_ns": started_ns,
    "ended_ns": time.time_ns(), "network_guard": NETWORK_GUARD,
    "read_files": ["input/manifest.json", "input/raw_source.json"],
}
sys.stdout.buffer.write(canonical(event))
'''


_CANDIDATE_PROGRAM = _CHILD_COMMON + r'''
started_ns = time.time_ns()
input_dir = os.path.join(ROOT, "input")
output_dir = os.path.join(ROOT, "output")
ALLOWED_SCANS.update({input_dir, output_dir})
ALLOWED_READS.update({
    os.path.join(input_dir, "manifest.json"),
    os.path.join(input_dir, "primitive_runtime.json"),
    os.path.join(input_dir, "sealed_algebra.json"),
})
ALLOWED_WRITES.add(os.path.join(output_dir, "candidate_result.json"))

manifest, manifest_bytes = verify_manifest(
    input_dir, "ssqac-candidate-closed-world-manifest-v1",
    {"primitive_runtime.json", "sealed_algebra.json"}
)
exact_files(output_dir, set())
runtime, runtime_bytes = read_json(os.path.join(input_dir, "primitive_runtime.json"))
sealed, sealed_bytes = read_json(os.path.join(input_dir, "sealed_algebra.json"))
if set(runtime) != {
    "allowed_opcodes", "branching", "field", "host_solver", "schema"
}:
    raise RuntimeError("primitive runtime keys differ")
if runtime != {
    "schema": "ssqac-primitive-runtime-v1",
    "field": 257,
    "allowed_opcodes": ["AXPY", "HALT"],
    "branching": False,
    "host_solver": False,
}:
    raise RuntimeError("primitive runtime contract differs")
if set(sealed) != {
    "field", "instructions", "mechanics_only", "schema", "tensor"
}:
    raise RuntimeError("sealed algebra keys differ")
if sealed["schema"] != "ssqac-sealed-algebra-mechanics-v1":
    raise RuntimeError("sealed algebra schema differs")
if sealed["mechanics_only"] is not True or sealed["field"] != runtime["field"]:
    raise RuntimeError("sealed algebra contract differs")
if any(
    instruction.get("opcode") not in runtime["allowed_opcodes"]
    for instruction in sealed["instructions"]
):
    raise RuntimeError("instruction is outside primitive runtime")
result = primitive_execute(
    sealed["tensor"], sealed["instructions"], int(runtime["field"])
)
output = {
    "schema": "ssqac-candidate-mechanics-result-v1",
    "candidate_input_manifest_sha256": digest(manifest_bytes),
    "primitive_runtime_sha256": digest(runtime_bytes),
    "sealed_algebra_sha256": digest(sealed_bytes),
    "result": result,
    "mechanics_only": True,
    "reasoning_claim": False,
    "network_guard": NETWORK_GUARD,
}
output_ref = write_once(os.path.join(output_dir, "candidate_result.json"), output)
exact_files(output_dir, {"candidate_result.json"})
event = {
    "role": "candidate", "pid": os.getpid(), "started_ns": started_ns,
    "ended_ns": time.time_ns(), "network_guard": NETWORK_GUARD,
    "input_manifest_sha256": digest(manifest_bytes),
    "read_files": [
        "input/manifest.json", "input/primitive_runtime.json",
        "input/sealed_algebra.json"
    ],
    "output": output_ref,
}
sys.stdout.buffer.write(canonical(event))
'''


_ASSESSOR_PROGRAM = _CHILD_COMMON + r'''
started_ns = time.time_ns()
input_dir = os.path.join(ROOT, "input")
output_dir = os.path.join(ROOT, "output")
ALLOWED_SCANS.update({input_dir, output_dir})
ALLOWED_READS.update({
    os.path.join(input_dir, "manifest.json"),
    os.path.join(input_dir, "candidate_result.json"),
    os.path.join(input_dir, "expected_result.json"),
})
ALLOWED_WRITES.add(os.path.join(output_dir, "assessment.json"))

manifest, manifest_bytes = verify_manifest(
    input_dir, "ssqac-assessor-closed-world-manifest-v1",
    {"candidate_result.json", "expected_result.json"}
)
exact_files(output_dir, set())
candidate, candidate_bytes = read_json(
    os.path.join(input_dir, "candidate_result.json")
)
expected, expected_bytes = read_json(os.path.join(input_dir, "expected_result.json"))
if candidate.get("schema") != "ssqac-candidate-mechanics-result-v1":
    raise RuntimeError("candidate result schema differs")
if candidate.get("mechanics_only") is not True:
    raise RuntimeError("candidate result is not mechanics-only")
if candidate.get("reasoning_claim") is not False:
    raise RuntimeError("candidate made a reasoning claim")
passed = candidate.get("result") == expected
assessment = {
    "schema": "ssqac-assessment-mechanics-v1",
    "assessor_input_manifest_sha256": digest(manifest_bytes),
    "candidate_result_sha256": digest(candidate_bytes),
    "expected_result_sha256": digest(expected_bytes),
    "passed": passed,
    "mechanics_only": True,
    "reasoning_claim": False,
    "network_guard": NETWORK_GUARD,
}
write_once(os.path.join(output_dir, "assessment.json"), assessment)
exact_files(output_dir, {"assessment.json"})
if not passed:
    raise RuntimeError("candidate mechanics result differs")
event = {
    "role": "assessor", "pid": os.getpid(), "started_ns": started_ns,
    "ended_ns": time.time_ns(), "network_guard": NETWORK_GUARD,
    "read_files": [
        "input/manifest.json", "input/candidate_result.json",
        "input/expected_result.json"
    ],
}
sys.stdout.buffer.write(canonical(event))
'''


def _minimal_environment(root: Path, *, fail: bool) -> dict[str, str]:
    environment = {
        "HOME": str(root),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
    }
    if fail:
        environment["SSQAC_TEST_FAIL"] = "1"
    return environment


def _run_role(
    role: str,
    program: str,
    workspace: Path,
    *,
    inject_failure: bool,
) -> dict[str, Any]:
    observed_start = time.time_ns()
    process = subprocess.Popen(
        [sys.executable, "-I", "-S", "-c", program],
        cwd=workspace,
        env=_minimal_environment(workspace, fail=inject_failure),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate()
    observed_end = time.time_ns()
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise CustodyError(
            f"{role} subprocess failed with exit {process.returncode}: {detail}"
        )
    if stderr:
        raise CustodyError(f"{role} subprocess wrote unexpected stderr")
    try:
        event = json.loads(stdout)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CustodyError(f"{role} subprocess emitted malformed receipt") from exc
    if stdout != _canonical_json_bytes(event):
        raise CustodyError(f"{role} subprocess receipt is not canonical")
    if (
        set(event).issuperset(
            {"ended_ns", "network_guard", "pid", "read_files", "role", "started_ns"}
        )
        is False
        or event["role"] != role
        or event["pid"] != process.pid
        or event["network_guard"] is not True
        or not (
            observed_start
            <= event["started_ns"]
            <= event["ended_ns"]
            <= observed_end
        )
    ):
        raise CustodyError(f"{role} process receipt differs")
    return {
        "role": role,
        "pid": process.pid,
        "returncode": process.returncode,
        "parent_observed_start_ns": observed_start,
        "child_started_ns": event["started_ns"],
        "child_ended_ns": event["ended_ns"],
        "parent_observed_end_ns": observed_end,
        "network_guard": True,
        "read_files": event["read_files"],
    }


def _default_source() -> dict[str, Any]:
    return {
        "field": 257,
        "tensor": [[1, 2], [0, 1]],
        "instructions": [
            {"opcode": "AXPY", "destination": 0, "source": 1, "scale": -2},
            {"opcode": "HALT"},
        ],
        "mechanics_only": True,
        "sentinel": "source-only-custody-sentinel",
    }


def _references(directory: Path, names: Sequence[str], root: Path) -> list[dict]:
    return [_file_reference(directory / name, relative_to=root) for name in sorted(names)]


def _invoke_hook(
    hook: Callable[[str, Path], None] | None, phase: str, directory: Path
) -> None:
    if hook is not None:
        hook(phase, directory)


def run_three_process_custody(
    root: str | Path,
    *,
    _phase_hook: Callable[[str, Path], None] | None = None,
    _fault_role: str | None = None,
) -> CustodyRun:
    """Run and verify the three-process mechanics harness.

    ``_phase_hook`` and ``_fault_role`` exist only for adversarial unit tests.
    Normal callers should leave them unset.
    """

    root = Path(root).resolve()
    if _fault_role not in {None, "compiler", "candidate", "assessor"}:
        raise ValueError("unknown injected fault role")
    if root.exists():
        if root.is_symlink() or not root.is_dir() or any(root.iterdir()):
            raise CustodyError("custody root must be absent or an empty real directory")
    else:
        root.mkdir(parents=True, mode=0o700)

    compiler = root / "compiler_workspace"
    compiler_input = compiler / "input"
    compiler_output = compiler / "output"
    compiler_input.mkdir(parents=True, mode=0o700)
    compiler_output.mkdir(mode=0o700)
    source = _default_source()
    source_bytes = _canonical_json_bytes(source)
    source_hash = _write_once(compiler_input / "raw_source.json", source_bytes)
    compiler_manifest = _manifest_for(
        compiler_input,
        schema=COMPILER_INPUT_SCHEMA,
        payload_files=["raw_source.json"],
    )
    compiler_manifest_hash = _write_json_once(
        compiler_input / "manifest.json", compiler_manifest
    )
    _verify_manifest(
        compiler_input,
        schema=COMPILER_INPUT_SCHEMA,
        payload_files={"raw_source.json"},
    )
    _lock_directory(compiler_input)
    compiler_event = _run_role(
        "compiler",
        _COMPILER_PROGRAM,
        compiler,
        inject_failure=_fault_role == "compiler",
    )
    _assert_directory_files(compiler_output, _COMPILER_OUTPUT_FILES)
    compiler_receipt_payload = _read_regular(
        compiler_output / "compiler_receipt.json", canonical_json=True
    )
    compiler_receipt = json.loads(compiler_receipt_payload)
    if (
        compiler_receipt.get("schema") != "ssqac-compiler-receipt-v1"
        or compiler_receipt.get("input_manifest_sha256") != compiler_manifest_hash
        or compiler_receipt.get("raw_source_sha256") != source_hash
        or compiler_receipt.get("mechanics_only") is not True
        or compiler_receipt.get("network_guard") is not True
    ):
        raise CustodyError("compiler receipt binding differs")
    for name in ("expected_result.json", "primitive_runtime.json", "sealed_algebra.json"):
        payload = _read_regular(compiler_output / name, canonical_json=True)
        reference = compiler_receipt.get("outputs", {}).get(name)
        if reference != {"sha256": _sha256_bytes(payload), "bytes": len(payload)}:
            raise CustodyError(f"compiler output receipt differs: {name}")
    sealed_bytes = _read_regular(
        compiler_output / "sealed_algebra.json", canonical_json=True
    )
    runtime_bytes = _read_regular(
        compiler_output / "primitive_runtime.json", canonical_json=True
    )
    expected_bytes = _read_regular(
        compiler_output / "expected_result.json", canonical_json=True
    )
    compiler_tree_before_deletion = {
        "input": _references(
            compiler_input, sorted(_COMPILER_INPUT_FILES), compiler
        ),
        "output": _references(
            compiler_output, sorted(_COMPILER_OUTPUT_FILES), compiler
        ),
    }

    raw_source_path = compiler_input / "raw_source.json"
    deletion_started_ns = time.time_ns()
    compiler_input.chmod(0o700)
    shutil.rmtree(compiler)
    deletion_completed_ns = time.time_ns()
    if compiler.exists() or raw_source_path.exists():
        raise CustodyError("compiler source or workspace survived deletion")

    candidate = root / "candidate"
    candidate_input = candidate / "input"
    candidate_output = candidate / "output"
    candidate_input.mkdir(parents=True, mode=0o700)
    candidate_output.mkdir(mode=0o700)
    _write_once(candidate_input / "sealed_algebra.json", sealed_bytes)
    _write_once(candidate_input / "primitive_runtime.json", runtime_bytes)
    candidate_manifest = _manifest_for(
        candidate_input,
        schema=CANDIDATE_INPUT_SCHEMA,
        payload_files=list(_CANDIDATE_DATA_FILES),
    )
    candidate_manifest_hash = _write_json_once(
        candidate_input / "manifest.json", candidate_manifest
    )
    assessor = root / "assessor"
    assessor_absent_before_candidate = not assessor.exists()
    _invoke_hook(_phase_hook, "candidate_staged", candidate_input)
    _verify_manifest(
        candidate_input,
        schema=CANDIDATE_INPUT_SCHEMA,
        payload_files=set(_CANDIDATE_DATA_FILES),
    )
    candidate_scan = _scan_candidate_inputs(candidate_input, raw_source=source_bytes)
    _assert_directory_files(candidate_output, set())
    _lock_directory(candidate_input)
    candidate_event = _run_role(
        "candidate",
        _CANDIDATE_PROGRAM,
        candidate,
        inject_failure=_fault_role == "candidate",
    )
    assessor_absent_through_candidate_exit = not assessor.exists()
    _assert_directory_files(candidate_output, _CANDIDATE_OUTPUT_FILES)
    candidate_result_bytes = _read_regular(
        candidate_output / "candidate_result.json", canonical_json=True
    )
    candidate_result = json.loads(candidate_result_bytes)
    if (
        candidate_result.get("schema") != "ssqac-candidate-mechanics-result-v1"
        or candidate_result.get("candidate_input_manifest_sha256")
        != candidate_manifest_hash
        or candidate_result.get("sealed_algebra_sha256")
        != _sha256_bytes(sealed_bytes)
        or candidate_result.get("primitive_runtime_sha256")
        != _sha256_bytes(runtime_bytes)
        or candidate_result.get("reasoning_claim") is not False
        or candidate_result.get("mechanics_only") is not True
    ):
        raise CustodyError("candidate result binding differs")
    _lock_directory(candidate_output)

    assessor_created_ns = time.time_ns()
    assessor_input = assessor / "input"
    assessor_output = assessor / "output"
    assessor_input.mkdir(parents=True, mode=0o700)
    assessor_output.mkdir(mode=0o700)
    _write_once(assessor_input / "candidate_result.json", candidate_result_bytes)
    _write_once(assessor_input / "expected_result.json", expected_bytes)
    assessor_manifest = _manifest_for(
        assessor_input,
        schema=ASSESSOR_INPUT_SCHEMA,
        payload_files=["candidate_result.json", "expected_result.json"],
    )
    assessor_manifest_hash = _write_json_once(
        assessor_input / "manifest.json", assessor_manifest
    )
    _invoke_hook(_phase_hook, "assessor_staged", assessor_input)
    _verify_manifest(
        assessor_input,
        schema=ASSESSOR_INPUT_SCHEMA,
        payload_files={"candidate_result.json", "expected_result.json"},
    )
    _assert_directory_files(assessor_output, set())
    _lock_directory(assessor_input)
    assessor_event = _run_role(
        "assessor",
        _ASSESSOR_PROGRAM,
        assessor,
        inject_failure=_fault_role == "assessor",
    )
    _assert_directory_files(assessor_output, _ASSESSOR_OUTPUT_FILES)
    assessment_payload = _read_regular(
        assessor_output / "assessment.json", canonical_json=True
    )
    assessment = json.loads(assessment_payload)
    if (
        assessment.get("schema") != "ssqac-assessment-mechanics-v1"
        or assessment.get("assessor_input_manifest_sha256")
        != assessor_manifest_hash
        or assessment.get("candidate_result_sha256")
        != _sha256_bytes(candidate_result_bytes)
        or assessment.get("expected_result_sha256") != _sha256_bytes(expected_bytes)
        or assessment.get("passed") is not True
        or assessment.get("mechanics_only") is not True
        or assessment.get("reasoning_claim") is not False
    ):
        raise CustodyError("assessment binding or result differs")
    _lock_directory(assessor_output)

    processes = [compiler_event, candidate_event, assessor_event]
    if len({event["pid"] for event in processes}) != 3:
        raise CustodyError("three custody roles did not have distinct PIDs")
    if not (
        compiler_event["parent_observed_end_ns"]
        <= deletion_started_ns
        <= deletion_completed_ns
        <= candidate_event["parent_observed_start_ns"]
        <= candidate_event["parent_observed_end_ns"]
        <= assessor_created_ns
        <= assessor_event["parent_observed_start_ns"]
    ):
        raise CustodyError("three-process launch or deletion order differs")
    if not assessor_absent_before_candidate or not assessor_absent_through_candidate_exit:
        raise CustodyError("assessor workspace existed during candidate custody")

    final_names = [
        "candidate/input/manifest.json",
        "candidate/input/primitive_runtime.json",
        "candidate/input/sealed_algebra.json",
        "candidate/output/candidate_result.json",
        "assessor/input/candidate_result.json",
        "assessor/input/expected_result.json",
        "assessor/input/manifest.json",
        "assessor/output/assessment.json",
    ]
    final_files = [
        _file_reference(root / name, relative_to=root) for name in final_names
    ]
    receipt = {
        "schema": SCHEMA,
        "classification": "cpu-only-custody-mechanics",
        "mechanics_only": True,
        "reasoning_claim": False,
        "network_access": False,
        "subprocess_program_delivery": "dedicated-python-c-program-per-role",
        "compiler": {
            "input_manifest_sha256": compiler_manifest_hash,
            "raw_source_sha256": source_hash,
            "receipt_sha256": _sha256_bytes(compiler_receipt_payload),
            "tree_before_deletion": compiler_tree_before_deletion,
            "source_unlinked_before_candidate": True,
            "workspace_unlinked_before_candidate": True,
            "deletion_started_ns": deletion_started_ns,
            "deletion_completed_ns": deletion_completed_ns,
        },
        "candidate": {
            "closed_world_manifest_sha256": candidate_manifest_hash,
            "data_payload_files": list(_CANDIDATE_DATA_FILES),
            "control_manifest_file": "manifest.json",
            "delivered_files": sorted(_CANDIDATE_INPUT_FILES),
            "read_files": candidate_event["read_files"],
            "forbidden_scan": candidate_scan,
            "assessor_workspace_absent_before_launch": True,
            "assessor_workspace_absent_through_exit": True,
            "result_sha256": _sha256_bytes(candidate_result_bytes),
        },
        "assessor": {
            "created_after_candidate_exit": True,
            "created_ns": assessor_created_ns,
            "closed_world_manifest_sha256": assessor_manifest_hash,
            "delivered_files": sorted(_ASSESSOR_INPUT_FILES),
            "read_files": assessor_event["read_files"],
            "assessment_sha256": _sha256_bytes(assessment_payload),
            "passed": True,
        },
        "processes": processes,
        "process_order": ["compiler", "candidate", "assessor"],
        "distinct_process_ids": True,
        "final_files": final_files,
        "boundary": {
            "delivered_file_closure_proven": True,
            "guarded_child_file_reads": True,
            "kernel_namespace_isolation_claimed": False,
            "promotion_eligible": False,
        },
    }
    receipt_path = root / "custody_receipt.json"
    receipt_hash = _write_json_once(receipt_path, receipt)
    verified = verify_three_process_custody(root)
    if verified != receipt:
        raise CustodyError("post-write custody verification differs")
    return CustodyRun(
        root=root,
        receipt_path=receipt_path,
        receipt_sha256=receipt_hash,
        receipt=receipt,
    )


def _walk_regular_files(root: Path) -> set[str]:
    files: set[str] = set()
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            path = current_path / name
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise CustodyError(f"invalid directory in final tree: {path}")
        for name in filenames:
            path = current_path / name
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise CustodyError(f"invalid file in final tree: {path}")
            files.add(path.relative_to(root).as_posix())
    return files


def verify_three_process_custody(root: str | Path) -> dict[str, Any]:
    """Independently replay the persisted mechanics receipt and tree closure."""

    root = Path(root).resolve()
    receipt_path = root / "custody_receipt.json"
    receipt_payload = _read_regular(receipt_path, canonical_json=True)
    receipt = json.loads(receipt_payload)
    required_keys = {
        "assessor",
        "boundary",
        "candidate",
        "classification",
        "compiler",
        "distinct_process_ids",
        "final_files",
        "mechanics_only",
        "network_access",
        "process_order",
        "processes",
        "reasoning_claim",
        "schema",
        "subprocess_program_delivery",
    }
    if set(receipt) != required_keys:
        raise CustodyError("custody receipt keys differ")
    if (
        receipt["schema"] != SCHEMA
        or receipt["classification"] != "cpu-only-custody-mechanics"
        or receipt["mechanics_only"] is not True
        or receipt["reasoning_claim"] is not False
        or receipt["network_access"] is not False
        or receipt["process_order"] != ["compiler", "candidate", "assessor"]
        or receipt["distinct_process_ids"] is not True
    ):
        raise CustodyError("custody receipt contract differs")
    if (root / "compiler_workspace").exists():
        raise CustodyError("deleted compiler workspace reappeared")
    if not (
        receipt["compiler"]["source_unlinked_before_candidate"] is True
        and receipt["compiler"]["workspace_unlinked_before_candidate"] is True
        and receipt["compiler"]["deletion_completed_ns"]
        <= receipt["processes"][1]["parent_observed_start_ns"]
    ):
        raise CustodyError("compiler deletion proof differs")

    processes = receipt["processes"]
    if (
        not isinstance(processes, list)
        or len(processes) != 3
        or [entry.get("role") for entry in processes]
        != ["compiler", "candidate", "assessor"]
        or len({entry.get("pid") for entry in processes}) != 3
        or any(
            entry.get("returncode") != 0 or entry.get("network_guard") is not True
            for entry in processes
        )
    ):
        raise CustodyError("process evidence differs")
    compiler, candidate, assessor = processes
    if not (
        compiler["parent_observed_end_ns"]
        <= receipt["compiler"]["deletion_started_ns"]
        <= receipt["compiler"]["deletion_completed_ns"]
        <= candidate["parent_observed_start_ns"]
        <= candidate["parent_observed_end_ns"]
        <= receipt["assessor"]["created_ns"]
        <= assessor["parent_observed_start_ns"]
    ):
        raise CustodyError("persisted process order differs")
    if (
        receipt["candidate"]["delivered_files"]
        != sorted(_CANDIDATE_INPUT_FILES)
        or receipt["candidate"]["data_payload_files"] != list(_CANDIDATE_DATA_FILES)
        or receipt["candidate"]["read_files"]
        != [
            "input/manifest.json",
            "input/primitive_runtime.json",
            "input/sealed_algebra.json",
        ]
        or receipt["candidate"]["assessor_workspace_absent_before_launch"] is not True
        or receipt["candidate"]["assessor_workspace_absent_through_exit"] is not True
        or receipt["assessor"]["created_after_candidate_exit"] is not True
    ):
        raise CustodyError("candidate exposure or assessor timing differs")

    _verify_manifest(
        root / "candidate" / "input",
        schema=CANDIDATE_INPUT_SCHEMA,
        payload_files=set(_CANDIDATE_DATA_FILES),
    )
    _assert_directory_files(root / "candidate" / "output", _CANDIDATE_OUTPUT_FILES)
    _verify_manifest(
        root / "assessor" / "input",
        schema=ASSESSOR_INPUT_SCHEMA,
        payload_files={"candidate_result.json", "expected_result.json"},
    )
    _assert_directory_files(root / "assessor" / "output", _ASSESSOR_OUTPUT_FILES)
    source_sentinel = b"source-only-custody-sentinel"
    candidate_scan = _scan_candidate_inputs(
        root / "candidate" / "input", raw_source=source_sentinel
    )
    if not (
        candidate_scan["forbidden_filenames_absent"]
        and candidate_scan["forbidden_content_absent"]
    ):
        raise CustodyError("candidate forbidden scan differs")

    candidate_manifest_payload = _read_regular(
        root / "candidate" / "input" / "manifest.json", canonical_json=True
    )
    candidate_result_payload = _read_regular(
        root / "candidate" / "output" / "candidate_result.json",
        canonical_json=True,
    )
    candidate_result = json.loads(candidate_result_payload)
    if (
        _sha256_bytes(candidate_manifest_payload)
        != receipt["candidate"]["closed_world_manifest_sha256"]
        or _sha256_bytes(candidate_result_payload)
        != receipt["candidate"]["result_sha256"]
        or candidate_result.get("candidate_input_manifest_sha256")
        != _sha256_bytes(candidate_manifest_payload)
        or candidate_result.get("reasoning_claim") is not False
    ):
        raise CustodyError("persisted candidate binding differs")

    assessor_manifest_payload = _read_regular(
        root / "assessor" / "input" / "manifest.json", canonical_json=True
    )
    assessment_payload = _read_regular(
        root / "assessor" / "output" / "assessment.json", canonical_json=True
    )
    assessment = json.loads(assessment_payload)
    if (
        _sha256_bytes(assessor_manifest_payload)
        != receipt["assessor"]["closed_world_manifest_sha256"]
        or _sha256_bytes(assessment_payload)
        != receipt["assessor"]["assessment_sha256"]
        or assessment.get("passed") is not True
        or assessment.get("reasoning_claim") is not False
    ):
        raise CustodyError("persisted assessment binding differs")

    expected_paths = {entry["path"] for entry in receipt["final_files"]}
    if len(expected_paths) != len(receipt["final_files"]):
        raise CustodyError("duplicate path in final manifest")
    observed_paths = _walk_regular_files(root)
    if observed_paths != expected_paths | {"custody_receipt.json"}:
        raise CustodyError("final tree file closure differs")
    for reference in receipt["final_files"]:
        path = root / reference["path"]
        observed = _file_reference(path, relative_to=root)
        if observed != reference:
            raise CustodyError(f"final file reference differs: {reference['path']}")
    boundary = receipt["boundary"]
    if boundary != {
        "delivered_file_closure_proven": True,
        "guarded_child_file_reads": True,
        "kernel_namespace_isolation_claimed": False,
        "promotion_eligible": False,
    }:
        raise CustodyError("honest mechanics boundary differs")
    return receipt


__all__ = [
    "CustodyError",
    "CustodyRun",
    "run_three_process_custody",
    "verify_three_process_custody",
]
