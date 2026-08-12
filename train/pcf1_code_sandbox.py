#!/usr/bin/env python3
"""Qualified OS-isolated execution for every PCF1 generated-code assessment."""

from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import stat
import subprocess
import tempfile
from typing import Any, Callable

from hf_product_reasoning_eval import (
    TASKS,
    _mbpp_program,
    _truncate_code,
    has_explicit_final_answer,
)

BWRAP = Path("/usr/bin/bwrap")
BWRAP_SHA256 = "eb767688b8224d8d3dbe1f8cb30ac3dff9ae8b02ff0452eaec9f94874d4e0011"
PYTHON_ROOT = Path("/lustre/fs1/home/sa305415/shohin/miniforge3")
PYTHON_EXECUTABLE = PYTHON_ROOT / "bin/python3.13"
PYTHON_SHA256 = "051a031d827eab9778e982571db754662809164c8a3ec01e9beea1e1088123e0"
MINIMAL_LIBRARY_NAMES = (
    "libz.so.1",
    "libncursesw.so.6",
    "libtinfo.so.6",
    "libtinfow.so.6",
    "libbz2.so.1.0",
    "libcrypto.so.3",
    "libexpat.so.1",
    "libffi.so.8",
    "liblzma.so.5",
    "libmpdec.so.4",
    "libpanelw.so.6",
    "libreadline.so.8",
    "libsqlite3.so.0",
    "libssl.so.3",
    "libtcl8.6.so",
    "libtk8.6.so",
    "libuuid.so.1",
)
SANDBOX_RUNTIME_TREE_SHA256 = (
    "7c6ed935cd9585475e82c64d4e591837c0c4ccd9538ad2f41adcfd3c8b71ce0d"
)
SANDBOX_RUNTIME_TREE_ENTRIES = 2_454
SANDBOX_RUNTIME_TREE_FILES = 2_342
SANDBOX_RUNTIME_TREE_DIRECTORIES = 112
SANDBOX_RUNTIME_TREE_BYTES = 112_224_747
ELF_CLOSURE_AUDIT_SHA256 = (
    "fe17b4712f24bd29cede3183316fc2c4a3b61abd93ab8f46ba2f0620258dd67c"
)
SYSTEM_LIBRARY_BINDINGS = (
    (
        Path("/usr/lib64/ld-2.28.so"),
        "/lib64/ld-linux-x86-64.so.2",
        "f54e08528da407525c1bf95d06b4b6426cedb835f018d1c596880fda4e308740",
        1_104_088,
    ),
    (
        Path("/usr/lib64/libX11.so.6.3.0"),
        "/lib64/libX11.so.6",
        "b5bd10236d0bb0c804d21501a78b790e63ae838983e67a42292220db854b52bf",
        1_344_032,
    ),
    (
        Path("/usr/lib64/libXau.so.6.0.0"),
        "/lib64/libXau.so.6",
        "e6d22245203e84096d1a5549e25d22ea111c2e03a85dc10597cee1d7276dc35d",
        16_360,
    ),
    (
        Path("/usr/lib64/libc-2.28.so"),
        "/lib64/libc.so.6",
        "7d3b8e8cf41b2d8a63841469400a64c247135f5210a95c81c6e4993e4c736ffa",
        2_164_744,
    ),
    (
        Path("/usr/lib64/libdl-2.28.so"),
        "/lib64/libdl.so.2",
        "414cca30a2b3f41d64d1b67fc987552dc2cca649d73eac3a3a5099d76363834c",
        19_128,
    ),
    (
        Path("/usr/lib64/libm-2.28.so"),
        "/lib64/libm.so.6",
        "e27d5b1be7b305214bc91e78d41c55964365cb9d06cec814e6a0b61c79d8ddba",
        1_599_096,
    ),
    (
        Path("/usr/lib64/libpthread-2.28.so"),
        "/lib64/libpthread.so.0",
        "0239064fff600cd4c9fe5bc8faaddfeb23dfd8444c8882ed0c95aa2edca5b985",
        149_936,
    ),
    (
        Path("/usr/lib64/libutil-2.28.so"),
        "/lib64/libutil.so.1",
        "e3f9a9ad55b94be376da0b7e9360821045f0a9d0339d95e615ace176176e0fe4",
        17_032,
    ),
    (
        Path("/usr/lib64/libxcb.so.1.1.0"),
        "/lib64/libxcb.so.1",
        "fcb2fa42338d24f9f67ee55351cec2f57dcf34b63fe0bb9e709c66543db6c5da",
        170_216,
    ),
)
SAFE_CANDIDATE_IMPORTS = (
    "bisect",
    "collections",
    "collections.abc",
    "decimal",
    "fractions",
    "functools",
    "heapq",
    "itertools",
    "math",
    "random",
    "re",
    "statistics",
    "string",
)
FORBIDDEN_CANDIDATE_NAMES = (
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "exit",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "quit",
    "setattr",
    "vars",
)
ALLOWED_PRIVATE_CANDIDATE_ATTRIBUTES = (
    "__call__",
    "__contains__",
    "__eq__",
    "__ge__",
    "__getitem__",
    "__gt__",
    "__hash__",
    "__init__",
    "__iter__",
    "__le__",
    "__len__",
    "__ne__",
    "__next__",
    "__repr__",
    "__setitem__",
    "__str__",
)
FORBIDDEN_CANDIDATE_ATTRIBUTE_PREFIXES = (
    "ag_",
    "cr_",
    "f_",
    "gi_",
    "tb_",
)
FORBIDDEN_CANDIDATE_ATTRIBUTES = (
    "Formatter",
    "Random",
    "SystemRandom",
    "attrgetter",
    "bltns",
    "builtins",
    "copyreg",
    "enum",
    "importlib",
    "io",
    "operator",
    "os",
    "posix",
    "subprocess",
    "sys",
    "methodcaller",
    "seed",
)
CANDIDATE_POLICY = {
    "schema": "shohin-pcf1-mbpp-candidate-policy-v1",
    "safe_import_roots": list(SAFE_CANDIDATE_IMPORTS),
    "forbidden_names": list(FORBIDDEN_CANDIDATE_NAMES),
    "allowed_private_attributes": list(ALLOWED_PRIVATE_CANDIDATE_ATTRIBUTES),
    "forbidden_attribute_prefixes": list(FORBIDDEN_CANDIDATE_ATTRIBUTE_PREFIXES),
    "forbidden_attributes": list(FORBIDDEN_CANDIDATE_ATTRIBUTES),
    "forbid_dunder_identifiers": True,
    "forbid_private_match_attributes": True,
    "forbid_supervisor_names_as_attributes": True,
    "safe_import_reachability_depth": 5,
    "random_entropy_reentry": "seed_random_systemrandom_forbidden",
    "trusted_completion_attestation": (
        "reserved_exit_code_after_capability_filter_and_phase_separation"
    ),
}
CANDIDATE_POLICY_SHA256 = hashlib.sha256(
    json.dumps(CANDIDATE_POLICY, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
TRUSTED_COMPLETION_EXIT_CODE = 73
CANDIDATE_FAILURE_EXIT_CODE = 74
INFRASTRUCTURE_FAILURE_EXIT_CODE = 75
TEST_FAILURE_EXIT_CODE = 76
SETUP_FAILURE_EXIT_CODE = 77
POLICY_REJECTION_EXIT_CODE = 78
RESOURCE_LIMIT_EXIT_CODE = 79
RESOURCE_PROBE_TIMEOUT_SECONDS = 2.0
CANDIDATE_RANDOM_SEED = 2026080816
MAX_CANDIDATE_BYTES = 1 << 20
MAX_ASSESSOR_TRANSPORT_BYTES = 1 << 20
MEMFD_ABI = {
    "backend": "libc.memfd_create",
    "mfd_cloexec": 0x1,
    "mfd_allow_sealing": 0x2,
    "f_add_seals": 1033,
    "f_get_seals": 1034,
    "f_seal_seal": 0x1,
    "f_seal_shrink": 0x2,
    "f_seal_grow": 0x4,
    "f_seal_write": 0x8,
    "required_seals": 0xF,
}
EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR = {
    "executable": "/opt/python/bin/python3.13",
    "prefix": "/opt/python",
    "base_prefix": "/opt/python",
    "sys_path": [
        "/opt/python/lib/python313.zip",
        "/opt/python/lib/python3.13",
        "/opt/python/lib/python3.13/lib-dynload",
    ],
    "flags": {
        "debug": 0,
        "inspect": 0,
        "interactive": 0,
        "optimize": 0,
        "dont_write_bytecode": 1,
        "no_user_site": 1,
        "no_site": 1,
        "ignore_environment": 0,
        "verbose": 0,
        "bytes_warning": 0,
        "quiet": 0,
        "hash_randomization": 0,
        "isolated": 0,
        "dev_mode": False,
        "utf8_mode": 1,
        "safe_path": True,
    },
    "hash_probe": -1545367155142879260,
    "set_order": [
        "pcf1-beta",
        "pcf1-gamma",
        "pcf1-alpha",
        "pcf1-delta",
    ],
}
EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR_SHA256 = hashlib.sha256(
    json.dumps(
        EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()
SANDBOX_PROBES = frozenset(
    {
        "aggregate_file_creation_policy_blocked",
        "address_space_bounded",
        "assessor_sentinel_hidden",
        "assessor_transport_closed",
        "candidate_is_pid_1",
        "candidate_mount_excludes_assessor",
        "candidate_policy_escape_blocked",
        "candidate_status_fd_closed",
        "cpu_bounded",
        "elf_projection_exact",
        "environment_exact",
        "failed_official_assertion_blocked",
        "filesystem_projection_exact",
        "flood_bounded",
        "fork_blocked",
        "known_pass",
        "network_unreachable",
        "os_exit_bypass_blocked",
        "outer_wall_timeout_is_infrastructure",
        "parent_environment_hidden",
        "parent_proc_hidden",
        "private_tmp_writable",
        "read_only_inputs_immutable",
        "root_filesystem_read_only",
        "random_entropy_reentry_blocked",
        "resource_limit_exit_code_forgery_blocked",
        "python_hash_seed_effective",
        "python_runtime_descriptor_cross_process",
        "python_runtime_descriptor_exact",
        "safe_import_namespace_reachability",
        "site_packages_mask_read_only",
        "random_seed_deterministic",
        "status_marker_forgery_blocked",
        "subprocess_blocked",
        "symlink_traversal_blocked",
        "system_exit_bypass_blocked",
        "post_candidate_setup_failure_is_scientific",
        "policy_rejection_runs_in_sandbox",
        "setup_compile_failure_is_infrastructure",
        "trusted_completion_exit_attested",
    }
)
SAFE_IMPORT_CAPABILITIES_PROBE_SOURCE = f"""import builtins
import importlib
import operator
import types

safe_modules = {list(SAFE_CANDIDATE_IMPORTS)!r}
blocked_attributes = {sorted(set(FORBIDDEN_CANDIDATE_ATTRIBUTES) | set(FORBIDDEN_CANDIDATE_NAMES))!r}
forbidden_builtin_names = {list(FORBIDDEN_CANDIDATE_NAMES)!r}
forbidden_builtin_capabilities = tuple(
    value
    for name, value in vars(builtins).items()
    if name in forbidden_builtin_names
)
dangerous_operator_names = {{"attrgetter", "methodcaller"}}
dangerous_operator_capabilities = (operator.attrgetter, operator.methodcaller)
dangerous_origins = {{"builtins", "importlib", "io", "os", "posix", "subprocess", "sys"}}
dangerous_module_roots = dangerous_origins | {{"operator"}}

def module_allowed(value):
    module_root = value.__name__.split(".", 1)[0].lstrip("_")
    return module_root not in dangerous_module_roots

def capability_allowed(public_name, value):
    if public_name in forbidden_builtin_names:
        return False
    if any(value is forbidden for forbidden in forbidden_builtin_capabilities):
        return False
    raw_origin = getattr(value, "__module__", "")
    origin = (
        raw_origin.split(".", 1)[0].lstrip("_")
        if isinstance(raw_origin, str)
        else ""
    )
    if origin == "builtins":
        return True
    if origin == "operator":
        return (
            public_name not in dangerous_operator_names
            and all(value is not forbidden for forbidden in dangerous_operator_capabilities)
        )
    return origin not in dangerous_origins

assert capability_allowed("EllipsisType", type(Ellipsis))
assert not capability_allowed("harmless_alias", open)
assert not capability_allowed("open", type(Ellipsis))
assert capability_allowed("itemgetter", operator.itemgetter)
assert not capability_allowed("harmless_alias", operator.attrgetter)
assert not capability_allowed("harmless_alias", operator.methodcaller)
assert not module_allowed(types.ModuleType("_io"))
assert not module_allowed(operator)
statistics_module = importlib.import_module("statistics")
if hasattr(statistics_module, "itemgetter"):
    assert statistics_module.itemgetter is operator.itemgetter
    assert capability_allowed("itemgetter", statistics_module.itemgetter)
queue = []
for module_name in safe_modules:
    module = importlib.import_module(module_name)
    queue.append((module_name, module, 0))
seen = set()
visited = 0
while queue:
    object_path, exposed, depth = queue.pop(0)
    if id(exposed) in seen:
        continue
    seen.add(id(exposed))
    visited += 1
    for public_name, value in vars(exposed).items():
        if public_name.startswith("_") or public_name in blocked_attributes:
            continue
        if isinstance(value, types.ModuleType):
            assert module_allowed(value), object_path + "." + public_name
        assert capability_allowed(public_name, value), object_path + "." + public_name
        if depth < 5 and isinstance(value, (types.ModuleType, type)):
            queue.append((object_path + "." + public_name, value, depth + 1))
assert visited >= len(safe_modules)
"""

# This bootstrap is sandboxed PID 1. /candidate.py is the raw model program and
# the only assessment mount. A separately sealed assessor memfd arrives as fd 0;
# it is bounded-read, compiled, discarded, and replaced with /dev/null before
# READY or model execution. Candidate stdout/stderr go to /dev/null. Only
# trusted fall-through after all official tests emits the reserved success
# status; trusted-harness/bootstrap failures use a distinct infrastructure code.
BOOTSTRAP_SOURCE_TEMPLATE = r"""import hashlib
import json
import os
import random
import resource
import signal
import sys

EXPECTED_RUNTIME_DESCRIPTOR = __EXPECTED_RUNTIME_DESCRIPTOR__
MAX_CANDIDATE_BYTES = 1048576
MAX_ASSESSOR_BYTES = 1048576
RESOURCE_LIMIT_EXIT_CODE = __RESOURCE_LIMIT_EXIT_CODE__

def cpu_limit_handler(_signum, _frame):
    os._exit(RESOURCE_LIMIT_EXIT_CODE)

def bounded_read(fd, limit):
    chunks = []
    size = 0
    while True:
        block = os.read(fd, min(65536, limit + 1 - size))
        if not block:
            return b"".join(chunks)
        chunks.append(block)
        size += len(block)
        if size > limit:
            raise ValueError("anonymous input exceeds bound")

def runtime_descriptor():
    flag_names = tuple(EXPECTED_RUNTIME_DESCRIPTOR["flags"])
    return {
        "executable": sys.executable,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "sys_path": list(sys.path),
        "flags": {name: getattr(sys.flags, name) for name in flag_names},
        "hash_probe": hash("PCF1_HASH_PROBE"),
        "set_order": list(set(("pcf1-alpha", "pcf1-beta", "pcf1-gamma", "pcf1-delta"))),
    }

def main():
    status_fd = os.dup(1)
    null_output_fd = os.open("/dev/null", os.O_WRONLY)
    os.dup2(null_output_fd, 1)
    os.dup2(null_output_fd, 2)
    os.close(null_output_fd)
    resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    try:
        assessor_raw = bounded_read(0, MAX_ASSESSOR_BYTES)
        os.close(0)
        null_input_fd = os.open("/dev/null", os.O_RDONLY)
        if null_input_fd != 0:
            os.dup2(null_input_fd, 0)
            os.close(null_input_fd)
        payload = json.loads(assessor_raw)
        with open("/candidate.py", "rb") as candidate_file:
            candidate_raw = candidate_file.read(MAX_CANDIDATE_BYTES + 1)
        if len(candidate_raw) > MAX_CANDIDATE_BYTES:
            raise ValueError("candidate source exceeds bound")
        candidate_source = candidate_raw.decode("utf-8")
        expected_fields = {
            "assessment_mode",
            "candidate_policy_failure",
            "candidate_policy_passed",
            "candidate_policy_sha256",
            "candidate_source_sha256",
            "schema",
            "seed",
            "setup_source",
            "tests_source",
        }
        if set(payload) != expected_fields:
            raise ValueError("assessor transport fields differ")
        if (
            payload["schema"] != "shohin-pcf1-sandbox-assessor-transport-v1"
            or payload["seed"] != 2026080816
            or payload["candidate_policy_sha256"] != __CANDIDATE_POLICY_SHA256__
            or payload["assessment_mode"]
            not in {
                "candidate",
                "trusted_probe",
                "trusted_reference",
                "trusted_setup_compile",
            }
            or not isinstance(payload["candidate_policy_passed"], bool)
            or not isinstance(payload["candidate_policy_failure"], str)
            or not isinstance(payload["setup_source"], str)
            or not isinstance(payload["tests_source"], str)
            or payload["candidate_source_sha256"] != hashlib.sha256(candidate_raw).hexdigest()
            or (
                payload["assessment_mode"] == "trusted_probe"
                and (
                    payload["candidate_policy_passed"] is not True
                    or payload["candidate_policy_failure"] != "not_applicable_trusted_probe"
                )
            )
            or (
                payload["assessment_mode"] == "trusted_reference"
                and (
                    payload["candidate_policy_passed"] is not True
                    or payload["candidate_policy_failure"]
                    != "not_applicable_trusted_reference"
                )
            )
            or (
                payload["assessment_mode"] == "trusted_setup_compile"
                and (
                    payload["candidate_policy_passed"] is not True
                    or payload["candidate_policy_failure"]
                    != "not_applicable_trusted_setup_compile"
                )
            )
            or (
                payload["assessment_mode"] == "candidate"
                and (
                    payload["candidate_policy_passed"]
                    != (payload["candidate_policy_failure"] == "accepted")
                    or payload["candidate_policy_failure"]
                    not in {
                        "accepted",
                        "forbidden_name",
                        "import",
                        "match_introspection",
                        "private_attribute",
                        "syntax",
                    }
                )
            )
        ):
            raise ValueError("assessor transport contract differs")
        setup_code = compile(payload["setup_source"], "/trusted-setup.py", "exec")
        tests_code = compile(payload["tests_source"], "/official-tests.py", "exec")
        policy_passed = payload["candidate_policy_passed"]
        assessment_mode = payload["assessment_mode"]
        observed_runtime = runtime_descriptor()
        if observed_runtime != EXPECTED_RUNTIME_DESCRIPTOR:
            raise RuntimeError("Python runtime descriptor differs")
        payload.clear()
        assessor_raw = b""
        candidate_raw = b""
    except BaseException:
        os._exit(75)
    runtime_line = json.dumps(observed_runtime, sort_keys=True, separators=(",", ":"))
    os.write(status_fd, ("PCF1_RUNTIME_DESCRIPTOR " + runtime_line + "\n").encode())
    os.write(status_fd, b"PCF1_PYTHON_READY\n")
    os.close(status_fd)
    if assessment_mode == "trusted_setup_compile":
        os._exit(73)
    if not policy_passed:
        os._exit(78)
    try:
        signal.signal(signal.SIGXCPU, cpu_limit_handler)
    except BaseException:
        os._exit(75)
    try:
        candidate_code = compile(candidate_source, "/candidate.py", "exec")
    except BaseException:
        os._exit(74)
    namespace = {"__name__": "__main__", "__file__": "/candidate.py"}
    random.seed(2026080816)
    try:
        exec(candidate_code, namespace)
    except BaseException:
        os._exit(74)
    try:
        exec(setup_code, namespace)
    except BaseException:
        os._exit(77)
    try:
        exec(tests_code, namespace)
    except BaseException:
        os._exit(76)
    os._exit(73)

main()
"""


def _render_bootstrap_source(runtime_descriptor: dict[str, Any]) -> str:
    return (
        BOOTSTRAP_SOURCE_TEMPLATE.replace(
            "__EXPECTED_RUNTIME_DESCRIPTOR__", repr(runtime_descriptor)
        )
        .replace("__CANDIDATE_POLICY_SHA256__", repr(CANDIDATE_POLICY_SHA256))
        .replace("__RESOURCE_LIMIT_EXIT_CODE__", repr(RESOURCE_LIMIT_EXIT_CODE))
    )


BOOTSTRAP_SOURCE = _render_bootstrap_source(EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR)
BOOTSTRAP_SHA256 = hashlib.sha256(BOOTSTRAP_SOURCE.encode()).hexdigest()
SANDBOX_CONFIG = {
    "bwrap": str(BWRAP),
    "bwrap_sha256": BWRAP_SHA256,
    "python_root": str(PYTHON_ROOT),
    "python_executable": str(PYTHON_EXECUTABLE),
    "python_sha256": PYTHON_SHA256,
    "unshare": "all",
    "network": "none",
    "proc": "child_pid_namespace_only",
    "candidate_mount": "raw-sealed-ro-bind-data:/candidate.py",
    "assessor_transport": (
        "sealed-read-only-memfd-on-stdin; bounded-read-compile-drop; "
        "fd0-replaced-with-dev-null-before-ready"
    ),
    "candidate_process": "direct_pid_1",
    "bootstrap_sha256": BOOTSTRAP_SHA256,
    "candidate_policy_sha256": CANDIDATE_POLICY_SHA256,
    "trusted_completion_exit_code": TRUSTED_COMPLETION_EXIT_CODE,
    "candidate_failure_exit_code": CANDIDATE_FAILURE_EXIT_CODE,
    "infrastructure_failure_exit_code": INFRASTRUCTURE_FAILURE_EXIT_CODE,
    "test_failure_exit_code": TEST_FAILURE_EXIT_CODE,
    "setup_failure_exit_code": SETUP_FAILURE_EXIT_CODE,
    "policy_rejection_exit_code": POLICY_REJECTION_EXIT_CODE,
    "resource_limit_exit_code": RESOURCE_LIMIT_EXIT_CODE,
    "resource_probe_timeout_seconds": RESOURCE_PROBE_TIMEOUT_SECONDS,
    "candidate_random_seed": CANDIDATE_RANDOM_SEED,
    "successful_code_assessment": "reserved_exit_after_official_tests",
    "sandbox_runtime_tree_sha256": SANDBOX_RUNTIME_TREE_SHA256,
    "read_only_bindings": [
        "python_root/bin/python3.13:/opt/python/bin/python3.13",
        "python_root/lib/python3.13:/opt/python/lib/python3.13",
        *[
            f"python_root/lib/{name}:/opt/python/lib/{name}"
            for name in MINIMAL_LIBRARY_NAMES
        ],
        *[
            f"{source}:{destination}"
            for source, destination, _, _ in SYSTEM_LIBRARY_BINDINGS
        ],
        "remount-ro:/opt/python/lib/python3.13/site-packages",
        "remount-ro:/",
    ],
    "root_filesystem_read_only": True,
    "masked_site_packages_read_only": True,
    "writable_bindings": ["tmpfs:/tmp"],
    "temporary_storage_control": (
        "private tmpfs; candidate file-creation capabilities denied; "
        "RLIMIT_FSIZE and outer wall limit retained; bubblewrap 0.4 has no "
        "tmpfs-size option"
    ),
    "safe_import_probe_scope": (
        "public module/type namespace edges through depth 5; callable return "
        "surfaces constrain forbidden builtin and operator capabilities by "
        "public name and object identity plus frozen-reference execution"
    ),
    "environment": {
        "HOME": "/tmp",
        "PATH": "/opt/python/bin",
        "PYTHONHOME": "/opt/python",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "TMPDIR": "/tmp",
        "LANG": "C.UTF-8",
        "PWD": "/tmp",
    },
    "python_flags": ["-P", "-s", "-S", "-B"],
    "python_runtime_descriptor": EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR,
    "python_runtime_descriptor_sha256": (EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR_SHA256),
    "max_candidate_bytes": MAX_CANDIDATE_BYTES,
    "max_assessor_transport_bytes": MAX_ASSESSOR_TRANSPORT_BYTES,
    "memfd_abi": MEMFD_ABI,
    "rlimit_as_bytes": 1 << 30,
    "rlimit_fsize_bytes": 1 << 20,
    "rlimit_nproc": 1,
    "rlimit_nofile": 64,
    "site_packages_visible": False,
}
SANDBOX_CONFIG_SHA256 = hashlib.sha256(
    json.dumps(SANDBOX_CONFIG, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
_ALLOCATION_PROBE_SHA256: str | None = None
_QUALIFIED_SETUP_RECEIPTS: dict[str, str] = {}


class PCF1SandboxError(RuntimeError):
    """The qualified bwrap host or isolated code execution differs."""


def expected_system_library_members() -> list[dict[str, Any]]:
    return [
        {
            "source": str(source),
            "destination": destination,
            "sha256": digest,
            "size": size,
        }
        for source, destination, digest, size in SYSTEM_LIBRARY_BINDINGS
    ]


def validate_sandbox_receipt_payload(receipt: dict[str, Any]) -> None:
    """Validate the exact deterministic allocation-qualification receipt."""

    probes = receipt.get("probe_results")
    expected_keys = {
        "schema",
        "status",
        "bwrap_path",
        "bwrap_sha256",
        "bwrap_version",
        "python_executable",
        "python_sha256",
        "sandbox_config_sha256",
        "candidate_policy_sha256",
        "trusted_completion_exit_code",
        "candidate_failure_exit_code",
        "infrastructure_failure_exit_code",
        "test_failure_exit_code",
        "setup_failure_exit_code",
        "policy_rejection_exit_code",
        "resource_limit_exit_code",
        "candidate_random_seed",
        "python_runtime_descriptor",
        "python_runtime_descriptor_sha256",
        "memfd_abi",
        "sandbox_runtime_tree_sha256",
        "sandbox_runtime_tree_entries",
        "sandbox_runtime_tree_files",
        "sandbox_runtime_tree_directories",
        "sandbox_runtime_tree_bytes",
        "elf_closure_audit_sha256",
        "system_library_members",
        "clear_environment",
        "network_namespace",
        "candidate_read_only",
        "candidate_direct_pid_1",
        "site_packages_visible",
        "probe_results",
        "probe_sha256",
        "sandbox_isolation_passed",
    }
    canonical_probe = (
        hashlib.sha256(
            json.dumps(probes, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if isinstance(probes, dict)
        else None
    )
    if (
        set(receipt) != expected_keys
        or receipt.get("schema") != "shohin-pcf1-code-sandbox-receipt-v1"
        or receipt.get("status") != "pass"
        or receipt.get("bwrap_path") != str(BWRAP)
        or receipt.get("bwrap_sha256") != BWRAP_SHA256
        or receipt.get("bwrap_version") != "bubblewrap 0.4.0"
        or receipt.get("python_executable") != str(PYTHON_EXECUTABLE)
        or receipt.get("python_sha256") != PYTHON_SHA256
        or receipt.get("sandbox_config_sha256") != SANDBOX_CONFIG_SHA256
        or receipt.get("candidate_policy_sha256") != CANDIDATE_POLICY_SHA256
        or receipt.get("trusted_completion_exit_code") != TRUSTED_COMPLETION_EXIT_CODE
        or receipt.get("candidate_failure_exit_code") != CANDIDATE_FAILURE_EXIT_CODE
        or receipt.get("infrastructure_failure_exit_code")
        != INFRASTRUCTURE_FAILURE_EXIT_CODE
        or receipt.get("test_failure_exit_code") != TEST_FAILURE_EXIT_CODE
        or receipt.get("setup_failure_exit_code") != SETUP_FAILURE_EXIT_CODE
        or receipt.get("policy_rejection_exit_code") != POLICY_REJECTION_EXIT_CODE
        or receipt.get("resource_limit_exit_code") != RESOURCE_LIMIT_EXIT_CODE
        or receipt.get("candidate_random_seed") != CANDIDATE_RANDOM_SEED
        or receipt.get("python_runtime_descriptor")
        != EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR
        or receipt.get("python_runtime_descriptor_sha256")
        != EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR_SHA256
        or receipt.get("memfd_abi") != MEMFD_ABI
        or receipt.get("sandbox_runtime_tree_sha256") != SANDBOX_RUNTIME_TREE_SHA256
        or receipt.get("sandbox_runtime_tree_entries") != SANDBOX_RUNTIME_TREE_ENTRIES
        or receipt.get("sandbox_runtime_tree_files") != SANDBOX_RUNTIME_TREE_FILES
        or receipt.get("sandbox_runtime_tree_directories")
        != SANDBOX_RUNTIME_TREE_DIRECTORIES
        or receipt.get("sandbox_runtime_tree_bytes") != SANDBOX_RUNTIME_TREE_BYTES
        or receipt.get("elf_closure_audit_sha256") != ELF_CLOSURE_AUDIT_SHA256
        or receipt.get("system_library_members") != expected_system_library_members()
        or receipt.get("clear_environment") is not True
        or receipt.get("network_namespace") != "isolated"
        or receipt.get("candidate_read_only") is not True
        or receipt.get("candidate_direct_pid_1") is not True
        or receipt.get("site_packages_visible") is not False
        or not isinstance(probes, dict)
        or set(probes) != SANDBOX_PROBES
        or any(value is not True for value in probes.values())
        or receipt.get("probe_sha256") != canonical_probe
        or receipt.get("sandbox_isolation_passed") is not True
    ):
        raise PCF1SandboxError("PCF1 sandbox allocation receipt differs")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_sandbox_host() -> None:
    if (
        BWRAP.is_symlink()
        or not BWRAP.is_file()
        or sha256_file(BWRAP) != BWRAP_SHA256
        or PYTHON_EXECUTABLE.is_symlink()
        or not PYTHON_EXECUTABLE.is_file()
        or sha256_file(PYTHON_EXECUTABLE) != PYTHON_SHA256
    ):
        raise PCF1SandboxError("PCF1 qualified sandbox host differs")
    version = subprocess.run(
        [str(BWRAP), "--version"],
        env={},
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if version.returncode != 0 or version.stdout.strip() != "bubblewrap 0.4.0":
        raise PCF1SandboxError("PCF1 qualified bubblewrap version differs")

    _validate_minimal_runtime_tree()
    _validate_system_library_closure()


def _validate_system_library_closure() -> None:
    """Require every host ELF member projected into the sandbox exactly."""

    for source, destination, digest, size in SYSTEM_LIBRARY_BINDINGS:
        if (
            source.is_symlink()
            or not source.is_file()
            or source.stat().st_size != size
            or sha256_file(source) != digest
            or not destination.startswith("/lib64/")
        ):
            raise PCF1SandboxError("PCF1 sandbox ELF closure differs")


def _validate_minimal_runtime_tree() -> None:
    """Replay the canonical virtual minimal-runtime tree receipt."""

    virtual_files: dict[str, Path] = {"bin/python3.13": PYTHON_EXECUTABLE}
    stdlib = PYTHON_ROOT / "lib/python3.13"
    if not stdlib.is_dir() or stdlib.is_symlink():
        raise PCF1SandboxError("PCF1 sandbox stdlib root differs")
    virtual_directories: set[str] = {"bin", "lib", "lib/python3.13"}
    for path in sorted(stdlib.rglob("*")):
        relative = path.relative_to(PYTHON_ROOT).as_posix()
        if relative == "lib/python3.13/site-packages" or relative.startswith(
            "lib/python3.13/site-packages/"
        ):
            continue
        mode = path.lstat().st_mode
        if stat.S_ISREG(mode) or stat.S_ISLNK(mode) and path.resolve().is_file():
            virtual_files[relative] = path.resolve()
        elif stat.S_ISDIR(mode):
            virtual_directories.add(relative)
        else:
            raise PCF1SandboxError("PCF1 sandbox runtime has a special member")
    for name in MINIMAL_LIBRARY_NAMES:
        path = PYTHON_ROOT / "lib" / name
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or resolved.is_symlink():
            raise PCF1SandboxError("PCF1 sandbox library differs")
        virtual_files[f"lib/{name}"] = resolved

    for relative in virtual_files:
        parent = Path(relative).parent
        while parent != Path("."):
            virtual_directories.add(parent.as_posix())
            parent = parent.parent
    rows: list[dict[str, Any]] = []
    byte_count = 0
    for relative, path in virtual_files.items():
        size = path.stat().st_size
        byte_count += size
        rows.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size": size,
                "type": "file",
            }
        )
    rows.extend(
        {"path": relative, "type": "directory"} for relative in virtual_directories
    )
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item["path"])):
        digest.update(
            (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
    observed = {
        "sha256": digest.hexdigest(),
        "entries": len(rows),
        "files": len(virtual_files),
        "directories": len(virtual_directories),
        "file_bytes": byte_count,
    }
    expected = {
        "sha256": SANDBOX_RUNTIME_TREE_SHA256,
        "entries": SANDBOX_RUNTIME_TREE_ENTRIES,
        "files": SANDBOX_RUNTIME_TREE_FILES,
        "directories": SANDBOX_RUNTIME_TREE_DIRECTORIES,
        "file_bytes": SANDBOX_RUNTIME_TREE_BYTES,
    }
    if observed != expected:
        raise PCF1SandboxError("PCF1 minimal sandbox runtime tree differs")


def _limits(timeout_seconds: float) -> Callable[[], None]:
    def apply() -> None:
        cpu = max(1, int(math.ceil(timeout_seconds)))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(
            resource.RLIMIT_AS,
            (SANDBOX_CONFIG["rlimit_as_bytes"], SANDBOX_CONFIG["rlimit_as_bytes"]),
        )
        resource.setrlimit(
            resource.RLIMIT_FSIZE,
            (
                SANDBOX_CONFIG["rlimit_fsize_bytes"],
                SANDBOX_CONFIG["rlimit_fsize_bytes"],
            ),
        )

    return apply


def sandbox_command(
    candidate_fd: int,
    info_fd: int | None = None,
) -> list[str]:
    command = [
        str(BWRAP),
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--as-pid-1",
        "--dir",
        "/opt",
        "--dir",
        "/opt/python",
        "--dir",
        "/opt/python/bin",
        "--dir",
        "/opt/python/lib",
        "--ro-bind",
        str(PYTHON_EXECUTABLE.resolve()),
        "/opt/python/bin/python3.13",
        "--ro-bind",
        str((PYTHON_ROOT / "lib/python3.13").resolve()),
        "/opt/python/lib/python3.13",
        "--tmpfs",
        "/opt/python/lib/python3.13/site-packages",
    ]
    for name in MINIMAL_LIBRARY_NAMES:
        command.extend(
            [
                "--ro-bind",
                str((PYTHON_ROOT / "lib" / name).resolve()),
                f"/opt/python/lib/{name}",
            ]
        )
    command.extend(["--dir", "/lib64"])
    for source, destination, _, _ in SYSTEM_LIBRARY_BINDINGS:
        command.extend(["--ro-bind", str(source), destination])
    command.extend(
        [
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--setenv",
            "PATH",
            "/opt/python/bin",
            "--setenv",
            "HOME",
            "/tmp",
            "--setenv",
            "TMPDIR",
            "/tmp",
            "--setenv",
            "PYTHONHOME",
            "/opt/python",
            "--setenv",
            "LANG",
            "C.UTF-8",
            "--setenv",
            "PYTHONDONTWRITEBYTECODE",
            "1",
            "--setenv",
            "PYTHONHASHSEED",
            "0",
            "--setenv",
            "PWD",
            "/tmp",
            "--ro-bind-data",
            str(candidate_fd),
            "/candidate.py",
            "--remount-ro",
            "/opt/python/lib/python3.13/site-packages",
            "--remount-ro",
            "/",
            "--chdir",
            "/tmp",
            "--",
            "/opt/python/bin/python3.13",
            "-P",
            "-s",
            "-S",
            "-B",
            "-c",
            BOOTSTRAP_SOURCE,
        ]
    )
    if info_fd is not None:
        command[1:1] = ["--info-fd", str(info_fd)]
    return command


def _diagnostic(value: str | bytes | None, limit: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-limit:]


def _mbpp_candidate_source(completion: str) -> str:
    return _truncate_code(
        completion,
        ("\n[DONE]", "\nQuestion:", "\nif __name__", "\n>>>"),
    )


def _assessor_transport_payload(
    candidate_source: str,
    setup_source: str,
    tests_source: str,
    *,
    trusted_probe: bool,
    trusted_reference: bool = False,
    trusted_setup_compile: bool = False,
) -> bytes:
    if not all(
        isinstance(value, str)
        for value in (candidate_source, setup_source, tests_source)
    ):
        raise PCF1SandboxError("PCF1 sandbox assessment source is not text")
    if sum(map(int, (trusted_probe, trusted_reference, trusted_setup_compile))) > 1:
        raise PCF1SandboxError("PCF1 trusted assessment mode is ambiguous")
    if trusted_probe:
        policy_passed, policy_reason = True, "not_applicable_trusted_probe"
        assessment_mode = "trusted_probe"
    elif trusted_reference:
        policy_passed, policy_reason = True, "not_applicable_trusted_reference"
        assessment_mode = "trusted_reference"
    elif trusted_setup_compile:
        policy_passed = True
        policy_reason = "not_applicable_trusted_setup_compile"
        assessment_mode = "trusted_setup_compile"
    else:
        policy_passed, policy_reason = validate_mbpp_candidate(candidate_source)
        assessment_mode = "candidate"
    payload = {
        "schema": "shohin-pcf1-sandbox-assessor-transport-v1",
        "seed": CANDIDATE_RANDOM_SEED,
        "assessment_mode": assessment_mode,
        "candidate_policy_sha256": CANDIDATE_POLICY_SHA256,
        "candidate_policy_passed": policy_passed,
        "candidate_policy_failure": policy_reason,
        "candidate_source_sha256": hashlib.sha256(
            candidate_source.encode()
        ).hexdigest(),
        "setup_source": setup_source,
        "tests_source": tests_source,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    if len(encoded) > MAX_ASSESSOR_TRANSPORT_BYTES:
        raise PCF1SandboxError("PCF1 assessor transport exceeds bound")
    return encoded


def _libc_memfd_create(name: str) -> int:
    """Call the pinned Linux libc ABI absent from the qualified Python build."""

    if not isinstance(name, str) or not name or "\x00" in name:
        raise PCF1SandboxError("PCF1 anonymous input name differs")
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        create = libc.memfd_create
        create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
        create.restype = ctypes.c_int
        ctypes.set_errno(0)
        fd = create(
            name.encode("ascii"),
            MEMFD_ABI["mfd_cloexec"] | MEMFD_ABI["mfd_allow_sealing"],
        )
        if fd < 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number))
    except (AttributeError, ImportError, OSError, UnicodeEncodeError) as error:
        raise PCF1SandboxError("PCF1 libc memfd_create admission failed") from error
    return int(fd)


def _sealed_read_only_memfd(name: str, content: bytes) -> int:
    """Return an anonymous, immutable, read-only descriptor at offset zero."""

    writable_fd = _libc_memfd_create(name)
    required_seals = MEMFD_ABI["required_seals"]
    try:
        offset = 0
        while offset < len(content):
            offset += os.write(writable_fd, content[offset:])
        fcntl.fcntl(writable_fd, MEMFD_ABI["f_add_seals"], required_seals)
        read_only_fd = os.open(
            f"/proc/self/fd/{writable_fd}", os.O_RDONLY | os.O_CLOEXEC
        )
    except OSError as error:
        raise PCF1SandboxError("PCF1 anonymous input sealing failed") from error
    finally:
        os.close(writable_fd)
    try:
        if (
            fcntl.fcntl(read_only_fd, MEMFD_ABI["f_get_seals"]) != required_seals
            or fcntl.fcntl(read_only_fd, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY
        ):
            raise PCF1SandboxError("PCF1 anonymous input is not sealed read-only")
        os.lseek(read_only_fd, 0, os.SEEK_SET)
    except (OSError, PCF1SandboxError) as error:
        os.close(read_only_fd)
        if isinstance(error, PCF1SandboxError):
            raise
        raise PCF1SandboxError("PCF1 anonymous input validation failed") from error
    return read_only_fd


def validate_mbpp_candidate(completion: str) -> tuple[bool, str]:
    """Reject process/FD/introspection capabilities before candidate launch.

    A policy rejection is a scientific malformed/incorrect outcome, not a
    sandbox infrastructure failure.  The official assessor tests are appended
    only after this check and are not constrained by the model-code policy.
    """

    source = _mbpp_candidate_source(completion)
    try:
        tree = ast.parse(source, filename="/candidate-model.py", mode="exec")
    except (SyntaxError, ValueError, TypeError):
        return False, "syntax"
    forbidden_names = set(FORBIDDEN_CANDIDATE_NAMES)
    forbidden_attributes = set(FORBIDDEN_CANDIDATE_ATTRIBUTES) | forbidden_names
    safe_imports = set(SAFE_CANDIDATE_IMPORTS)
    allowed_private_attributes = set(ALLOWED_PRIVATE_CANDIDATE_ATTRIBUTES)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name not in safe_imports
                or any(part.startswith("_") for part in alias.name.split("."))
                or (alias.asname is not None and alias.asname.startswith("_"))
                for alias in node.names
            ):
                return False, "import"
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if (
                node.level != 0
                or module not in safe_imports
                or any(alias.name == "*" for alias in node.names)
                or any(part.startswith("_") for part in module.split("."))
                or any(
                    alias.name.startswith("_")
                    or alias.name in forbidden_attributes
                    or (alias.asname is not None and alias.asname.startswith("_"))
                    for alias in node.names
                )
            ):
                return False, "import"
        elif isinstance(node, ast.Attribute) and (
            node.attr in forbidden_attributes
            or node.attr.startswith(FORBIDDEN_CANDIDATE_ATTRIBUTE_PREFIXES)
            or (
                node.attr.startswith("_")
                and node.attr not in allowed_private_attributes
            )
        ):
            return False, "private_attribute"
        elif isinstance(node, ast.Name):
            if node.id in forbidden_names or node.id.startswith("__"):
                return False, "forbidden_name"
        elif isinstance(node, ast.MatchClass) and any(
            attribute.startswith("_") or attribute in forbidden_attributes
            for attribute in node.kwd_attrs
        ):
            return False, "match_introspection"
    return True, "accepted"


def _trusted_tmpdir() -> Path:
    rendered = os.environ.get("SLURM_TMPDIR")
    if not rendered:
        raise PCF1SandboxError("PCF1 SLURM_TMPDIR is absent")
    path = Path(rendered)
    resolved = path.resolve(strict=True)
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not resolved.is_dir()
        or "lustre" in str(resolved).casefold()
        or resolved.stat().st_uid != os.getuid()
    ):
        raise PCF1SandboxError("PCF1 SLURM_TMPDIR is not a private local directory")
    return resolved


def _file_tail(handle: Any, limit: int) -> str:
    handle.flush()
    size = handle.seek(0, os.SEEK_END)
    handle.seek(max(0, size - limit))
    return _diagnostic(handle.read(), limit)


def classify_sandbox_termination(
    returncode: int | None, timed_out: bool
) -> tuple[bool, str]:
    """Classify only explicitly qualified scientific process outcomes."""

    if timed_out:
        raise PCF1SandboxError("PCF1 sandbox exceeded the outer wall timeout")
    if returncode == TRUSTED_COMPLETION_EXIT_CODE:
        return True, "trusted_tests_completed"
    if returncode == CANDIDATE_FAILURE_EXIT_CODE:
        return False, "candidate_execution_exception"
    if returncode == TEST_FAILURE_EXIT_CODE:
        return False, "official_test_failure"
    if returncode == SETUP_FAILURE_EXIT_CODE:
        return False, "candidate_induced_setup_failure"
    if returncode == POLICY_REJECTION_EXIT_CODE:
        return False, "candidate_policy_rejection"
    if returncode == RESOURCE_LIMIT_EXIT_CODE:
        return False, "candidate_resource_limit"
    if returncode == INFRASTRUCTURE_FAILURE_EXIT_CODE:
        raise PCF1SandboxError("PCF1 trusted sandbox phase failed")
    raise PCF1SandboxError("PCF1 sandbox terminated without trusted classification")


def isolated_program_result(
    candidate_source: str,
    timeout_seconds: float,
    *,
    setup_source: str = "",
    tests_source: str = "",
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    validate_host: bool = True,
    require_test_completion_attestation: bool = False,
    trusted_probe: bool = False,
    trusted_reference: bool = False,
    trusted_setup_compile: bool = False,
) -> dict[str, Any]:
    if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
        raise PCF1SandboxError("PCF1 code timeout differs")
    if validate_host:
        validate_sandbox_host()
    if not isinstance(candidate_source, str):
        raise PCF1SandboxError("PCF1 candidate source is not text")
    candidate_bytes = candidate_source.encode()
    if len(candidate_bytes) > MAX_CANDIDATE_BYTES:
        raise PCF1SandboxError("PCF1 candidate source exceeds bound")
    assessor_bytes = _assessor_transport_payload(
        candidate_source,
        setup_source,
        tests_source,
        trusted_probe=trusted_probe,
        trusted_reference=trusted_reference,
        trusted_setup_compile=trusted_setup_compile,
    )
    if sum(map(int, (trusted_probe, trusted_reference, trusted_setup_compile))) > 1:
        raise PCF1SandboxError("PCF1 trusted assessment mode is ambiguous")
    if trusted_probe:
        policy_passed, policy_reason = True, "not_applicable_trusted_probe"
        assessment_mode = "trusted_probe"
    elif trusted_reference:
        policy_passed, policy_reason = True, "not_applicable_trusted_reference"
        assessment_mode = "trusted_reference"
    elif trusted_setup_compile:
        policy_passed = True
        policy_reason = "not_applicable_trusted_setup_compile"
        assessment_mode = "trusted_setup_compile"
    else:
        policy_passed, policy_reason = validate_mbpp_candidate(candidate_source)
        assessment_mode = "candidate"
    candidate_fd = _sealed_read_only_memfd("pcf1-candidate", candidate_bytes)
    assessor_fd: int | None = None
    try:
        assessor_fd = _sealed_read_only_memfd("pcf1-assessor", assessor_bytes)
        trusted_tmp = _trusted_tmpdir()
        with (
            tempfile.TemporaryFile(dir=trusted_tmp) as stdout_file,
            tempfile.TemporaryFile(dir=trusted_tmp) as stderr_file,
            tempfile.TemporaryFile(dir=trusted_tmp) as info_file,
        ):
            timed_out = False
            result: subprocess.CompletedProcess[str] | None = None
            try:
                result = runner(
                    sandbox_command(candidate_fd, info_file.fileno()),
                    cwd="/",
                    env={},
                    stdin=assessor_fd,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    timeout=timeout_seconds + 2.0,
                    preexec_fn=_limits(timeout_seconds),
                    pass_fds=(candidate_fd, info_file.fileno()),
                    check=False,
                )
            except subprocess.TimeoutExpired:
                timed_out = True
            info_file.seek(0)
            if not info_file.read(1):
                raise PCF1SandboxError("PCF1 sandbox failed before namespace launch")
            stdout_file.seek(0)
            stdout_raw = stdout_file.read(4096).decode("utf-8", errors="replace")
            stderr = _file_tail(stderr_file, 4000)
            lines = stdout_raw.splitlines()
            ready = any(line == "PCF1_PYTHON_READY" for line in lines)
            if not ready:
                raise PCF1SandboxError("PCF1 sandbox Python launch failed")
            runtime_lines = [
                line.removeprefix("PCF1_RUNTIME_DESCRIPTOR ")
                for line in lines
                if line.startswith("PCF1_RUNTIME_DESCRIPTOR ")
            ]
            try:
                runtime_descriptor = json.loads(runtime_lines[0])
            except (IndexError, json.JSONDecodeError) as error:
                raise PCF1SandboxError(
                    "PCF1 sandbox runtime descriptor is absent"
                ) from error
            if (
                len(runtime_lines) != 1
                or runtime_descriptor != EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR
            ):
                raise PCF1SandboxError("PCF1 sandbox runtime descriptor differs")
    finally:
        os.close(candidate_fd)
        if assessor_fd is not None:
            os.close(assessor_fd)
    returncode = (
        None if timed_out else result.returncode if result is not None else None
    )
    passed, termination = classify_sandbox_termination(returncode, timed_out)
    test_completion_attested = passed
    receipt = {
        "passed": passed,
        "returncode": returncode,
        "timed_out": timed_out,
        "stdout_bytes": 0,
        "stderr_bytes": 0,
        "stdout": "",
        "stderr": stderr,
        "sandbox_config_sha256": SANDBOX_CONFIG_SHA256,
        "output_is_bounded": True,
        "direct_candidate_pid_1": True,
        "namespace_launch_verified": True,
        "python_launch_verified": True,
        "candidate_policy_sha256": CANDIDATE_POLICY_SHA256,
        "candidate_policy_passed": policy_passed,
        "candidate_policy_failure": policy_reason,
        "assessment_mode": assessment_mode,
        "python_runtime_descriptor": runtime_descriptor,
        "python_runtime_descriptor_sha256": (
            EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR_SHA256
        ),
        "memfd_abi": MEMFD_ABI,
        "resource_limit_exit_code": RESOURCE_LIMIT_EXIT_CODE,
        "test_completion_attestation_required": require_test_completion_attestation,
        "test_completion_attested": test_completion_attested,
        "infrastructure_failure": False,
        "termination_classification": termination,
    }
    return receipt


def score_completion(
    row: dict[str, Any], completion: str, code_timeout: float = 3.0
) -> dict[str, Any]:
    """Route every PCF1 MBPP candidate through the qualified OS sandbox."""

    task_name = str(row.get("task"))
    if task_name != "mbpp":
        task = TASKS.get(task_name)
        if task_name not in {"math500", "bbh_logic"} or task is None:
            raise PCF1SandboxError("PCF1 assessment task is unsupported")
        if task.get("kind") != "answer":
            raise PCF1SandboxError("PCF1 code task bypassed the sandbox")
        prediction = task["extract"](completion)
        if (
            task_name == "bbh_logic"
            and row.get("expected_answer_normalized") is not None
        ):
            gold = str(row["expected_answer_normalized"])
        else:
            gold = task["gold"](row)
        explicit = has_explicit_final_answer(completion)
        return {
            "prediction": prediction,
            "gold": gold,
            "explicit_final_answer": explicit,
            "correct": explicit and bool(task["match"](prediction, gold)),
            "program": None,
            "execution": None,
        }
    if _ALLOCATION_PROBE_SHA256 is None:
        raise PCF1SandboxError("PCF1 sandbox allocation is not qualified")
    tests = row.get("test_list")
    setup = row.get("test_setup_code", "")
    if (
        not isinstance(tests, list)
        or not tests
        or any(not isinstance(test, str) or not test.strip() for test in tests)
        or not isinstance(setup, str)
    ):
        raise PCF1SandboxError("PCF1 MBPP assessor harness is malformed")
    setup_sha256 = hashlib.sha256(setup.encode()).hexdigest()
    if setup_sha256 not in _QUALIFIED_SETUP_RECEIPTS:
        raise PCF1SandboxError("PCF1 MBPP setup is not qualified on this allocation")
    program = _mbpp_program(row, completion)
    execution = isolated_program_result(
        _mbpp_candidate_source(completion),
        code_timeout,
        setup_source=setup,
        tests_source="\n".join(tests),
        validate_host=False,
        require_test_completion_attestation=True,
    )
    return {
        "prediction": "pass" if execution["passed"] else "fail",
        "gold": "pass",
        "explicit_final_answer": True,
        "correct": bool(execution["passed"]),
        "program": program,
        "execution": execution,
    }


def preflight_mbpp_setup(setup_source: str) -> dict[str, Any]:
    """Compile one unique trusted setup before any candidate is assessed."""

    if _ALLOCATION_PROBE_SHA256 is None:
        raise PCF1SandboxError("PCF1 sandbox allocation is not qualified")
    if not isinstance(setup_source, str):
        raise PCF1SandboxError("PCF1 MBPP setup is not text")
    execution = isolated_program_result(
        "pass\n",
        3.0,
        setup_source=setup_source,
        validate_host=False,
        require_test_completion_attestation=True,
        trusted_setup_compile=True,
    )
    if (
        execution.get("passed") is not True
        or execution.get("termination_classification") != "trusted_tests_completed"
        or execution.get("test_completion_attested") is not True
        or execution.get("assessment_mode") != "trusted_setup_compile"
        or execution.get("candidate_policy_failure")
        != "not_applicable_trusted_setup_compile"
    ):
        raise PCF1SandboxError("PCF1 MBPP setup failed compile qualification")
    receipt = {
        "schema": "shohin-pcf1-mbpp-setup-qualification-v1",
        "status": "pass",
        "setup_source_sha256": hashlib.sha256(setup_source.encode()).hexdigest(),
        "candidate_policy_sha256": CANDIDATE_POLICY_SHA256,
        "sandbox_config_sha256": SANDBOX_CONFIG_SHA256,
        "allocation_probe_sha256": _ALLOCATION_PROBE_SHA256,
        "setup_qualification_mode": "compile_only_before_candidate",
        "termination_classification": "trusted_tests_completed",
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _QUALIFIED_SETUP_RECEIPTS[receipt["setup_source_sha256"]] = receipt[
        "receipt_sha256"
    ]
    return receipt


def qualify_mbpp_assessor_setups(
    assessors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Qualify every unique MBPP setup before any candidate is assessed."""

    receipts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for assessor in assessors:
        if not isinstance(assessor, dict) or assessor.get("task") not in {
            "math500",
            "bbh_logic",
            "mbpp",
        }:
            raise PCF1SandboxError("PCF1 assessor setup scope differs")
        if assessor.get("task") != "mbpp":
            continue
        setup_source = assessor.get("test_setup_code", "")
        if not isinstance(setup_source, str):
            raise PCF1SandboxError("PCF1 MBPP setup is not text")
        setup_sha256 = hashlib.sha256(setup_source.encode()).hexdigest()
        if setup_sha256 in seen:
            continue
        seen.add(setup_sha256)
        receipts.append(preflight_mbpp_setup(setup_source))
    return receipts


def mbpp_allocation_setup_receipts_sha256(receipts: list[dict[str, Any]]) -> str:
    """Hash the exact ordered hash-only setup qualification receipts."""

    digest = hashlib.sha256()
    for receipt in receipts:
        digest.update(
            (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
    return digest.hexdigest()


def validate_mbpp_setup_qualification_receipt(
    receipt: dict[str, Any],
    *,
    allocation_probe_sha256: str,
    setup_source_sha256: str | None = None,
) -> str:
    """Validate one hash-only standalone-setup qualification receipt."""

    unsigned = dict(receipt)
    receipt_sha256 = unsigned.pop("receipt_sha256", None)
    canonical_sha256 = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    expected_keys = {
        "schema",
        "status",
        "setup_source_sha256",
        "candidate_policy_sha256",
        "sandbox_config_sha256",
        "allocation_probe_sha256",
        "setup_qualification_mode",
        "termination_classification",
        "receipt_sha256",
    }
    if (
        set(receipt) != expected_keys
        or receipt.get("schema") != "shohin-pcf1-mbpp-setup-qualification-v1"
        or receipt.get("status") != "pass"
        or not isinstance(receipt.get("setup_source_sha256"), str)
        or len(receipt["setup_source_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in receipt["setup_source_sha256"]
        )
        or (
            setup_source_sha256 is not None
            and receipt.get("setup_source_sha256") != setup_source_sha256
        )
        or receipt.get("candidate_policy_sha256") != CANDIDATE_POLICY_SHA256
        or receipt.get("sandbox_config_sha256") != SANDBOX_CONFIG_SHA256
        or receipt.get("allocation_probe_sha256") != allocation_probe_sha256
        or receipt.get("setup_qualification_mode") != "compile_only_before_candidate"
        or receipt.get("termination_classification") != "trusted_tests_completed"
        or receipt_sha256 != canonical_sha256
    ):
        raise PCF1SandboxError("PCF1 MBPP setup qualification differs")
    return canonical_sha256


def preflight_mbpp_reference(
    row: dict[str, Any],
    *,
    split: str,
    setup_qualification: dict[str, Any],
    code_timeout: float = 3.0,
) -> dict[str, Any]:
    """Execute one exact nonsealed supervisor reference in trusted mode."""

    if split not in {"train", "development"} or row.get("task") != "mbpp":
        raise PCF1SandboxError("PCF1 reference preflight scope differs")
    identity = row.get("identity_sha256")
    completion = row.get("code")
    setup_source = row.get("test_setup_code", "")
    if (
        not isinstance(identity, str)
        or len(identity) != 64
        or any(character not in "0123456789abcdef" for character in identity)
        or not isinstance(completion, str)
        or not completion.strip()
        or not isinstance(setup_source, str)
    ):
        raise PCF1SandboxError("PCF1 MBPP reference row is malformed")
    expected_setup_sha256 = hashlib.sha256(setup_source.encode()).hexdigest()
    setup_receipt_sha256 = validate_mbpp_setup_qualification_receipt(
        setup_qualification,
        allocation_probe_sha256=str(_ALLOCATION_PROBE_SHA256),
        setup_source_sha256=expected_setup_sha256,
    )
    tests = row.get("test_list")
    if (
        not isinstance(tests, list)
        or not tests
        or any(not isinstance(test, str) or not test.strip() for test in tests)
    ):
        raise PCF1SandboxError("PCF1 frozen MBPP reference tests differ")
    execution = isolated_program_result(
        _mbpp_candidate_source(completion),
        code_timeout,
        setup_source=setup_source,
        tests_source="\n".join(tests),
        validate_host=False,
        require_test_completion_attestation=True,
        trusted_reference=True,
    )
    if (
        not isinstance(execution, dict)
        or execution.get("passed") is not True
        or execution.get("test_completion_attested") is not True
        or execution.get("termination_classification") != "trusted_tests_completed"
        or execution.get("assessment_mode") != "trusted_reference"
        or execution.get("candidate_policy_passed") is not True
        or execution.get("candidate_policy_failure")
        != "not_applicable_trusted_reference"
    ):
        raise PCF1SandboxError("PCF1 frozen MBPP reference did not pass")
    program = _mbpp_program(row, completion)
    return {
        "identity_sha256": identity,
        "split": split,
        "candidate_source_sha256": hashlib.sha256(
            _mbpp_candidate_source(completion).encode()
        ).hexdigest(),
        "program_sha256": hashlib.sha256(program.encode()).hexdigest(),
        "setup_source_sha256": expected_setup_sha256,
        "setup_qualification_sha256": setup_receipt_sha256,
        "candidate_policy_sha256": CANDIDATE_POLICY_SHA256,
        "sandbox_config_sha256": SANDBOX_CONFIG_SHA256,
        "allocation_probe_sha256": _ALLOCATION_PROBE_SHA256,
        "reference_assessment_mode": "trusted_reference",
        "generated_candidate_policy_applied": False,
        "termination_classification": "trusted_tests_completed",
    }


def qualify_allocation() -> dict[str, Any]:
    """Run the full adversarial isolation battery in this allocation."""

    global _ALLOCATION_PROBE_SHA256

    _ALLOCATION_PROBE_SHA256 = None
    _QUALIFIED_SETUP_RECEIPTS.clear()
    validate_sandbox_host()
    trusted_tmp = _trusted_tmpdir()
    sentinel_fd, sentinel_name = tempfile.mkstemp(
        prefix="pcf1-assessor-sentinel-", dir=trusted_tmp
    )
    try:
        os.write(sentinel_fd, b"PCF1_ASSESSOR_SENTINEL")
    finally:
        os.close(sentinel_fd)
    sentinel = Path(sentinel_name)
    try:

        def run(
            source: str,
            timeout: float = 3.0,
            *,
            setup_source: str = "",
            tests_source: str = "",
        ) -> dict[str, Any]:
            return isolated_program_result(
                source,
                timeout,
                setup_source=setup_source,
                tests_source=tests_source,
                validate_host=False,
                trusted_probe=True,
            )

        known_pass = run("assert 2 + 2 == 4\n")
        runtime_repeat = run("assert 3 * 3 == 9\n")
        assessor_token = b"PCF1_ASSESSOR_TRANSPORT_SENTINEL_20260811"
        candidate_visibility = run(
            f"""import os
from pathlib import Path

token = bytes({list(assessor_token)!r})
assert token not in Path("/candidate.py").read_bytes()
assert token not in Path("/proc/1/cmdline").read_bytes()
assert token not in Path("/proc/1/environ").read_bytes()
assert sorted(os.listdir("/")) == ["candidate.py", "dev", "lib64", "opt", "proc", "tmp"]
open_fds = []
for name in os.listdir("/proc/self/fd"):
    try:
        os.fstat(int(name))
    except OSError:
        continue
    open_fds.append(int(name))
assert sorted(open_fds) == [0, 1, 2]
assert os.read(0, 1) == b""
assert all(os.readlink(f"/proc/self/fd/{{fd}}") == "/dev/null" for fd in open_fds)
""",
            setup_source=(
                "ASSESSOR_TRANSPORT_SENTINEL = " + repr(assessor_token.decode()) + "\n"
            ),
            tests_source="assert ASSESSOR_TRANSPORT_SENTINEL.startswith('PCF1_')\n",
        )
        projected_system_libraries = sorted(
            Path(destination).name for _, destination, _, _ in SYSTEM_LIBRARY_BINDINGS
        )
        filesystem = run(f"""import os
from pathlib import Path

sentinel = Path({str(sentinel)!r})
assert not sentinel.exists()
try:
    sentinel.read_bytes()
except (FileNotFoundError, PermissionError):
    pass
else:
    raise AssertionError("host sentinel visible")
assert sorted(os.listdir("/")) == ["candidate.py", "dev", "lib64", "opt", "proc", "tmp"]
assert sorted(os.listdir("/lib64")) == {projected_system_libraries!r}
assert os.listdir("/opt/python/lib/python3.13/site-packages") == []
for protected_creation in (
    Path("/outside-work"),
    Path("/opt/python/lib/python3.13/site-packages/forbidden"),
):
    try:
        protected_creation.write_text("forbidden")
    except OSError:
        pass
    else:
        raise AssertionError("read-only sandbox mount was writable")
for protected in (Path("/candidate.py"), Path("/opt/python/bin/python3.13")):
    try:
        protected.write_bytes(b"mutated")
    except OSError:
        pass
    else:
        raise AssertionError("read-only sandbox input was writable")
Path("/tmp/private-write").write_text("ok")
assert Path("/tmp/private-write").read_text() == "ok"
Path("/tmp/link").symlink_to(sentinel)
assert not Path("/tmp/link").exists()
try:
    Path("/tmp/link").read_bytes()
except (FileNotFoundError, PermissionError):
    pass
else:
    raise AssertionError("symlink escaped the sandbox root")
""")
        previous_secret = os.environ.get("PCF1_PARENT_SECRET")
        os.environ["PCF1_PARENT_SECRET"] = "must-not-cross-namespace"
        try:
            environment = run("""import os
expected = {
    "HOME": "/tmp",
    "LANG": "C.UTF-8",
    "PATH": "/opt/python/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONHOME": "/opt/python",
    "TMPDIR": "/tmp",
    "PWD": "/tmp",
}
assert os.environ == expected
assert "PCF1_PARENT_SECRET" not in os.environ
""")
        finally:
            if previous_secret is None:
                os.environ.pop("PCF1_PARENT_SECRET", None)
            else:
                os.environ["PCF1_PARENT_SECRET"] = previous_secret
        process = run("""import os
from pathlib import Path
assert os.getpid() == 1
numeric = sorted(path.name for path in Path("/proc").iterdir() if path.name.isdigit())
assert numeric == ["1"]
open_fds = []
for name in os.listdir("/proc/self/fd"):
    try:
        os.fstat(int(name))
    except OSError:
        continue
    open_fds.append(int(name))
assert sorted(open_fds) == [0, 1, 2]
assert b"pcf1_code_sandbox.py" not in Path("/proc/1/cmdline").read_bytes()
""")
        safe_import_namespace_reachability = run(SAFE_IMPORT_CAPABILITIES_PROBE_SOURCE)
        deterministic_random = run("""import random
assert [random.random() for _ in range(3)] == [
    0.27005403657241234,
    0.339988502969811,
    0.21577936677708065,
]
""")
        network = run("""import errno
import socket
sock = socket.socket()
try:
    assert sock.connect_ex(("1.1.1.1", 53)) in (errno.ENETUNREACH, errno.EHOSTUNREACH)
finally:
    sock.close()
""")
        subprocess_and_fork = run("""import os
import subprocess
fork_blocked = subprocess_blocked = False
try:
    child = os.fork()
except OSError:
    fork_blocked = True
else:
    if child == 0:
        os._exit(71)
    os.waitpid(child, 0)
try:
    subprocess.run(["/opt/python/bin/python3.13", "-I", "-S", "-c", "pass"], check=True)
except (OSError, subprocess.SubprocessError):
    subprocess_blocked = True
assert fork_blocked and subprocess_blocked
""")
        flood = run("""import errno
import os
import resource
import signal
assert resource.getrlimit(resource.RLIMIT_FSIZE) == (1048576, 1048576)
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
fd = os.open("/tmp/flood", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
blocked = False
try:
    while True:
        os.write(fd, b"x" * 65536)
except OSError as error:
    blocked = error.errno == errno.EFBIG
finally:
    os.close(fd)
assert blocked and os.stat("/tmp/flood").st_size <= 1048576
""")
        memory = run("""import resource
limit = 1073741824
assert resource.getrlimit(resource.RLIMIT_AS) == (limit, limit)
try:
    bytearray(limit * 2)
except (MemoryError, OverflowError):
    pass
else:
    raise AssertionError("address-space limit was bypassed")
""")
        # Admission may share a busy host with another evaluation shard. Give
        # the trusted SIGXCPU path enough wall time to observe its two-second
        # CPU limit; this does not alter any candidate assessment timeout.
        cpu = run("while True: pass\n", timeout=RESOURCE_PROBE_TIMEOUT_SECONDS)
        system_exit_zero = run("raise SystemExit(0)\n")
        resource_limit_exit_code_forgery = run(
            f"raise SystemExit({RESOURCE_LIMIT_EXIT_CODE})\n"
        )
        failed_official_assertion = run(
            "def f():\n    return 0\n", tests_source="assert f() == 1\n"
        )
        forged_marker = run(
            "print('PCF1_TRUSTED_TESTS_COMPLETED')\nraise AssertionError\n"
        )
        standalone_setup = run("pass\n", setup_source="assert callable(len)\n")
        post_candidate_setup_failure = run(
            "len = None\n", setup_source="assert callable(len)\n"
        )
        try:
            run("pass\n", setup_source="not valid python !\n")
        except PCF1SandboxError:
            setup_compile_failure_is_infrastructure = True
        else:
            setup_compile_failure_is_infrastructure = False
        try:
            run("import time\ntime.sleep(10)\n", timeout=0.1)
        except PCF1SandboxError:
            outer_wall_timeout_is_infrastructure = True
        else:
            outer_wall_timeout_is_infrastructure = False
        policy_escapes = (
            "import os\nos._exit(73)",
            "from random import _os as x\nx._exit(73)",
            "import re\nre.enum.sys.exit(73)",
            "from re import enum as x\nx.sys.exit(73)",
            "import fractions\nfractions.sys.exit(73)",
            "import statistics\nstatistics.sys.exit(73)",
            (
                "import string\n"
                "string.Formatter().get_field("
                '"0.__class__.__mro__[1].__subclasses__", [0], {})'
            ),
            "__import__('os')._exit(73)",
            "print(globals())",
            "print((1).__class__)",
        )
        candidate_policy_escape_blocked = all(
            validate_mbpp_candidate(source)[0] is False for source in policy_escapes
        )
        aggregate_file_creation_policy_blocked = all(
            validate_mbpp_candidate(source)[0] is False
            for source in (
                "for index in range(10000):\n    open(f'/tmp/{index}', 'w').close()",
                "import os\nfor index in range(10000):\n    os.open(f'/tmp/{index}', os.O_CREAT)",
                "import pathlib\npathlib.Path('/tmp/x').write_text('x')",
                "import tempfile\ntempfile.mkstemp()",
            )
        )
        random_entropy_reentry_blocked = all(
            validate_mbpp_candidate(source)[0] is False
            for source in (
                "import random\nrandom.seed()",
                "import random\nrandom.seed(None)",
                "import random\nrandom.seed(1)",
                "from random import seed\nseed()",
                "import random\nrandom.Random()",
                "from random import Random\nRandom()",
                "import random\nrandom.SystemRandom().random()",
                "from random import SystemRandom\nSystemRandom().random()",
            )
        )
        policy_rejection = isolated_program_result(
            "import os\nos._exit(73)\n",
            3.0,
            validate_host=False,
        )
    finally:
        sentinel.unlink(missing_ok=True)
    probe_results = {
        "known_pass": known_pass.get("passed") is True,
        "assessor_sentinel_hidden": filesystem.get("passed") is True,
        "assessor_transport_closed": candidate_visibility.get("passed") is True,
        "candidate_mount_excludes_assessor": (
            candidate_visibility.get("passed") is True
        ),
        "filesystem_projection_exact": filesystem.get("passed") is True,
        "read_only_inputs_immutable": filesystem.get("passed") is True,
        "root_filesystem_read_only": filesystem.get("passed") is True,
        "symlink_traversal_blocked": filesystem.get("passed") is True,
        "private_tmp_writable": filesystem.get("passed") is True,
        "aggregate_file_creation_policy_blocked": (
            aggregate_file_creation_policy_blocked
        ),
        "parent_proc_hidden": process.get("passed") is True,
        "candidate_is_pid_1": process.get("passed") is True,
        "parent_environment_hidden": environment.get("passed") is True,
        "environment_exact": environment.get("passed") is True,
        "safe_import_namespace_reachability": (
            safe_import_namespace_reachability.get("passed") is True
        ),
        "site_packages_mask_read_only": filesystem.get("passed") is True,
        "random_seed_deterministic": deterministic_random.get("passed") is True,
        "random_entropy_reentry_blocked": random_entropy_reentry_blocked,
        "python_hash_seed_effective": (
            known_pass.get("python_runtime_descriptor", {})
            .get("flags", {})
            .get("hash_randomization")
            == 0
        ),
        "python_runtime_descriptor_exact": (
            known_pass.get("python_runtime_descriptor")
            == EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR
            and known_pass.get("python_runtime_descriptor_sha256")
            == EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR_SHA256
        ),
        "python_runtime_descriptor_cross_process": (
            runtime_repeat.get("python_runtime_descriptor")
            == known_pass.get("python_runtime_descriptor")
            and runtime_repeat.get("python_runtime_descriptor_sha256")
            == known_pass.get("python_runtime_descriptor_sha256")
        ),
        "network_unreachable": network.get("passed") is True,
        "subprocess_blocked": subprocess_and_fork.get("passed") is True,
        "fork_blocked": subprocess_and_fork.get("passed") is True,
        "flood_bounded": flood.get("passed") is True,
        "address_space_bounded": memory.get("passed") is True,
        "cpu_bounded": cpu.get("passed") is False
        and cpu.get("returncode") == RESOURCE_LIMIT_EXIT_CODE
        and cpu.get("termination_classification") == "candidate_resource_limit",
        "elf_projection_exact": filesystem.get("passed") is True,
        "candidate_status_fd_closed": process.get("passed") is True,
        "trusted_completion_exit_attested": (
            known_pass.get("returncode") == TRUSTED_COMPLETION_EXIT_CODE
            and known_pass.get("test_completion_attested") is True
        ),
        "system_exit_bypass_blocked": system_exit_zero.get("passed") is False,
        "resource_limit_exit_code_forgery_blocked": (
            resource_limit_exit_code_forgery.get("passed") is False
            and resource_limit_exit_code_forgery.get("returncode")
            == CANDIDATE_FAILURE_EXIT_CODE
            and resource_limit_exit_code_forgery.get("termination_classification")
            == "candidate_execution_exception"
        ),
        "outer_wall_timeout_is_infrastructure": (outer_wall_timeout_is_infrastructure),
        "os_exit_bypass_blocked": validate_mbpp_candidate("import os\nos._exit(0)")[0]
        is False,
        "failed_official_assertion_blocked": (
            failed_official_assertion.get("passed") is False
        ),
        "status_marker_forgery_blocked": forged_marker.get("passed") is False,
        "candidate_policy_escape_blocked": candidate_policy_escape_blocked,
        "policy_rejection_runs_in_sandbox": (
            policy_rejection.get("passed") is False
            and policy_rejection.get("returncode") == POLICY_REJECTION_EXIT_CODE
            and policy_rejection.get("candidate_policy_passed") is False
            and policy_rejection.get("termination_classification")
            == "candidate_policy_rejection"
        ),
        "post_candidate_setup_failure_is_scientific": (
            standalone_setup.get("passed") is True
            and post_candidate_setup_failure.get("passed") is False
            and post_candidate_setup_failure.get("returncode")
            == SETUP_FAILURE_EXIT_CODE
            and post_candidate_setup_failure.get("termination_classification")
            == "candidate_induced_setup_failure"
        ),
        "setup_compile_failure_is_infrastructure": (
            setup_compile_failure_is_infrastructure
        ),
    }
    if set(probe_results) != SANDBOX_PROBES or not all(probe_results.values()):
        raise PCF1SandboxError("PCF1 sandbox isolation probe failed")
    probe_sha256 = hashlib.sha256(
        json.dumps(probe_results, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _ALLOCATION_PROBE_SHA256 = probe_sha256
    receipt = {
        "schema": "shohin-pcf1-code-sandbox-receipt-v1",
        "status": "pass",
        "bwrap_path": str(BWRAP),
        "bwrap_sha256": BWRAP_SHA256,
        "bwrap_version": "bubblewrap 0.4.0",
        "python_executable": str(PYTHON_EXECUTABLE),
        "python_sha256": PYTHON_SHA256,
        "sandbox_config_sha256": SANDBOX_CONFIG_SHA256,
        "candidate_policy_sha256": CANDIDATE_POLICY_SHA256,
        "trusted_completion_exit_code": TRUSTED_COMPLETION_EXIT_CODE,
        "candidate_failure_exit_code": CANDIDATE_FAILURE_EXIT_CODE,
        "infrastructure_failure_exit_code": INFRASTRUCTURE_FAILURE_EXIT_CODE,
        "test_failure_exit_code": TEST_FAILURE_EXIT_CODE,
        "setup_failure_exit_code": SETUP_FAILURE_EXIT_CODE,
        "policy_rejection_exit_code": POLICY_REJECTION_EXIT_CODE,
        "resource_limit_exit_code": RESOURCE_LIMIT_EXIT_CODE,
        "candidate_random_seed": CANDIDATE_RANDOM_SEED,
        "python_runtime_descriptor": EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR,
        "python_runtime_descriptor_sha256": (
            EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR_SHA256
        ),
        "memfd_abi": MEMFD_ABI,
        "sandbox_runtime_tree_sha256": SANDBOX_RUNTIME_TREE_SHA256,
        "sandbox_runtime_tree_entries": SANDBOX_RUNTIME_TREE_ENTRIES,
        "sandbox_runtime_tree_files": SANDBOX_RUNTIME_TREE_FILES,
        "sandbox_runtime_tree_directories": SANDBOX_RUNTIME_TREE_DIRECTORIES,
        "sandbox_runtime_tree_bytes": SANDBOX_RUNTIME_TREE_BYTES,
        "elf_closure_audit_sha256": ELF_CLOSURE_AUDIT_SHA256,
        "system_library_members": expected_system_library_members(),
        "clear_environment": True,
        "network_namespace": "isolated",
        "candidate_read_only": True,
        "candidate_direct_pid_1": True,
        "site_packages_visible": False,
        "probe_results": probe_results,
        "probe_sha256": probe_sha256,
        "sandbox_isolation_passed": True,
    }
    validate_sandbox_receipt_payload(receipt)
    return receipt


def atomic_json(path: Path, payload: dict[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise PCF1SandboxError(f"refusing existing sandbox receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise PCF1SandboxError("sandbox receipt publication race") from error
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("qualify",))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = qualify_allocation()
    digest = atomic_json(args.output, receipt)
    print(json.dumps({"sandbox_receipt_sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
