"""Adversarial contract tests for the PCF1 generated-code sandbox."""

from __future__ import annotations

import ast
import ctypes
import json
import os
import operator
from pathlib import Path
import signal
import subprocess
import sys
import types
from typing import Any

import pytest

import pcf1_code_sandbox as sandbox


def test_runtime_import_graph_excludes_generic_rollout_executor() -> None:
    source = Path(sandbox.__file__).read_text(encoding="utf-8")
    imports = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    }

    assert "hf_product_reasoning_rollouts" not in imports
    assert "hf_product_reasoning_rollouts" not in source


def test_safe_import_capability_probe_source_executes() -> None:
    namespace: dict[str, Any] = {}
    exec(sandbox.SAFE_IMPORT_CAPABILITIES_PROBE_SOURCE, namespace)
    capability_allowed = namespace["capability_allowed"]
    module_allowed = namespace["module_allowed"]
    assert capability_allowed("EllipsisType", type(Ellipsis)) is True
    assert capability_allowed("harmless_alias", open) is False
    assert capability_allowed("open", type(Ellipsis)) is False
    assert capability_allowed("itemgetter", operator.itemgetter) is True
    assert capability_allowed("open", operator.itemgetter) is False
    assert capability_allowed("harmless_alias", operator.attrgetter) is False
    assert capability_allowed("harmless_alias", operator.methodcaller) is False

    def fake_io_capability() -> None:
        pass

    fake_io_capability.__module__ = "_io"
    assert capability_allowed("harmless_alias", fake_io_capability) is False
    assert module_allowed(types.ModuleType("_io")) is False
    assert module_allowed(operator) is False


@pytest.mark.parametrize(
    "source",
    (
        "import statistics\nstatistics.attrgetter('value')\n",
        "import statistics\nstatistics.methodcaller('value')\n",
        "from statistics import attrgetter\n",
        "from statistics import methodcaller\n",
        "import statistics\nstatistics.open('path')\n",
        "from statistics import open\n",
    ),
)
def test_candidate_policy_blocks_operator_introspection_reexports(source: str) -> None:
    assert sandbox.validate_mbpp_candidate(source)[0] is False


@pytest.mark.parametrize(
    "source",
    (
        "import statistics\nkey = statistics.itemgetter(0)\n",
        "from statistics import itemgetter\nkey = itemgetter(0)\n",
    ),
)
def test_candidate_policy_retains_statistics_itemgetter(source: str) -> None:
    assert sandbox.validate_mbpp_candidate(source) == (True, "accepted")


def test_minimal_runtime_receipt_includes_projection_root_directory() -> None:
    assert sandbox.SANDBOX_RUNTIME_TREE_SHA256 == (
        "7c6ed935cd9585475e82c64d4e591837c0c4ccd9538ad2f41adcfd3c8b71ce0d"
    )
    assert sandbox.SANDBOX_RUNTIME_TREE_DIRECTORIES == 112
    assert sandbox.SANDBOX_RUNTIME_TREE_ENTRIES == (
        sandbox.SANDBOX_RUNTIME_TREE_FILES + sandbox.SANDBOX_RUNTIME_TREE_DIRECTORIES
    )


def test_effective_python_runtime_descriptor_is_exact_full_environment_pin() -> None:
    descriptor = sandbox.EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR
    assert descriptor["sys_path"] == [
        "/opt/python/lib/python313.zip",
        "/opt/python/lib/python3.13",
        "/opt/python/lib/python3.13/lib-dynload",
    ]
    assert descriptor["flags"]["ignore_environment"] == 0
    assert descriptor["flags"]["hash_randomization"] == 0
    assert descriptor["flags"]["safe_path"] is True
    assert descriptor["flags"]["utf8_mode"] == 1
    assert descriptor["hash_probe"] == -1545367155142879260
    assert descriptor["set_order"] == [
        "pcf1-beta",
        "pcf1-gamma",
        "pcf1-alpha",
        "pcf1-delta",
    ]
    assert sandbox.SANDBOX_CONFIG["python_flags"] == ["-P", "-s", "-S", "-B"]


def test_native_noncode_scoring_preserves_frozen_semantics() -> None:
    math = sandbox.score_completion(
        {"task": "math500", "answer": r"\boxed{4}"}, "Final answer: 4"
    )
    logic = sandbox.score_completion(
        {
            "task": "bbh_logic",
            "answer": "(A)",
            "expected_answer_normalized": "(A)",
        },
        "The final answer is (A)",
    )

    assert math == {
        "prediction": "4",
        "gold": "4",
        "explicit_final_answer": True,
        "correct": True,
        "program": None,
        "execution": None,
    }
    assert logic["correct"] is True
    assert logic["explicit_final_answer"] is True


def test_native_noncode_router_rejects_every_non_pcf_task() -> None:
    with pytest.raises(sandbox.PCF1SandboxError, match="unsupported"):
        sandbox.score_completion({"task": "humaneval"}, "pass")


def test_command_runs_anonymous_candidate_directly_as_pid_one() -> None:
    command = sandbox.sandbox_command(17, 19)
    rendered = "\n".join(command)

    assert command[0] == "/usr/bin/bwrap"
    assert "--as-pid-1" in command
    assert "--unshare-all" in command
    assert "--clearenv" not in command
    assert "/work/runner.py" not in rendered
    assert "pcf1-runner" not in rendered
    assert str(sandbox.PYTHON_ROOT) not in command
    assert command.count("--ro-bind-data") == 1
    assert "/candidate.py" in command
    assert command[-6:] == [
        "-P",
        "-s",
        "-S",
        "-B",
        "-c",
        sandbox.BOOTSTRAP_SOURCE,
    ]
    separator = command.index("--")
    site_packages_remount = next(
        index
        for index in range(separator)
        if command[index : index + 2]
        == ["--remount-ro", "/opt/python/lib/python3.13/site-packages"]
    )
    root_remount = next(
        index
        for index in range(separator)
        if command[index : index + 2] == ["--remount-ro", "/"]
    )
    candidate_mount = command.index("/candidate.py")
    assert candidate_mount < site_packages_remount < root_remount < separator
    assert command[root_remount + 2 : separator] == ["--chdir", "/tmp"]
    assert sandbox.SANDBOX_CONFIG["root_filesystem_read_only"] is True
    assert sandbox.SANDBOX_CONFIG["masked_site_packages_read_only"] is True
    assert command[separator : separator + 8] == [
        "--",
        "/opt/python/bin/python3.13",
        "-P",
        "-s",
        "-S",
        "-B",
        "-c",
        sandbox.BOOTSTRAP_SOURCE,
    ]
    assert command[separator:].count("/opt/python/bin/python3.13") == 1
    assert sandbox.BOOTSTRAP_SOURCE.index('compile(payload["setup_source"]') < (
        sandbox.BOOTSTRAP_SOURCE.index("PCF1_PYTHON_READY")
    )
    assert sandbox.BOOTSTRAP_SOURCE.index('compile(payload["tests_source"]') < (
        sandbox.BOOTSTRAP_SOURCE.index("PCF1_PYTHON_READY")
    )
    assert sandbox.BOOTSTRAP_SOURCE.index("os.close(status_fd)") < (
        sandbox.BOOTSTRAP_SOURCE.index("exec(candidate_code")
    )


def test_launcher_applies_exact_limits_after_exec_without_preexec_callback() -> None:
    command = sandbox.sandbox_launch_command(17, 19, 3.0)

    assert command[:5] == [
        "/usr/bin/prlimit",
        "--cpu=3:4",
        "--as=1073741824:1073741824",
        "--fsize=1048576:1048576",
        "--",
    ]
    assert command[5:] == sandbox.sandbox_command(17, 19)
    assert sandbox.SANDBOX_CONFIG["resource_limit_launcher"] == (
        "exec-prlimit-before-bwrap-no-preexec-fn"
    )
    assert sandbox.SANDBOX_CONFIG["prlimit_sha256"] == (
        "2c1c7948498f2cb755d8c93ecf72c0651f5a5db23f79cc39cfa6727693d241d5"
    )
    assert sandbox.BOOTSTRAP_SOURCE.index("os.close(0)") < (
        sandbox.BOOTSTRAP_SOURCE.index("PCF1_PYTHON_READY")
    )
    assert sandbox.BOOTSTRAP_SOURCE.index("random.seed(2026080816)") < (
        sandbox.BOOTSTRAP_SOURCE.index("exec(candidate_code")
    )
    assert sandbox.BOOTSTRAP_SOURCE.index("exec(candidate_code") < (
        sandbox.BOOTSTRAP_SOURCE.index("exec(setup_code")
    )
    assert sandbox.BOOTSTRAP_SOURCE.index("exec(setup_code") < (
        sandbox.BOOTSTRAP_SOURCE.index("exec(tests_code")
    )
    assert sandbox.BOOTSTRAP_SOURCE.index("exec(tests_code") < (
        sandbox.BOOTSTRAP_SOURCE.rindex("os._exit(73)")
    )
    assert sandbox.BOOTSTRAP_SOURCE.index("os.dup2(null_output_fd, 1)") < (
        sandbox.BOOTSTRAP_SOURCE.index("exec(candidate_code")
    )
    assert "status_fd" not in '{"__name__": "__main__", "__file__": "/candidate.py"}'
    assert "subprocess" not in sandbox.BOOTSTRAP_SOURCE
    assert "RLIMIT_NPROC, (1, 1)" in sandbox.BOOTSTRAP_SOURCE
    for name in sandbox.MINIMAL_LIBRARY_NAMES:
        assert f"/opt/python/lib/{name}" in command
    assert not any(
        command[index : index + 3] == ["--ro-bind", "/lib64", "/lib64"]
        for index in range(len(command) - 2)
    )
    for source, destination, _, _ in sandbox.SYSTEM_LIBRARY_BINDINGS:
        assert str(source) in command
        assert destination in command


def test_filesystem_probe_requires_read_only_root_and_site_mask() -> None:
    source = Path(sandbox.__file__).read_text(encoding="utf-8")
    assert 'Path("/outside-work")' in source
    assert 'Path("/opt/python/lib/python3.13/site-packages/forbidden")' in source
    assert 'Path("/tmp/private-write").write_text("ok")' in source


def test_assessor_transport_excludes_candidate_and_binds_frozen_contract() -> None:
    payload = json.loads(
        sandbox._assessor_transport_payload(
            "candidate", "setup", "tests", trusted_probe=False
        )
    )
    assert payload == {
        "schema": "shohin-pcf1-sandbox-assessor-transport-v1",
        "seed": 2026080816,
        "assessment_mode": "candidate",
        "candidate_policy_sha256": sandbox.CANDIDATE_POLICY_SHA256,
        "candidate_policy_passed": True,
        "candidate_policy_failure": "accepted",
        "candidate_source_sha256": sandbox.hashlib.sha256(b"candidate").hexdigest(),
        "candidate_timeout_seconds": 3.0,
        "setup_source": "setup",
        "tests_source": "tests",
    }
    assert "candidate_source" not in payload
    assert sandbox.SANDBOX_CONFIG["candidate_random_seed"] == 2026080816


@pytest.mark.parametrize(
    ("candidate", "setup", "tests", "returncode", "ready"),
    (
        ("def f(): return 1", "", "assert f() == 1", 73, True),
        ("raise RuntimeError('candidate')", "", "", 74, True),
        ("pass", "raise RuntimeError('setup')", "", 77, True),
        ("def f(): return 0", "", "assert f() == 1", 76, True),
        (
            "import random\nassert random.random() == 0.27005403657241234",
            "",
            "",
            73,
            True,
        ),
        ("pass", "", "not valid python !", 75, False),
        ("import os\nos._exit(73)", "", "", 78, True),
    ),
)
def test_bootstrap_phase_exit_codes_execute_exactly(
    tmp_path: Path,
    candidate: str,
    setup: str,
    tests: str,
    returncode: int,
    ready: bool,
) -> None:
    flag_names = list(sandbox.EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR["flags"])
    descriptor_query = f"""import json
import sys
flags = {{name: getattr(sys.flags, name) for name in {flag_names!r}}}
print(json.dumps({{
    "executable": sys.executable,
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "sys_path": list(sys.path),
    "flags": flags,
    "hash_probe": hash("PCF1_HASH_PROBE"),
    "set_order": list(set(("pcf1-alpha", "pcf1-beta", "pcf1-gamma", "pcf1-delta"))),
}}, sort_keys=True))
"""
    launch_environment = {
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    }
    query = subprocess.run(
        [sys.executable, "-P", "-s", "-S", "-B", "-c", descriptor_query],
        check=True,
        capture_output=True,
        text=True,
        env=launch_environment,
    )
    runtime_descriptor = json.loads(query.stdout)
    candidate_path = tmp_path / "candidate.py"
    candidate_path.write_text(candidate, encoding="utf-8")
    assessor_path = tmp_path / "assessor.json"
    assessor_path.write_bytes(
        sandbox._assessor_transport_payload(
            candidate, setup, tests, trusted_probe=False
        )
    )
    source = sandbox._render_bootstrap_source(runtime_descriptor).replace(
        '"/candidate.py"', repr(str(candidate_path.resolve()))
    )
    with assessor_path.open("rb") as assessor_handle:
        completed = subprocess.run(
            [sys.executable, "-P", "-s", "-S", "-B", "-c", source],
            check=False,
            stdin=assessor_handle,
            capture_output=True,
            timeout=5,
            env=launch_environment,
        )
    assert completed.returncode == returncode
    assert (b"PCF1_PYTHON_READY" in completed.stdout) is ready


def test_isolated_transport_mounts_only_raw_candidate_and_hides_assessor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, bytes] = {}

    def fake_memfd(_name: str, content: bytes) -> int:
        path = tmp_path / f"anonymous-{len(observed)}"
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        path.unlink()
        os.write(fd, content)
        os.lseek(fd, 0, os.SEEK_SET)
        return fd

    def fake_runner(
        command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        assert "preexec_fn" not in kwargs
        candidate_index = command.index("--ro-bind-data") + 1
        candidate_fd = int(command[candidate_index])
        assessor_fd = int(kwargs["stdin"])
        observed["candidate"] = os.pread(candidate_fd, 1 << 20, 0)
        observed["assessor"] = os.pread(assessor_fd, 1 << 20, 0)
        os.write(kwargs["stdout"].fileno(), b"PCF1_RUNTIME_DESCRIPTOR ")
        os.write(
            kwargs["stdout"].fileno(),
            json.dumps(
                sandbox.EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\nPCF1_PYTHON_READY\n",
        )
        os.write(kwargs["stderr"].fileno(), b"")
        info_fd = int(command[command.index("--info-fd") + 1])
        os.write(info_fd, b"{}")
        return subprocess.CompletedProcess(
            command, sandbox.TRUSTED_COMPLETION_EXIT_CODE
        )

    monkeypatch.setattr(sandbox, "validate_sandbox_host", lambda: None)
    monkeypatch.setattr(sandbox, "_trusted_tmpdir", lambda: tmp_path)
    monkeypatch.setattr(sandbox, "_sealed_read_only_memfd", fake_memfd)
    result = sandbox.isolated_program_result(
        "MODEL_ONLY_TOKEN = 1\n",
        3.0,
        setup_source="ASSESSOR_ONLY_TOKEN = 2\n",
        tests_source="assert ASSESSOR_ONLY_TOKEN == 2\n",
        runner=fake_runner,
        validate_host=False,
    )

    assert result["passed"] is True
    assert result["resource_limit_exit_code"] == 79
    assert observed["candidate"] == b"MODEL_ONLY_TOKEN = 1\n"
    assessor = json.loads(observed["assessor"])
    assert "candidate_source" not in assessor
    assert b"MODEL_ONLY_TOKEN" not in observed["assessor"]
    assert b"ASSESSOR_ONLY_TOKEN" not in observed["candidate"]


def test_libc_memfd_backend_uses_exact_pinned_linux_abi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    class FakeCreate:
        argtypes: Any = None
        restype: Any = None

        def __call__(self, name: bytes, flags: int) -> int:
            calls["name"] = name
            calls["flags"] = flags
            calls["argtypes"] = self.argtypes
            calls["restype"] = self.restype
            return 41

    create = FakeCreate()

    class FakeLibc:
        memfd_create = create

    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: FakeLibc())
    assert sandbox._libc_memfd_create("pcf1-test") == 41
    assert calls == {
        "name": b"pcf1-test",
        "flags": 0x1 | 0x2,
        "argtypes": [ctypes.c_char_p, ctypes.c_uint],
        "restype": ctypes.c_int,
    }


def test_libc_memfd_backend_preserves_errno_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailedCreate:
        argtypes: Any = None
        restype: Any = None

        def __call__(self, _name: bytes, _flags: int) -> int:
            ctypes.set_errno(38)
            return -1

    class FakeLibc:
        memfd_create = FailedCreate()

    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: FakeLibc())
    with pytest.raises(
        sandbox.PCF1SandboxError, match="memfd_create admission"
    ) as caught:
        sandbox._libc_memfd_create("pcf1-test")
    assert isinstance(caught.value.__cause__, OSError)
    assert caught.value.__cause__.errno == 38


@pytest.mark.skipif(sys.platform != "linux", reason="Linux memfd ABI required")
def test_anonymous_transport_descriptor_is_sealed_and_read_only() -> None:
    fd = sandbox._sealed_read_only_memfd("pcf1-test", b"immutable")
    try:
        assert sandbox.MEMFD_ABI == {
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
        assert sandbox.fcntl.fcntl(fd, 1034) == 15
        assert (
            sandbox.fcntl.fcntl(fd, sandbox.fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY
        )
        assert os.read(fd, 9) == b"immutable"
        with pytest.raises(OSError):
            os.write(fd, b"x")
    finally:
        os.close(fd)


def test_candidate_and_bootstrap_cannot_access_ctypes_memfd_backend() -> None:
    assert "ctypes" not in sandbox.BOOTSTRAP_SOURCE
    assert sandbox.validate_mbpp_candidate("import ctypes\n") == (
        False,
        "import",
    )


def test_qualification_contains_every_required_adversarial_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_execution(
        source: str,
        _: float,
        *,
        setup_source: str = "",
        tests_source: str = "",
        validate_host: bool,
        trusted_probe: bool = False,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        assert validate_host is False
        calls.append(
            {
                "source": source,
                "setup_source": setup_source,
                "tests_source": tests_source,
                "trusted_probe": trusted_probe,
            }
        )
        runtime = {
            "python_runtime_descriptor": (sandbox.EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR),
            "python_runtime_descriptor_sha256": (
                sandbox.EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR_SHA256
            ),
        }
        if setup_source == "not valid python !\n":
            raise sandbox.PCF1SandboxError("trusted phase failed")
        if source == "len = None\n" and setup_source:
            return {
                "passed": False,
                "timed_out": False,
                "returncode": sandbox.SETUP_FAILURE_EXIT_CODE,
                "test_completion_attested": False,
                "termination_classification": "candidate_induced_setup_failure",
                **runtime,
            }
        if "time.sleep(10)" in source:
            return {
                "passed": False,
                "timed_out": False,
                "returncode": sandbox.RESOURCE_LIMIT_EXIT_CODE,
                "test_completion_attested": False,
                "termination_classification": "candidate_resource_limit",
                **runtime,
            }
        if source == "while True: pass\n":
            return {
                "passed": False,
                "timed_out": False,
                "returncode": sandbox.RESOURCE_LIMIT_EXIT_CODE,
                "test_completion_attested": False,
                "termination_classification": "candidate_resource_limit",
                **runtime,
            }
        if source == "import os\nos._exit(73)\n" and not trusted_probe:
            return {
                "passed": False,
                "timed_out": False,
                "returncode": sandbox.POLICY_REJECTION_EXIT_CODE,
                "candidate_policy_passed": False,
                "test_completion_attested": False,
                "termination_classification": "candidate_policy_rejection",
                **runtime,
            }
        if source in {
            "raise SystemExit(0)\n",
            f"raise SystemExit({sandbox.RESOURCE_LIMIT_EXIT_CODE})\n",
            "import os\nos._exit(0)\n",
            "print('PCF1_TRUSTED_TESTS_COMPLETED')\nraise AssertionError\n",
        }:
            return {
                "passed": False,
                "timed_out": False,
                "returncode": sandbox.CANDIDATE_FAILURE_EXIT_CODE,
                "test_completion_attested": False,
                "termination_classification": "candidate_execution_exception",
                **runtime,
            }
        if source == "def f():\n    return 0\n" and tests_source:
            return {
                "passed": False,
                "timed_out": False,
                "returncode": sandbox.TEST_FAILURE_EXIT_CODE,
                "test_completion_attested": False,
                "termination_classification": "official_test_failure",
                **runtime,
            }
        return {
            "passed": True,
            "timed_out": False,
            "returncode": sandbox.TRUSTED_COMPLETION_EXIT_CODE,
            "test_completion_attested": True,
            "termination_classification": "trusted_tests_completed",
            "assessment_mode": (
                "trusted_setup_compile"
                if _kwargs.get("trusted_setup_compile")
                else "candidate"
            ),
            "candidate_policy_failure": (
                "not_applicable_trusted_setup_compile"
                if _kwargs.get("trusted_setup_compile")
                else "accepted"
            ),
            **runtime,
        }

    monkeypatch.setattr(sandbox, "validate_sandbox_host", lambda: None)
    monkeypatch.setattr(sandbox, "_trusted_tmpdir", lambda: tmp_path)
    monkeypatch.setattr(sandbox, "isolated_program_result", fake_execution)
    receipt = sandbox.qualify_allocation()

    assert receipt["sandbox_isolation_passed"] is True
    assert receipt["resource_limit_exit_code"] == 79
    assert receipt["candidate_direct_pid_1"] is True
    assert receipt["site_packages_visible"] is False
    expected = {
        "assessor_sentinel_hidden",
        "assessor_transport_closed",
        "candidate_mount_excludes_assessor",
        "filesystem_projection_exact",
        "read_only_inputs_immutable",
        "symlink_traversal_blocked",
        "parent_environment_hidden",
        "parent_proc_hidden",
        "candidate_is_pid_1",
        "network_unreachable",
        "subprocess_blocked",
        "fork_blocked",
        "flood_bounded",
        "address_space_bounded",
        "cpu_bounded",
        "elf_projection_exact",
        "candidate_status_fd_closed",
        "trusted_completion_exit_attested",
        "system_exit_bypass_blocked",
        "os_exit_bypass_blocked",
        "failed_official_assertion_blocked",
        "status_marker_forgery_blocked",
        "candidate_policy_escape_blocked",
        "safe_import_namespace_reachability",
        "aggregate_file_creation_policy_blocked",
        "outer_wall_timeout_is_infrastructure",
        "random_seed_deterministic",
        "random_entropy_reentry_blocked",
        "python_hash_seed_effective",
        "python_runtime_descriptor_exact",
        "python_runtime_descriptor_cross_process",
        "post_candidate_setup_failure_is_scientific",
        "setup_compile_failure_is_infrastructure",
        "policy_rejection_runs_in_sandbox",
    }
    assert expected <= receipt["probe_results"].keys()
    joined = "\n".join(
        value
        for call in calls
        for value in call.values()
        if isinstance(value, str) and value
    )
    for token in (
        "sentinel.read_bytes",
        'os.listdir("/")',
        "symlink_to",
        "os.environ == expected",
        "os.getpid() == 1",
        "dangerous_origins",
        "connect_ex",
        "os.fork()",
        "subprocess.run",
        "RLIMIT_FSIZE",
        "RLIMIT_AS",
        "while True: pass",
        "raise SystemExit(0)",
    ):
        assert token in joined


def test_mbpp_never_executes_without_successful_allocation_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sandbox, "_ALLOCATION_PROBE_SHA256", None)
    with pytest.raises(sandbox.PCF1SandboxError, match="not qualified"):
        sandbox.score_completion({"task": "mbpp"}, "pass")


@pytest.mark.parametrize(
    "source",
    [
        "import os\nos._exit(0)",
        "__import__('os')._exit(0)",
        "print(globals())",
        "print((1).__class__)",
        "from os import _exit\n_exit(0)",
        "from random import _os as x\nx._exit(73)",
        "import re\nre.enum.sys.exit(73)",
        "from re import enum as x\nx.sys.exit(73)",
        (
            "import string\n"
            "string.Formatter().get_field("
            '"0.__class__.__mro__[1].__subclasses__", [0], {})'
        ),
        "def leak():\n    yield g.gi_frame.f_back\ng = leak()\nnext(g)",
        "match Exception():\n    case Exception(__traceback__=value):\n        pass",
    ],
)
def test_mbpp_policy_rejects_early_exit_and_introspection(source: str) -> None:
    accepted, _ = sandbox.validate_mbpp_candidate(source)
    assert accepted is False


def test_mbpp_policy_accepts_representative_benchmark_solution() -> None:
    accepted, reason = sandbox.validate_mbpp_candidate(
        """from collections import Counter
import math

def repeated(values):
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > math.floor(1))
"""
    )
    assert (accepted, reason) == (True, "accepted")


@pytest.mark.parametrize(
    "source",
    (
        "for index in range(10000):\n    open(f'/tmp/{index}', 'w').close()",
        "import os\nos.open('/tmp/x', os.O_CREAT)",
        "import pathlib\npathlib.Path('/tmp/x').write_text('x')",
        "import tempfile\ntempfile.mkstemp()",
    ),
)
def test_candidate_policy_blocks_aggregate_file_creation(source: str) -> None:
    assert sandbox.validate_mbpp_candidate(source)[0] is False


def test_mbpp_policy_preserves_representative_class_mechanics() -> None:
    accepted, reason = sandbox.validate_mbpp_candidate("""import math

class Base:
    def __init__(self, value):
        self.value = value

class Child(Base):
    def __init__(self, value):
        super().__init__(value)

    def __lt__(self, other):
        return self.value < other.value

def classify(value):
    item = Child(math.floor(value))
    return type(item) is Child
""")
    assert (accepted, reason) == (True, "accepted")


def test_mbpp_policy_preserves_public_regex_and_collection_imports() -> None:
    accepted, reason = sandbox.validate_mbpp_candidate("""import re
from collections.abc import Iterable

def words(value):
    assert isinstance(value, Iterable)
    return re.findall(r"[a-z]+", value.lower())
""")
    assert (accepted, reason) == (True, "accepted")


def test_mbpp_policy_preserves_fractions_statistics_and_string_constants() -> None:
    accepted, reason = sandbox.validate_mbpp_candidate("""from fractions import Fraction
from statistics import mean
from string import ascii_lowercase

def summarize(values):
    return Fraction(sum(values), len(values)), mean(values), ascii_lowercase[:3]
""")
    assert (accepted, reason) == (True, "accepted")


def test_mbpp_policy_preserves_nonintrospective_match_statements() -> None:
    accepted, reason = sandbox.validate_mbpp_candidate("""def classify(value):
    match value:
        case [first, *rest]:
            return first, len(rest)
        case _:
            return None
""")
    assert (accepted, reason) == (True, "accepted")


@pytest.mark.parametrize(
    "source",
    (
        "import random\nrandom.seed()",
        "import random\nrandom.seed(None)",
        "import random\nrandom.seed(1)",
        "from random import seed\nseed()",
        "import random\nrandom.Random()",
        "from random import Random\nRandom()",
        "import random\nrandom.SystemRandom().random()",
        "from random import SystemRandom\nSystemRandom().random()",
    ),
)
def test_mbpp_policy_blocks_random_entropy_reentry(source: str) -> None:
    assert sandbox.validate_mbpp_candidate(source)[0] is False


def test_system_exit_is_caught_as_scientific_failure_not_attestation() -> None:
    assert sandbox.validate_mbpp_candidate("raise SystemExit(73)") == (
        True,
        "accepted",
    )
    assert "except BaseException:" in sandbox.BOOTSTRAP_SOURCE
    assert "os._exit(74)" in sandbox.BOOTSTRAP_SOURCE
    assert sandbox.CANDIDATE_FAILURE_EXIT_CODE != sandbox.TRUSTED_COMPLETION_EXIT_CODE


def test_only_pinned_termination_states_are_scientific_outcomes() -> None:
    assert sandbox.classify_sandbox_termination(73, False) == (
        True,
        "trusted_tests_completed",
    )
    assert sandbox.classify_sandbox_termination(74, False) == (
        False,
        "candidate_execution_exception",
    )
    assert sandbox.classify_sandbox_termination(76, False) == (
        False,
        "official_test_failure",
    )
    assert sandbox.classify_sandbox_termination(77, False) == (
        False,
        "candidate_induced_setup_failure",
    )
    assert sandbox.classify_sandbox_termination(78, False) == (
        False,
        "candidate_policy_rejection",
    )
    assert sandbox.classify_sandbox_termination(
        sandbox.RESOURCE_LIMIT_EXIT_CODE, False
    ) == (
        False,
        "candidate_resource_limit",
    )
    for unexpected in (
        0,
        1,
        137,
        -signal.SIGTERM,
        -signal.SIGKILL,
        -signal.SIGXCPU,
        -signal.SIGXFSZ,
    ):
        with pytest.raises(sandbox.PCF1SandboxError, match="trusted classification"):
            sandbox.classify_sandbox_termination(unexpected, False)
    with pytest.raises(sandbox.PCF1SandboxError, match="trusted sandbox phase"):
        sandbox.classify_sandbox_termination(75, False)
    with pytest.raises(sandbox.PCF1SandboxError, match="outer wall timeout"):
        sandbox.classify_sandbox_termination(None, True)


def test_bootstrap_reserves_cpu_and_wall_signals_for_scientific_resource_limit() -> (
    None
):
    source = sandbox.BOOTSTRAP_SOURCE
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "signal" in imported
    assert "RESOURCE_LIMIT_EXIT_CODE = 79" in source
    assert (
        "def resource_limit_handler(_signum, _frame):\n    os._exit(RESOURCE_LIMIT_EXIT_CODE)"
        in source
    )
    handler_install = source.index(
        "signal.signal(signal.SIGXCPU, resource_limit_handler)"
    )
    wall_handler_install = source.index(
        "signal.signal(signal.SIGALRM, resource_limit_handler)"
    )
    wall_timer_install = source.index(
        "signal.setitimer(signal.ITIMER_REAL, candidate_timeout_seconds)"
    )
    ready = source.index('os.write(status_fd, b"PCF1_PYTHON_READY\\n")')
    policy_rejection = source.index("if not policy_passed:")
    candidate_execution = source.index("exec(candidate_code, namespace)")
    assert (
        ready
        < policy_rejection
        < handler_install
        < wall_handler_install
        < wall_timer_install
        < candidate_execution
    )
    assert sandbox.RESOURCE_PROBE_TIMEOUT_SECONDS == 2.0
    assert 'run("while True: pass\\n", timeout=RESOURCE_PROBE_TIMEOUT_SECONDS)' in Path(
        sandbox.__file__
    ).read_text(encoding="utf-8")


def test_assessor_transport_binds_candidate_wall_timeout() -> None:
    payload = json.loads(
        sandbox._assessor_transport_payload(
            "pass\n", "", "", trusted_probe=False, timeout_seconds=4.25
        )
    )
    assert payload["candidate_timeout_seconds"] == 4.25
    for invalid in (True, 0, -1, float("inf"), 61):
        with pytest.raises(sandbox.PCF1SandboxError, match="wall timeout"):
            sandbox._assessor_transport_payload(
                "pass\n", "", "", trusted_probe=False, timeout_seconds=invalid
            )


def test_execution_receipt_binds_candidate_wall_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sandbox, "validate_sandbox_host", lambda: None)
    monkeypatch.setattr(sandbox, "_trusted_tmpdir", lambda: tmp_path)
    monkeypatch.setattr(
        sandbox,
        "_sealed_read_only_memfd",
        lambda _name, _content: os.open(os.devnull, os.O_RDONLY),
    )
    descriptor = sandbox.EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR
    stdout = (
        "PCF1_RUNTIME_DESCRIPTOR "
        + json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
        + "\nPCF1_PYTHON_READY\n"
    )

    def runner(*_args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        os.write(kwargs["pass_fds"][1], b"{}")
        kwargs["stdout"].write(stdout.encode())
        kwargs["stdout"].flush()
        return subprocess.CompletedProcess([], sandbox.RESOURCE_LIMIT_EXIT_CODE)

    receipt = sandbox.isolated_program_result(
        "pass\n", 4.25, runner=runner, validate_host=False
    )
    assert receipt["candidate_timeout_seconds"] == 4.25
    assert receipt["termination_classification"] == "candidate_resource_limit"


def test_candidate_cannot_forge_reserved_resource_limit_exit() -> None:
    assert sandbox.validate_mbpp_candidate("raise SystemExit(79)\n") == (
        True,
        "accepted",
    )
    assert sandbox.validate_mbpp_candidate("import signal\n") == (False, "import")
    source = sandbox.BOOTSTRAP_SOURCE
    candidate_try = source.index("exec(candidate_code, namespace)")
    candidate_failure = source.index("os._exit(74)", candidate_try)
    setup_try = source.index("exec(setup_code, namespace)")
    assert candidate_try < candidate_failure < setup_try


def test_mbpp_requires_trusted_post_test_attestation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_execution(program: str, timeout: float, **kwargs: Any) -> dict[str, Any]:
        calls.append({"program": program, "timeout": timeout, **kwargs})
        return {
            "passed": False,
            "returncode": 0,
            "timed_out": False,
            "test_completion_attested": False,
        }

    monkeypatch.setattr(sandbox, "_ALLOCATION_PROBE_SHA256", "a" * 64)
    monkeypatch.setattr(
        sandbox,
        "_QUALIFIED_SETUP_RECEIPTS",
        {sandbox.hashlib.sha256(b"").hexdigest(): "b" * 64},
    )
    monkeypatch.setattr(sandbox, "isolated_program_result", fake_execution)
    result = sandbox.score_completion(
        {"task": "mbpp", "test_setup_code": "", "test_list": ["assert f() == 1"]},
        "def f():\n    print('PCF1_TRUSTED_TESTS_COMPLETED')\n    return 1",
    )

    assert result["correct"] is False
    assert calls[0]["require_test_completion_attestation"] is True
    assert calls[0]["validate_host"] is False
    assert calls[0]["program"].startswith("def f():")
    assert calls[0]["setup_source"] == ""
    assert calls[0]["tests_source"] == "assert f() == 1"


def test_production_setup_qualification_registers_before_mbpp_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executions: list[dict[str, Any]] = []

    def fake_execution(source: str, timeout: float, **kwargs: Any) -> dict[str, Any]:
        executions.append({"source": source, "timeout": timeout, **kwargs})
        return {
            "passed": True,
            "returncode": sandbox.TRUSTED_COMPLETION_EXIT_CODE,
            "timed_out": False,
            "test_completion_attested": True,
            "termination_classification": "trusted_tests_completed",
            "assessment_mode": (
                "trusted_setup_compile"
                if kwargs.get("trusted_setup_compile")
                else "candidate"
            ),
            "candidate_policy_failure": (
                "not_applicable_trusted_setup_compile"
                if kwargs.get("trusted_setup_compile")
                else "accepted"
            ),
        }

    monkeypatch.setattr(sandbox, "_ALLOCATION_PROBE_SHA256", "c" * 64)
    monkeypatch.setattr(sandbox, "_QUALIFIED_SETUP_RECEIPTS", {})
    monkeypatch.setattr(sandbox, "isolated_program_result", fake_execution)
    assessors = [
        {"task": "math500"},
        {"task": "mbpp", "test_setup_code": ""},
        {"task": "mbpp", "test_setup_code": ""},
        {"task": "mbpp", "test_setup_code": "value = 1\n"},
    ]

    with pytest.raises(sandbox.PCF1SandboxError, match="setup is not qualified"):
        sandbox.score_completion(
            {"task": "mbpp", "test_setup_code": "", "test_list": ["assert True"]},
            "pass",
        )
    receipts = sandbox.qualify_mbpp_assessor_setups(assessors)
    result = sandbox.score_completion(
        {"task": "mbpp", "test_setup_code": "", "test_list": ["assert True"]},
        "pass",
    )

    assert len(receipts) == 2
    assert len(sandbox._QUALIFIED_SETUP_RECEIPTS) == 2
    assert result["correct"] is True
    assert [call["setup_source"] for call in executions] == ["", "value = 1\n", ""]
    assert executions[-1]["tests_source"] == "assert True"


def test_frozen_reference_preflight_uses_trusted_mode_and_sandbox_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = "a" * 64
    monkeypatch.setattr(sandbox, "_ALLOCATION_PROBE_SHA256", "b" * 64)
    calls: list[tuple[str, dict[str, Any]]] = []

    def trusted_execution(
        source: str, _timeout: float, **kwargs: Any
    ) -> dict[str, Any]:
        calls.append((source, kwargs))
        return {
            "passed": True,
            "test_completion_attested": True,
            "termination_classification": "trusted_tests_completed",
            "assessment_mode": "trusted_reference",
            "candidate_policy_passed": True,
            "candidate_policy_failure": "not_applicable_trusted_reference",
        }

    monkeypatch.setattr(sandbox, "isolated_program_result", trusted_execution)
    setup_qualification = {
        "schema": "shohin-pcf1-mbpp-setup-qualification-v1",
        "status": "pass",
        "setup_source_sha256": sandbox.hashlib.sha256(b"").hexdigest(),
        "candidate_policy_sha256": sandbox.CANDIDATE_POLICY_SHA256,
        "sandbox_config_sha256": sandbox.SANDBOX_CONFIG_SHA256,
        "allocation_probe_sha256": "b" * 64,
        "setup_qualification_mode": "compile_only_before_candidate",
        "termination_classification": "trusted_tests_completed",
    }
    setup_qualification["receipt_sha256"] = sandbox.hashlib.sha256(
        json.dumps(setup_qualification, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    receipt = sandbox.preflight_mbpp_reference(
        {
            "identity_sha256": identity,
            "task": "mbpp",
            "code": "def f():\n    return 1",
            "test_list": ["assert f() == 1"],
            "test_setup_code": "",
        },
        split="development",
        setup_qualification=setup_qualification,
    )
    assert receipt["identity_sha256"] == identity
    assert receipt["allocation_probe_sha256"] == "b" * 64
    assert receipt["reference_assessment_mode"] == "trusted_reference"
    assert receipt["generated_candidate_policy_applied"] is False
    assert receipt["termination_classification"] == "trusted_tests_completed"
    assert calls[0][1]["trusted_reference"] is True


def test_trusted_reference_bypasses_candidate_grammar_inside_sandbox() -> None:
    candidate = "import sys\nassert sys.maxsize > 1\n"
    assert sandbox.validate_mbpp_candidate(candidate) == (False, "import")
    assessor = json.loads(
        sandbox._assessor_transport_payload(
            candidate,
            "",
            "assert True",
            trusted_probe=False,
            trusted_reference=True,
        )
    )
    assert assessor["assessment_mode"] == "trusted_reference"
    assert assessor["candidate_policy_passed"] is True
    assert assessor["candidate_policy_failure"] == "not_applicable_trusted_reference"


def test_trusted_setup_compile_transport_never_treats_setup_as_candidate() -> None:
    assessor = json.loads(
        sandbox._assessor_transport_payload(
            "pass\n",
            "root = Node(1)\n",
            "assert True",
            trusted_probe=False,
            trusted_setup_compile=True,
        )
    )
    assert assessor["assessment_mode"] == "trusted_setup_compile"
    assert assessor["candidate_policy_passed"] is True
    assert (
        assessor["candidate_policy_failure"] == "not_applicable_trusted_setup_compile"
    )
    assert assessor["setup_source"] == "root = Node(1)\n"
    assert assessor["tests_source"] == "assert True"


def test_policy_rejection_is_scientific_incorrect_not_infrastructure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_execution(source: str, _timeout: float, **kwargs: Any) -> dict[str, Any]:
        calls.append((source, kwargs))
        return {
            "passed": False,
            "returncode": sandbox.POLICY_REJECTION_EXIT_CODE,
            "timed_out": False,
            "candidate_policy_passed": False,
            "candidate_policy_failure": "import",
            "test_completion_attested": False,
            "infrastructure_failure": False,
            "termination_classification": "candidate_policy_rejection",
        }

    monkeypatch.setattr(sandbox, "_ALLOCATION_PROBE_SHA256", "a" * 64)
    monkeypatch.setattr(
        sandbox,
        "_QUALIFIED_SETUP_RECEIPTS",
        {sandbox.hashlib.sha256(b"").hexdigest(): "b" * 64},
    )
    monkeypatch.setattr(sandbox, "isolated_program_result", fake_execution)
    result = sandbox.score_completion(
        {"task": "mbpp", "test_setup_code": "", "test_list": ["assert False"]},
        "import os\nos._exit(0)",
    )

    assert result["correct"] is False
    assert result["execution"]["candidate_policy_passed"] is False
    assert result["execution"]["infrastructure_failure"] is False
    assert calls == [
        (
            "import os\nos._exit(0)",
            {
                "setup_source": "",
                "tests_source": "assert False",
                "validate_host": False,
                "require_test_completion_attestation": True,
            },
        )
    ]
