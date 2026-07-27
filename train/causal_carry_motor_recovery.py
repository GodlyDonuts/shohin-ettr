#!/usr/bin/env python3
# ruff: noqa: E402
"""Dual-provenance recovery for the sealed a0c258e carry feature shards.

This module never rewrites or publishes into the upstream canonical root.  It
treats that root as immutable scientific input and records the independently
reviewed recovery executor as a second provenance domain.
"""

from __future__ import annotations

import argparse
import base64
import collections
import copy
import fcntl
import functools
import hashlib
import importlib
import inspect
import io
import json
import math
import os
import re
import shlex
import stat
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path


_PREIMPORT_UPSTREAM_COMMIT = "a0c258e6709766c643cf127a429a7d6ef4a4211b"
_PREIMPORT_RECOVERY_SOURCE_PATHS = (
    "R12_CAUSAL_CARRY_MOTOR_RECOVERY_PREREG.md",
    "train/causal_carry_motor_recovery.py",
    "train/test_causal_carry_motor_recovery.py",
    "train/jobs/causal_carry_motor_recovery.sbatch",
)
_PREIMPORT_GIT = Path("/usr/bin/git")


def _preimport_stat_identity(observed):
    return (
        observed.st_dev,
        observed.st_ino,
        stat.S_IFMT(observed.st_mode),
        stat.S_IMODE(observed.st_mode),
        observed.st_nlink,
        observed.st_uid,
        observed.st_gid,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )


def _preimport_read_regular(path, label, *, required_mode=None, one_link=True):
    expected = Path(path)
    before = os.stat(expected, follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or (one_link and before.st_nlink != 1)
        or (
            required_mode is not None
            and stat.S_IMODE(before.st_mode) != int(required_mode)
        )
    ):
        raise ValueError(f"{label} is not an exact regular file")
    descriptor = os.open(expected, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if _preimport_stat_identity(opened) != _preimport_stat_identity(before):
            raise RuntimeError(f"{label} changed during descriptor binding")
        chunks = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        linked = os.stat(expected, follow_symlinks=False)
        if _preimport_stat_identity(linked) != _preimport_stat_identity(opened):
            raise RuntimeError(f"{label} changed while reading")
        return b"".join(chunks), _preimport_stat_identity(opened)
    finally:
        os.close(descriptor)


def _preimport_exact_directory(path, label):
    expected = Path(path)
    if (
        not expected.is_absolute()
        or os.path.normpath(os.fspath(expected)) != os.fspath(expected)
        or expected.resolve(strict=True) != expected
    ):
        raise ValueError(f"{label} is not an exact physical directory")
    before = os.stat(expected, follow_symlinks=False)
    if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise ValueError(f"{label} is not a regular directory")
    descriptor = os.open(
        expected,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if _preimport_stat_identity(opened) != _preimport_stat_identity(before):
            raise RuntimeError(f"{label} changed during descriptor binding")
    finally:
        os.close(descriptor)
    return expected


def _preimport_git_layout(root):
    root = _preimport_exact_directory(root, "recovery source root")
    dot_git = root / ".git"
    observed = os.stat(dot_git, follow_symlinks=False)
    if stat.S_ISDIR(observed.st_mode) and not stat.S_ISLNK(observed.st_mode):
        git_dir = _preimport_exact_directory(dot_git, "Git directory")
        pointer = None
    elif stat.S_ISREG(observed.st_mode) and not stat.S_ISLNK(observed.st_mode):
        payload, pointer_identity = _preimport_read_regular(
            dot_git, "Git worktree pointer"
        )
        try:
            text = payload.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("Git worktree pointer is not ASCII") from exc
        match = re.fullmatch(r"gitdir: (/[^\r\n]+)\n", text)
        if match is None:
            raise ValueError("Git worktree pointer is not exact")
        git_dir = _preimport_exact_directory(match.group(1), "Git worktree directory")
        pointer = {
            "path": str(dot_git),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "identity": pointer_identity,
        }
    else:
        raise ValueError("recovery source .git control is unsupported")

    commondir_path = git_dir / "commondir"
    try:
        commondir_payload, commondir_identity = _preimport_read_regular(
            commondir_path, "Git commondir"
        )
    except FileNotFoundError:
        common_dir = git_dir
        commondir = None
    else:
        try:
            commondir_text = commondir_payload.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("Git commondir is not ASCII") from exc
        if not re.fullmatch(r"[^\r\n]+\n", commondir_text):
            raise ValueError("Git commondir is not exact")
        raw = commondir_text[:-1]
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = Path(os.path.normpath(git_dir / candidate))
        common_dir = _preimport_exact_directory(candidate, "Git common directory")
        commondir = {
            "path": str(commondir_path),
            "sha256": hashlib.sha256(commondir_payload).hexdigest(),
            "identity": commondir_identity,
        }
    objects = _preimport_exact_directory(common_dir / "objects", "Git object directory")
    return {
        "root": root,
        "git_dir": git_dir,
        "common_dir": common_dir,
        "objects": objects,
        "pointer": pointer,
        "commondir": commondir,
    }


def _preimport_resolve_head(layout):
    payload, identity = _preimport_read_regular(layout["git_dir"] / "HEAD", "Git HEAD")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("Git HEAD is not ASCII") from exc
    if re.fullmatch(r"[0-9a-f]{40}\n", text):
        commit = text[:-1]
        reference = None
    else:
        match = re.fullmatch(r"ref: (refs/[A-Za-z0-9._/-]+)\n", text)
        if match is None or ".." in match.group(1).split("/"):
            raise ValueError("Git HEAD reference is not exact")
        reference = match.group(1)
        loose = layout["common_dir"] / reference
        try:
            ref_payload, _ref_identity = _preimport_read_regular(loose, "Git HEAD ref")
        except FileNotFoundError:
            packed_payload, _packed_identity = _preimport_read_regular(
                layout["common_dir"] / "packed-refs", "Git packed refs"
            )
            commit = None
            for raw_line in packed_payload.splitlines():
                if not raw_line or raw_line.startswith((b"#", b"^")):
                    continue
                try:
                    line = raw_line.decode("ascii")
                except UnicodeDecodeError as exc:
                    raise ValueError("Git packed refs are not ASCII") from exc
                packed_match = re.fullmatch(
                    r"([0-9a-f]{40}) (refs/[A-Za-z0-9._/-]+)", line
                )
                if packed_match is None:
                    raise ValueError("Git packed refs contain a malformed entry")
                if packed_match.group(2) == reference:
                    if commit is not None:
                        raise ValueError("Git packed refs duplicate HEAD")
                    commit = packed_match.group(1)
            if commit is None:
                raise ValueError("Git HEAD reference is unresolved")
        else:
            try:
                ref_text = ref_payload.decode("ascii")
            except UnicodeDecodeError as exc:
                raise ValueError("Git HEAD ref is not ASCII") from exc
            if not re.fullmatch(r"[0-9a-f]{40}\n", ref_text):
                raise ValueError("Git HEAD ref is not exact")
            commit = ref_text[:-1]
    return commit, {
        "path": str(layout["git_dir"] / "HEAD"),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "identity": identity,
        "reference": reference,
    }


class _ConfigExcludedGit:
    """Read Git objects/index through a synthetic config-free bare repository."""

    def __init__(self, root, *, copy_index):
        self.layout = _preimport_git_layout(root)
        self.head_commit, self.head_binding = _preimport_resolve_head(self.layout)
        self._temporary = tempfile.TemporaryDirectory(
            prefix="carry-recovery-git-", dir="/tmp"
        )
        temporary = Path(self._temporary.name)
        os.chmod(temporary, 0o700)
        self.synthetic_git_dir = temporary / "git"
        self.synthetic_git_dir.mkdir(mode=0o700)
        (self.synthetic_git_dir / "objects").mkdir(mode=0o700)
        (self.synthetic_git_dir / "refs").mkdir(mode=0o700)
        (temporary / "home").mkdir(mode=0o700)
        (temporary / "xdg").mkdir(mode=0o700)
        self._write_synthetic("HEAD", (self.head_commit + "\n").encode("ascii"))
        self.index_binding = None
        if copy_index:
            index_payload, index_identity = _preimport_read_regular(
                self.layout["git_dir"] / "index", "Git worktree index"
            )
            self._write_synthetic("index", index_payload)
            self.index_binding = {
                "path": str(self.layout["git_dir"] / "index"),
                "sha256": hashlib.sha256(index_payload).hexdigest(),
                "identity": index_identity,
            }
        self.environment = {
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_COUNT": "0",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_LITERAL_PATHSPECS": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OBJECT_DIRECTORY": str(self.layout["objects"]),
            "GIT_OPTIONAL_LOCKS": "0",
            "HOME": str(temporary / "home"),
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "XDG_CONFIG_HOME": str(temporary / "xdg"),
        }
        executable = os.stat(_PREIMPORT_GIT, follow_symlinks=False)
        if not stat.S_ISREG(executable.st_mode) or stat.S_ISLNK(executable.st_mode):
            self.close()
            raise ValueError("pinned Git plumbing executable is unavailable")

    def _write_synthetic(self, name, payload):
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.synthetic_git_dir / name, flags, 0o600)
        try:
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def run(self, *arguments, input_bytes=None):
        command = [
            str(_PREIMPORT_GIT),
            "--no-pager",
            f"--git-dir={self.synthetic_git_dir}",
            *arguments,
        ]
        return subprocess.check_output(
            command,
            input=input_bytes,
            stderr=subprocess.STDOUT,
            env=self.environment,
        )

    def commit(self, oid):
        if not re.fullmatch(r"[0-9a-f]{40}", str(oid)):
            raise ValueError("Git commit id is malformed")
        return self.run("cat-file", "commit", str(oid))

    def blob(self, oid):
        if not re.fullmatch(r"[0-9a-f]{40}", str(oid)):
            raise ValueError("Git blob id is malformed")
        return self.run("cat-file", "blob", str(oid))

    def tree(self, commit):
        payload = self.run("ls-tree", "-rz", "--full-tree", str(commit))
        entries = {}
        for raw in payload.split(b"\0"):
            if not raw:
                continue
            try:
                metadata, path_bytes = raw.split(b"\t", 1)
                mode, kind, oid = metadata.decode("ascii").split(" ")
                path = path_bytes.decode("utf-8")
            except (UnicodeDecodeError, ValueError) as exc:
                raise ValueError("Git tree contains a malformed entry") from exc
            if (
                kind != "blob"
                or mode not in {"100644", "100755"}
                or not re.fullmatch(r"[0-9a-f]{40}", oid)
                or path in entries
                or not path
                or path.startswith("/")
                or ".." in Path(path).parts
            ):
                raise ValueError("Git tree is outside the closed-world file contract")
            entries[path] = (mode, oid)
        return entries

    def index(self):
        if self.index_binding is None:
            raise ValueError("Git index was not bound")
        payload = self.run("ls-files", "--stage", "-z")
        entries = {}
        for raw in payload.split(b"\0"):
            if not raw:
                continue
            try:
                metadata, path_bytes = raw.split(b"\t", 1)
                mode, oid, stage = metadata.decode("ascii").split(" ")
                path = path_bytes.decode("utf-8")
            except (UnicodeDecodeError, ValueError) as exc:
                raise ValueError("Git index contains a malformed entry") from exc
            if (
                stage != "0"
                or mode not in {"100644", "100755"}
                or not re.fullmatch(r"[0-9a-f]{40}", oid)
                or path in entries
            ):
                raise ValueError("Git index is not an exact stage-zero file index")
            entries[path] = (mode, oid)
        return entries

    def close(self):
        temporary = getattr(self, "_temporary", None)
        if temporary is not None:
            temporary.cleanup()
            self._temporary = None


def _preimport_commit_headers(payload, label):
    try:
        header = payload.split(b"\n\n", 1)[0].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} commit header is not UTF-8") from exc
    tree = None
    parents = []
    for line in header.splitlines():
        if line.startswith("tree "):
            if tree is not None or not re.fullmatch(r"tree [0-9a-f]{40}", line):
                raise ValueError(f"{label} commit tree header is malformed")
            tree = line[5:]
        elif line.startswith("parent "):
            if not re.fullmatch(r"parent [0-9a-f]{40}", line):
                raise ValueError(f"{label} commit parent header is malformed")
            parents.append(line[7:])
    if tree is None:
        raise ValueError(f"{label} commit has no tree")
    return tree, parents


def _preimport_tree_topology(repository, recovery_commit):
    _tree, parents = _preimport_commit_headers(
        repository.commit(recovery_commit), "recovery"
    )
    if parents != [_PREIMPORT_UPSTREAM_COMMIT]:
        raise ValueError("recovery commit must have a0c258e as its sole direct parent")
    _parent_tree, upstream_parents = _preimport_commit_headers(
        repository.commit(_PREIMPORT_UPSTREAM_COMMIT), "upstream"
    )
    if len(upstream_parents) > 1:
        raise ValueError("upstream source commit has an ambiguous parent topology")
    recovery_tree = repository.tree(recovery_commit)
    upstream_tree = repository.tree(_PREIMPORT_UPSTREAM_COMMIT)
    additions = sorted(set(recovery_tree) - set(upstream_tree))
    removed = set(upstream_tree) - set(recovery_tree)
    modified = {
        path
        for path in set(recovery_tree) & set(upstream_tree)
        if recovery_tree[path] != upstream_tree[path]
    }
    if (
        additions != sorted(_PREIMPORT_RECOVERY_SOURCE_PATHS)
        or removed
        or modified
        or any(recovery_tree[name][0] != "100644" for name in additions)
    ):
        raise ValueError(
            "recovery commit must contain exactly four added recovery files"
        )
    return recovery_tree, upstream_tree


def _preimport_hash_checkout(root, expected_tree):
    expected_directories = set()
    for path in expected_tree:
        parent = Path(path).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    observed_files = set()
    observed_directories = set()
    root_fd = os.open(
        root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )

    def walk(directory_fd, prefix):
        for name in sorted(os.listdir(directory_fd)):
            if not prefix and name == ".git":
                continue
            relative = f"{prefix}/{name}" if prefix else name
            linked = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(linked.st_mode) and not stat.S_ISLNK(linked.st_mode):
                child_fd = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_fd,
                )
                try:
                    opened = os.fstat(child_fd)
                    if _preimport_stat_identity(opened) != _preimport_stat_identity(
                        linked
                    ):
                        raise RuntimeError("recovery checkout directory changed")
                    observed_directories.add(relative)
                    walk(child_fd, relative)
                    current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    if _preimport_stat_identity(current) != _preimport_stat_identity(
                        opened
                    ):
                        raise RuntimeError("recovery checkout directory was retargeted")
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(linked.st_mode) or stat.S_ISLNK(linked.st_mode):
                raise ValueError("recovery checkout contains a non-regular leaf")
            expected = expected_tree.get(relative)
            if expected is None:
                raise ValueError("recovery checkout is not closed-world")
            mode, oid = expected
            required_mode = 0o755 if mode == "100755" else 0o644
            if stat.S_IMODE(linked.st_mode) != required_mode or linked.st_nlink != 1:
                raise ValueError("recovery checkout file identity mismatch")
            file_fd = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            try:
                opened = os.fstat(file_fd)
                if _preimport_stat_identity(opened) != _preimport_stat_identity(linked):
                    raise RuntimeError("recovery checkout file changed during binding")
                digest = hashlib.sha1(
                    f"blob {opened.st_size}\0".encode("ascii"), usedforsecurity=False
                )
                while True:
                    block = os.read(file_fd, 1024 * 1024)
                    if not block:
                        break
                    digest.update(block)
                current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if _preimport_stat_identity(current) != _preimport_stat_identity(
                    opened
                ):
                    raise RuntimeError("recovery checkout file changed while hashing")
                if digest.hexdigest() != oid:
                    raise ValueError("recovery source checkout is not clean")
            finally:
                os.close(file_fd)
            observed_files.add(relative)

    try:
        walk(root_fd, "")
    finally:
        os.close(root_fd)
    if (
        observed_files != set(expected_tree)
        or observed_directories != expected_directories
    ):
        raise ValueError("recovery checkout is not closed-world")


def _preimport_option(name):
    matches = [
        sys.argv[index + 1]
        for index, value in enumerate(sys.argv[:-1])
        if value == name
    ]
    if len(matches) != 1:
        raise ValueError(f"recovery startup requires exactly one {name}")
    return matches[0]


def _preimport_validate_and_release_sys_path():
    root = Path(__file__).parent.parent
    if os.fspath(root) != str(root.resolve(strict=True)):
        raise ValueError("recovery script path does not identify an exact source root")
    for raw_path in sys.path:
        if not raw_path:
            raise ValueError(
                "recovery startup inherited the current directory on sys.path"
            )
        candidate = Path(raw_path).resolve(strict=False)
        if candidate == root or root in candidate.parents:
            raise ValueError(
                "repository path entered sys.path before closed-world validation"
            )
    recovery_commit = _preimport_option("--recovery-source-commit")
    manifest_receipt = _preimport_option("--recovery-source-manifest-sha256")
    if (
        not re.fullmatch(r"[0-9a-f]{40}", recovery_commit)
        or recovery_commit == _PREIMPORT_UPSTREAM_COMMIT
        or not re.fullmatch(r"[0-9a-f]{64}", manifest_receipt)
    ):
        raise ValueError("recovery startup source receipts are malformed")
    repository = _ConfigExcludedGit(root, copy_index=True)
    try:
        if repository.head_commit != recovery_commit:
            raise ValueError("reviewed recovery commit is not checked out")
        recovery_tree, _upstream_tree = _preimport_tree_topology(
            repository, recovery_commit
        )
        if repository.index() != recovery_tree:
            raise ValueError("recovery source index differs from reviewed commit")
        _preimport_hash_checkout(root, recovery_tree)
        sources = {}
        for name in _PREIMPORT_RECOVERY_SOURCE_PATHS:
            mode, oid = recovery_tree[name]
            if mode != "100644":
                raise ValueError(f"recovery source Git mode mismatch: {name}")
            committed = repository.blob(oid)
            working, _identity = _preimport_read_regular(
                root / name,
                f"recovery source {name}",
                required_mode=0o644,
            )
            if working != committed:
                raise ValueError(f"recovery source differs from commit: {name}")
            sources[name] = hashlib.sha256(committed).hexdigest()
        payload = json.dumps(
            sources,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        if hashlib.sha256(payload).hexdigest() != manifest_receipt:
            raise ValueError("recovery startup source manifest mismatch")
    finally:
        repository.close()

    candidates = (
        root / "train",
        Path(sysconfig.get_path("stdlib")),
        Path(sysconfig.get_config_var("DESTSHARED")),
        Path(sysconfig.get_path("purelib")),
        Path(sysconfig.get_path("platlib")),
    )
    paths = []
    for candidate in candidates:
        resolved = str(candidate.resolve(strict=True))
        if resolved not in paths:
            paths.append(resolved)
    sys.path[:] = paths


if __name__ == "__main__":
    _preimport_validate_and_release_sys_path()

import torch
import tokenizers as tokenizers_module
from tokenizers import Tokenizer

import causal_carry_motor as upstream
import model as model_module


digitwise_controller_module = importlib.import_module("digitwise_controller")
digitwise_protocol_module = importlib.import_module("digitwise_protocol")
eval_suite_module = importlib.import_module("eval_suite")
probe_digitwise_workspace_module = importlib.import_module("probe_digitwise_workspace")
torch_adam_module = importlib.import_module("torch.optim.adam")
torch_adamw_module = importlib.import_module("torch.optim.adamw")
torch_c_module = importlib.import_module("torch._C")
torch_functional_module = importlib.import_module("torch.nn.functional")
torch_linear_module = importlib.import_module("torch.nn.modules.linear")
torch_module_module = importlib.import_module("torch.nn.modules.module")
torch_optimizer_module = importlib.import_module("torch.optim.optimizer")
torch_serialization_module = importlib.import_module("torch.serialization")
torch_weights_only_module = importlib.import_module("torch._weights_only_unpickler")
tokenizers_native_module = importlib.import_module("tokenizers.tokenizers")


def _stabilize_reviewed_torch_exports():
    """Resolve lazy wrappers without changing the reviewed executor environment."""
    environment = dict(os.environ)
    try:
        # Torch lazily installs optimizer hooks, imports Dynamo helpers, and
        # wraps serialization helpers on their first real call.
        torch.manual_seed(0)
        module = torch.nn.Linear(1, 1)
        state = module.state_dict()
        module.load_state_dict(state)
        module.to("cpu").eval().cpu()
        optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3, weight_decay=0.0)
        optimizer.zero_grad(set_to_none=True)
        module(torch.ones((1, 1))).sum().backward()
        optimizer.step()
        serialization_probe = io.BytesIO()
        torch.save(
            {"semantic_probe": torch.tensor([0], dtype=torch.uint8)},
            serialization_probe,
        )
        serialization_probe.seek(0)
        torch.load(serialization_probe, map_location="cpu", weights_only=True)
    finally:
        for name in set(os.environ).difference(environment):
            del os.environ[name]
        for name, value in environment.items():
            if os.environ.get(name) != value:
                os.environ[name] = value


_stabilize_reviewed_torch_exports()
del _stabilize_reviewed_torch_exports


# Capture every mutable export that can alter deserialization, optimization, or
# scientific validation before any recovery command runs.  The production
# launcher disables site startup; assert_reviewed_callable_exports() additionally
# detects mutation after import and all sensitive call sites use these objects.
REVIEWED_ADAMW = torch.optim.AdamW
REVIEWED_TORCH_LOAD = torch.load
REVIEWED_TORCH_SAVE = torch.save
REVIEWED_SAFE_GLOBALS = torch.serialization.safe_globals
REVIEWED_TORCH_VERSION_TYPE = torch.torch_version.TorchVersion
REVIEWED_CARRY_MOTOR = upstream.CarryMotor
REVIEWED_BATCH_SCHEDULE = upstream._batch_schedule
REVIEWED_FULL_VOCAB_MOTOR_LOSS = upstream.full_vocab_motor_loss
REVIEWED_INITIAL_MOTOR_STATE = upstream.initial_motor_state
REVIEWED_FIT_LINEAR_DIAGNOSTIC = upstream.fit_linear_diagnostic
REVIEWED_CANONICAL_FIT_EVIDENCE = upstream.canonical_fit_teacher_forced_evidence
REVIEWED_MERGE_FEATURE_SHARDS = upstream.merge_feature_shards
REVIEWED_VALIDATE_MOTOR_BUNDLE = (
    upstream._validate_motor_bundle_against_replayed_features
)
REVIEWED_GENERATE_FIT_ROWS = upstream.generate_fit_rows
REVIEWED_PERMUTED_CONTROL_LABELS = upstream.permuted_control_labels
REVIEWED_TENSOR_STATE_SHA256 = upstream.tensor_state_sha256
REVIEWED_GPT_CONFIG = model_module.GPTConfig
REVIEWED_LINEAR = torch.nn.Linear
REVIEWED_CROSS_ENTROPY = torch.nn.functional.cross_entropy
REVIEWED_REQUIRE_CANONICAL_CUDA = upstream.require_canonical_cuda_runtime
REVIEWED_MANUAL_SEED = torch.manual_seed
REVIEWED_CUDA_MANUAL_SEED_ALL = torch.cuda.manual_seed_all
REVIEWED_AS_TENSOR = torch.as_tensor
REVIEWED_TENSOR = torch.tensor
REVIEWED_ISFINITE = torch.isfinite
REVIEWED_RANDPERM = torch.randperm
REVIEWED_LOGSUMEXP = torch.logsumexp
REVIEWED_LOGADDEXP = torch.logaddexp
REVIEWED_STACK = torch.stack
REVIEWED_NEXTAFTER = torch.nextafter
REVIEWED_TORCH_EQUAL = torch.equal
REVIEWED_TORCH_GENERATOR = torch.Generator
REVIEWED_TORCH_NO_GRAD = torch.no_grad
REVIEWED_GET_DEFAULT_DTYPE = torch.get_default_dtype
REVIEWED_GET_FLOAT32_MATMUL_PRECISION = torch.get_float32_matmul_precision
REVIEWED_ARE_DETERMINISTIC_ALGORITHMS_ENABLED = (
    torch.are_deterministic_algorithms_enabled
)
REVIEWED_IS_DETERMINISTIC_WARN_ONLY_ENABLED = (
    torch.is_deterministic_algorithms_warn_only_enabled
)
REVIEWED_GET_NUM_THREADS = torch.get_num_threads
REVIEWED_GET_NUM_INTEROP_THREADS = torch.get_num_interop_threads
REVIEWED_CUDA_DEVICE_COUNT = torch.cuda.device_count
REVIEWED_CUDA_GET_DEVICE_NAME = torch.cuda.get_device_name
REVIEWED_STABLE_JSON_SHA256 = upstream.stable_json_sha256
REVIEWED_TOKENIZER_FROM_STR = Tokenizer.from_str
REVIEWED_ROLLOUT_EPISODE = upstream.rollout_episode
REVIEWED_ADAMW_FUNCTIONAL = torch_adamw_module.adamw
REVIEWED_ADAM_FUNCTIONAL = torch_adam_module.adam
REVIEWED_ADAMW_STEP = torch.optim.AdamW.step
REVIEWED_OPTIMIZER_ZERO_GRAD = torch.optim.Optimizer.zero_grad
REVIEWED_MODULE_CALL = torch.nn.Module.__call__
REVIEWED_MODULE_TO = torch.nn.Module.to
REVIEWED_MODULE_CPU = torch.nn.Module.cpu
REVIEWED_MODULE_EVAL = torch.nn.Module.eval
REVIEWED_MODULE_PARAMETERS = torch.nn.Module.parameters
REVIEWED_MODULE_LOAD_STATE_DICT = torch.nn.Module.load_state_dict
REVIEWED_MODULE_STATE_DICT = torch.nn.Module.state_dict
REVIEWED_TENSOR_BACKWARD = torch.Tensor.backward

_SERIALIZATION_HELPER_NAMES = tuple(
    sorted(
        {
            name
            for function in (REVIEWED_TORCH_LOAD, REVIEWED_TORCH_SAVE)
            for name in function.__code__.co_names
            if name in torch_serialization_module.__dict__
        }
    )
)
REVIEWED_SERIALIZATION_HELPERS = {
    name: torch_serialization_module.__dict__[name]
    for name in _SERIALIZATION_HELPER_NAMES
}


UPSTREAM_SOURCE_COMMIT = "a0c258e6709766c643cf127a429a7d6ef4a4211b"
UPSTREAM_SOURCE_MANIFEST_SHA256 = (
    "9ae61e1a3e8f672a71a01edc16e6a5f1f8f3c69f49afd5e97f41c6cde15350a9"
)
UPSTREAM_PLAN_SHA256 = (
    "1b845d47f6875df571169efb5adb0716dfbc5d266a2499e4a92451351a262b6d"
)
UPSTREAM_CONFIRMATION_COMMITMENT_SHA256 = (
    "1ee32e4e2e8f9eb56026b7b8de1fdff207e9fd3694e0ae354f103d58ebb820da"
)
UPSTREAM_BOARD_ROWS_SHA256 = (
    "6517b1ff3aa557e449a2eef9c5540c3d5f8699482d933d5c320b606adb4a0f1b"
)
UPSTREAM_CANONICAL_BOARD_SHA256 = (
    "d6282610ba845b23ebe849efe574233bf657a50aea0a7edb901e9e1d95b24391"
)
NORMALIZATION_MISMATCH_LEDGER_SHA256 = (
    "b43cb4a6fbfab97c659e8658f63185ae8b3dc1d8cce34089958d3b09df0593b6"
)
UPSTREAM_SHARD_SHA256 = (
    "4affa12434513ebe9587464ff38656abaaf7e47904d9db6ced252c3adea52a96",
    "4731c1644703e26c1978ca1ec1ba80af7c173c5d9676ae68fbd04368f3b54c2c",
    "e81639e68a838bfa6695be92f7c1333d100b2317c48fb2cf0d995f22a6e50a43",
    "ae86ec1b70dca21d67849fc4be17ffec682472851735c3b9523292836a74e70f",
    "ce5a151f89e20e774c7d37afc446ea026ec14a587c70fa614414f060f10a2144",
    "f02d8221bf3a393566c279e27bf888fcbd1ef9ea17bdd33262472c898950ea83",
    "009b83f0c2a70362654e3e3e4cad27d30f79f93f3bdd32d6ce3064695dd2b9db",
    "8214d356288c56a116a3de753a8948a35f731d52c520fa906f4e31c1b0f14fb4",
)

DATA_ROOT = Path("/lustre/fs1/home/sa305415/shohin")
UPSTREAM_ROOT = (
    DATA_ROOT / "artifacts" / "carry_motor" / f"canonical_{UPSTREAM_SOURCE_COMMIT}"
)
UPSTREAM_PLAN_PATH = UPSTREAM_ROOT / "plan.json"
UPSTREAM_CONFIRMATION_PATH = (
    DATA_ROOT
    / "artifacts"
    / "carry_motor"
    / "confirmation_commitments"
    / f"commitment_{UPSTREAM_SOURCE_COMMIT}"
    / "commitment.json"
)
RECOVERY_PARENT = DATA_ROOT / "artifacts" / "carry_motor" / "recoveries"
REVIEW_PARENT = DATA_ROOT / "artifacts" / "carry_motor" / "recovery_reviews"
PINNED_PYTHON_LAUNCHER = DATA_ROOT / "miniforge3" / "bin" / "python"
PINNED_GIT = Path("/usr/bin/git")
PINNED_SCONTROL = Path("/usr/bin/scontrol")
PINNED_SACCT = Path("/usr/bin/sacct")
PINNED_NVIDIA_SMI = Path("/usr/bin/nvidia-smi")

RECOVERY_PLAN_AUDIT = "causal_carry_motor_recovery_plan_v8"
RECOVERY_FIT_AUDIT = "causal_carry_motor_fit_v11_recovery_v7"
RECOVERY_REVIEW_AUDIT = "causal_carry_motor_recovery_signed_review_v8"
RECOVERY_REVIEW_STATEMENT_AUDIT = "causal_carry_motor_recovery_review_statement_v1"
RECOVERY_EXECUTOR_SOURCE_SCHEMA = "carry_motor_recovery_executor_source_v3"
RECOVERY_EXECUTOR_RUNTIME_SCHEMA = "carry_motor_recovery_executor_runtime_v4"
RECOVERY_DEPENDENCY_MANIFEST_SCHEMA = "carry_motor_recovery_dependencies_v2"
UPSTREAM_CUSTODY_SCHEMA = "carry_motor_upstream_custody_snapshot_v1"
RECOVERY_PARENT_RECEIPT_AUDIT = "carry_motor_recovery_parent_v1"
RECOVERY_PARENT_BINDING_SCHEMA = "carry_motor_recovery_parent_binding_v1"
RECOVERY_LAYOUT_RECEIPT_AUDIT = "carry_motor_recovery_layout_reservation_v1"
RECOVERY_LAYOUT_BINDING_SCHEMA = "carry_motor_recovery_layout_binding_v1"
RECOVERY_CALLABLE_CONTRACT_SCHEMA = "carry_motor_reviewed_callables_v1"
RECOVERY_GIT_CONTRACT_SCHEMA = "carry_motor_recovery_git_repository_v1"
RECOVERY_SLURM_CONTRACT_SCHEMA = "carry_motor_recovery_slurm_h100_v2"
RECOVERY_TRAJECTORY_PROOF_SCHEMA = "carry_motor_recovery_trajectory_replay_v5"
CONSTANT_BIAS_PAYLOAD_SCHEMA = "carry_motor_constant_bias_payload_v1"
CONSTANT_BIAS_CONTROL_ID = "zero_sum_delta_v1"
CONSTANT_BIAS_CLAIM_BOUNDARY = (
    "A fitted constant carry-logit threshold is a favorable calibration null only. "
    "It reads no residual feature and establishes no mechanism, autonomous capability, "
    "or reasoning result."
)
NUISANCE_ONLY_PAYLOAD_SCHEMA = "carry_motor_nuisance_only_payload_v2"
NUISANCE_ONLY_CONTROL_ID = "saturated_op_width_position_v1"
NUISANCE_METADATA_SCHEMA = "carry_motor_nuisance_metadata_v2"
NUISANCE_TRAIN_WIDTHS = (4, 6)
NUISANCE_PUBLIC_OOD_WIDTHS = (8,)
NUISANCE_FIT_CELLS = tuple(
    (operation, width, position)
    for operation in ("add", "sub")
    for width in NUISANCE_TRAIN_WIDTHS
    for position in range(width)
)
NUISANCE_METADATA_FEATURE_NAMES = tuple(
    f"{operation}_w{width}_p{position}"
    for operation, width, position in NUISANCE_FIT_CELLS
)
NUISANCE_PARAMETER_COUNT = len(NUISANCE_FIT_CELLS)
NUISANCE_CAPACITY_LEDGER = {
    "schema": "carry_motor_nuisance_capacity_ledger_v1",
    "fit_metadata_cells": NUISANCE_PARAMETER_COUNT,
    "fit_design_rows": NUISANCE_PARAMETER_COUNT,
    "fit_design_rank": NUISANCE_PARAMETER_COUNT,
    "trainable_scalar_deltas": NUISANCE_PARAMETER_COUNT,
    "intercept_parameters": 0,
    "width_extrapolation_parameters": 0,
    "total_trainable_parameters": NUISANCE_PARAMETER_COUNT,
    "cell_order": [list(cell) for cell in NUISANCE_FIT_CELLS],
    "treatment_parameter_formula": "rank*d_model + 3*rank + 2",
    "constant_bias_trainable_parameters": 1,
    "family_selection": (
        "singleton saturated_fit_cell_v1 fixed by reviewed source before width8 or "
        "confirmation"
    ),
}
NULL_OPTIMIZATION_SCHEMA = "carry_motor_full_board_convex_selection_v2"
NULL_OPTIMIZATION_BOUND = 64.0
NULL_OPTIMIZATION_BISECTION_STEPS = 80
NULL_OPTIMIZATION_GRADIENT_TOLERANCE = None
NULL_OPTIMIZATION_BRACKET_TOLERANCE = 2.0**-40
NUISANCE_ONLY_CLAIM_BOUNDARY = (
    "A fitted saturated fit-cell operation/width/position calibration arm is a "
    "favorable nuisance-only null. It receives no residual hidden state, prompt "
    "text, token history, style, current carry, or operand digits and establishes "
    "no mechanism, autonomous capability, or reasoning result."
)
NORMALIZATION_SCHEMA = "carry_motor_fit_board_strict_json_normalization_v1"
FIT_MODEL_BINDING_SCHEMA = "carry_motor_model_bound_fit_evidence_v1"
FIT_SELECTION_RAW_EVIDENCE_SCHEMA = "carry_motor_raw_fit_selection_evidence_v1"
DEPLOYMENT_VOCABULARY_SCHEMA = "carry_motor_deployment_vocabulary_v1"
CASE_GENERATOR_BINDING_SCHEMA = "carry_motor_case_generator_binding_v1"
CASE_SPLIT_RECEIPT_SCHEMA = "carry_motor_case_split_receipt_v1"
STRICT_PARAMETER_CAP = 150_000_000
EXPECTED_BASE_PARAMETER_COUNT = 125_081_664
EXPECTED_BASE_PARAMETER_CONFIG = {
    "vocab_size": 32_768,
    "n_layer": 30,
    "n_head": 9,
    "n_kv_head": 3,
    "d_model": 576,
    "d_ff": 1_536,
    "qk_norm": True,
    "tie_embeddings": True,
}
EXPECTED_EXCLUDED_NODES = (
    "evc22",
    "evc26",
    "evc31",
    "evc32",
    "evc36",
    "evc37",
    "evc40",
    "evc43",
    "evc44",
)
DESERIALIZATION_SCHEMA = "bound_weights_only_torchversion_allowlist_v1"
SECURE_CREATION_UMASK = 0o077
RECOVERY_LAYOUT_RECEIPT_NAME = "layout_receipt.json"
RECOVERY_PARENT_OWNER_NAME = ".parent_receipt.owner"
RECOVERY_PARENT_STAGE_NAME = ".parent_receipt.stage"
RECOVERY_FIT_OWNER_NAME = ".motor.pt.owner"
RECOVERY_FIT_STAGE_NAME = ".motor.pt.stage"
RECOVERY_PUBLISHER_SCHEMA = "carry_motor_recovery_publisher_owner_v1"
REVIEW_SIGNER_SEQUENCE = 1

# Public half of an externally held Ed25519 key.  No production private key or
# signing helper exists in this repository.  The fingerprint is SHA-256(raw key).
PRODUCTION_REVIEW_PUBLIC_KEY_HEX = (
    "63a0eb0b964ec482bb99857b311e45ea9b11e2fb834a8427b4c54f681e59abbb"
)
PRODUCTION_REVIEW_KEY_ID = (
    "ed25519-sha256:de00c061da12e04939933da597a399448c3cdc7136e25b29f09d3dbc3d0599d9"
)

RECOVERY_SOURCE_PATHS = (
    "R12_CAUSAL_CARRY_MOTOR_RECOVERY_PREREG.md",
    "train/causal_carry_motor_recovery.py",
    "train/test_causal_carry_motor_recovery.py",
    "train/jobs/causal_carry_motor_recovery.sbatch",
)
RECOVERY_NAME_STATUS_DIFF = tuple(
    f"A\t{name}" for name in sorted(RECOVERY_SOURCE_PATHS)
)
EXECUTOR_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "GIT_ATTR_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "LANG": "C",
    "LC_ALL": "C",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "1",
    "PATH": "/usr/local/bin:/usr/bin:/bin",
}
NUMERICAL_OVERRIDE_ENVIRONMENT = (
    "BLIS_ARCH_TYPE",
    "CUBLAS_FORCE_TF32",
    "CUBLASLT_WORKSPACE_SIZE",
    "CUDA_AUTO_BOOST",
    "CUDA_CACHE_DISABLE",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT",
    "CUDA_DEVICE_WAITS_ON_EXCEPTION",
    "CUDA_FORCE_PTX_JIT",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_MANAGED_FORCE_DEVICE_ALLOC",
    "CUDA_MODULE_LOADING",
    "CUDNN_BENCHMARK",
    "CUDNN_DETERMINISTIC",
    "KMP_AFFINITY",
    "KMP_BLOCKTIME",
    "MKL_CBWR",
    "MKL_DEBUG_CPU_TYPE",
    "MKL_ENABLE_INSTRUCTIONS",
    "MKL_SERVICE_FORCE_INTEL",
    "NVIDIA_TF32_OVERRIDE",
    "NPY_BLAS_ORDER",
    "NPY_LAPACK_ORDER",
    "OPENBLAS_CORETYPE",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    "TORCH_BLAS_PREFER_CUBLASLT",
    "TORCH_CUDNN_V8_API_DEBUG",
    "TORCH_CUDNN_V8_API_ENABLED",
    "TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT",
    "TORCH_LINALG_PREFER_CUSOLVER",
    "VECLIB_MAXIMUM_THREADS",
)
FORBIDDEN_EXECUTOR_ENVIRONMENT = (
    "BASH_ENV",
    "LD_AUDIT",
    "LD_DEBUG",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "LD_PROFILE",
    "PYTHONHOME",
    "PYTHONINSPECT",
    "PYTHONPATH",
    "PYTHONSAFEPATH",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
    "PYTHONWARNINGS",
    "SHELLOPTS",
    *NUMERICAL_OVERRIDE_ENVIRONMENT,
    "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",
    "TORCH_FORCE_WEIGHTS_ONLY_LOAD",
)
FORBIDDEN_EXECUTOR_ENVIRONMENT_PREFIXES = (
    "BASH_FUNC_",
    "BLIS_",
    "CUBLAS_",
    "CUBLASLT_",
    "CUDNN_",
    "DYLD_",
    "GIT_",
    "KMP_",
    "MKL_",
    "NCCL_",
    "NPY_BLAS_",
    "NPY_LAPACK_",
    "NUMEXPR_",
    "OMP_",
    "OPENBLAS_",
    "PYTORCH_",
    "SBATCH_",
    "TORCH_",
    "TORCHDYNAMO_",
    "TORCHINDUCTOR_",
    "VECLIB_",
)
PREFIX_ENVIRONMENT_EXCEPTIONS = frozenset(EXECUTOR_ENVIRONMENT)

GIT_CONFIGURATION_EXCLUSION = {
    "system": "GIT_CONFIG_NOSYSTEM=1 and GIT_CONFIG_SYSTEM=/dev/null",
    "global": "GIT_CONFIG_GLOBAL=/dev/null with isolated HOME/XDG_CONFIG_HOME",
    "local": "synthetic bare GIT_DIR with no config file",
    "attributes": "no work tree is attached to plumbing and GIT_ATTR_NOSYSTEM=1",
    "allowed_plumbing": ["cat-file", "ls-files --stage", "ls-tree"],
    "forbidden_porcelain": ["diff", "show", "status", "checkout", "add"],
}

EXPECTED_SLURM_REQUEST = {
    "account": "skattel",
    "partition": "normal",
    "nodes": 1,
    "tasks": 1,
    "cpus_per_task": 4,
    "memory_mib": 32768,
    "time_limit_seconds": 21600,
    "requeue": False,
    "gpu_tres_name": "nvidia_h100_pcie",
    "gpu_count": 1,
    "gpu_name": "NVIDIA H100 PCIe",
    "compute_capability": "9.0",
    "memory_total_mib_min": 80000,
    "memory_total_mib_max": 82000,
    "mig_mode": "Disabled",
    "excluded_nodes": list(EXPECTED_EXCLUDED_NODES),
}

RECOVERY_PARENT_RECEIPT_NAME = "parent_receipt.json"
RECOVERY_PARENT_RECEIPT_CLAIM_BOUNDARY = (
    "This receipt establishes only the durable directory identity used to hold "
    "recovery roots. It establishes no plan, fit, evaluation, or capability claim."
)

RECOVERY_PLAN_KEYS = frozenset(
    {
        "audit",
        "recovery",
        "recovery_plan_path",
        "recovery_executor_source_contract",
        "executor_runtime_contract",
        "recovery_parent_binding",
        "hostile_review_binding",
        "upstream_protocol",
        "normalization_proof",
        "allowed_transformation",
        "fit_contract",
        "downstream_evaluation_contract",
        "output_contract",
        "deserialization_contract",
        "claim_boundary",
    }
)
RECOVERY_FIT_KEYS = frozenset(
    {
        "audit",
        "recovery",
        "recovery_plan_sha256",
        "recovery_executor_source_contract",
        "executor_runtime_contract",
        "recovery_parent_binding",
        "recovery_layout_binding",
        "upstream_protocol_source_contract",
        "upstream_plan_binding",
        "upstream_shard_receipts",
        "normalization_proof",
        "allowed_transformation",
        "deserialization_contract",
        "slurm_h100_attestation",
        "parameter_ledger",
        "trajectory_replay_proof",
        "fit_payload",
        "constant_bias_payload",
        "nuisance_only_payload",
        "claim_boundary",
    }
)
LEGACY_PAYLOAD_KEYS = upstream.CANONICAL_FIT_KEYS - {"audit", "canonical"}

EXPECTED_NORMALIZATION_MISMATCHES = (
    {
        "path": "board.prompt_length_histogram",
        "generated_key_type": "int",
        "sealed_key_type": "str",
        "generated_keys": [97, 99, 103, 105],
        "sealed_keys": ["97", "99", "103", "105"],
    },
    {
        "path": "board.token_length_histogram",
        "generated_key_type": "int",
        "sealed_key_type": "str",
        "generated_keys": [114, 116, 120, 122],
        "sealed_keys": ["114", "116", "120", "122"],
    },
)

ALLOWED_TRANSFORMATION = {
    "schema": NORMALIZATION_SCHEMA,
    "operation": "strict_json_round_trip_of_complete_generated_fit_board",
    "changed_paths": [item["path"] for item in EXPECTED_NORMALIZATION_MISMATCHES],
    "permitted_semantic_changes": 0,
    "permitted_additional_transformations": 0,
}
DESERIALIZATION_CONTRACT = {
    "schema": DESERIALIZATION_SCHEMA,
    "weights_only": True,
    "safe_globals": ["torch.torch_version.TorchVersion"],
    "bind_before_deserialize": True,
    "ambient_override_environment_forbidden": [
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",
        "TORCH_FORCE_WEIGHTS_ONLY_LOAD",
    ],
    "fallback_to_unrestricted_pickle": False,
}
DOWNSTREAM_EVALUATION_CONTRACT = {
    "schema": "carry_motor_recovery_downstream_evaluation_v5",
    "deployment_vocabulary": {
        "schema": DEPLOYMENT_VOCABULARY_SCHEMA,
        "base_checkpoint_sha256": upstream.EXPECTED_CHECKPOINT_SHA256,
        "tokenizer_sha256": upstream.EXPECTED_TOKENIZER_SHA256,
        "output_vocab_width": EXPECTED_BASE_PARAMETER_CONFIG["vocab_size"],
        "token_order": "logit column i is exact tokenizer token id i",
        "deployment_logit_dtype": "torch.bfloat16",
    },
    "parameter_limit": {
        "strict_cap_exclusive": STRICT_PARAMETER_CAP,
        "base_unique_trainable_parameters": EXPECTED_BASE_PARAMETER_COUNT,
        "combined_base_treatment_and_null_heads_required": True,
    },
    "phase_order": [
        "fit_candidate_publication",
        "full_board_checkpoint_and_model_family_freeze",
        "fit_width_development_evaluation",
        "width8_development_evaluation",
        "single_confirmation_reveal_and_evaluation",
    ],
    "fit_metrics_are_capability_evidence": False,
    "arms": ["base", "treatment", "constant_bias", "nuisance_only", "shuffled"],
    "constant_bias_control": {
        "control_id": CONSTANT_BIAS_CONTROL_ID,
        "trainable_parameters": 1,
        "identifiable_parameter": "delta=b1-b0",
        "deployed_vector": ["-delta/2", "+delta/2"],
        "hidden_value_dependence": False,
        "width_dependence": False,
        "prompt_dependence": False,
        "token_history_dependence": False,
        "deployment_site": "exact treatment apply_motor_logits carry-token site",
        "grammar_gate": "exact treatment CarryRouter gate",
        "gate_off_full_logit_identity": "exact",
        "training_rows": "exact treatment true-label rows",
        "objective": "exact treatment full-vocabulary cross entropy",
        "optimization": {
            "schema": NULL_OPTIMIZATION_SCHEMA,
            "solver": "global convex derivative bracketing on the complete fit board",
            "deployment_arithmetic": (
                "delta cast to base01 dtype, then float32 full-vocabulary log-sum-exp"
            ),
            "candidate_neighborhood": (
                "root bracket plus adjacent deployed-dtype levels for both signed "
                "half-deltas"
            ),
            "bound": NULL_OPTIMIZATION_BOUND,
            "bisection_steps": NULL_OPTIMIZATION_BISECTION_STEPS,
            "nonconverged_policy": "fail_closed_without_final_iterate_fallback",
        },
        "optimization_access": (
            "complete fit board on every objective and gradient evaluation"
        ),
    },
    "nuisance_only_control": {
        "control_id": NUISANCE_ONLY_CONTROL_ID,
        "trainable_parameters": NUISANCE_PARAMETER_COUNT,
        "identifiable_parameter": "one delta per saturated fit metadata cell",
        "deployed_vector": ["-delta(metadata)/2", "+delta(metadata)/2"],
        "allowed_metadata": ["operation", "width", "position"],
        "feature_names": list(NUISANCE_METADATA_FEATURE_NAMES),
        "capacity_ledger": NUISANCE_CAPACITY_LEDGER,
        "hidden_residual_dependence": False,
        "prompt_text_dependence": False,
        "token_history_dependence": False,
        "style_dependence": False,
        "current_carry_dependence": False,
        "operand_digit_dependence": False,
        "fit_widths": list(NUISANCE_TRAIN_WIDTHS),
        "public_width_ood": list(NUISANCE_PUBLIC_OOD_WIDTHS),
        "confirmation_width_ood": "every generated width outside fit widths 4 and 6",
        "width_rule": (
            "saturated fit cells with normalized-position interpolation and "
            "immutable linear width extrapolation from widths 4 and 6"
        ),
        "deployment_site": "exact treatment apply_motor_logits carry-token site",
        "grammar_gate": "exact treatment CarryRouter gate",
        "gate_off_full_logit_identity": "exact",
        "training_rows": "exact treatment true-label rows",
        "objective": "exact treatment full-vocabulary cross entropy",
        "optimization": {
            "schema": NULL_OPTIMIZATION_SCHEMA,
            "objective": "exact full-board full-vocabulary cross entropy",
            "solver": "independent convex derivative bracketing per fit cell",
            "deployment_arithmetic": (
                "delta cast to base01 dtype, then float32 full-vocabulary log-sum-exp"
            ),
            "candidate_neighborhood": (
                "root bracket plus adjacent deployed-dtype levels for both signed "
                "half-deltas"
            ),
            "bound": NULL_OPTIMIZATION_BOUND,
            "bisection_steps": NULL_OPTIMIZATION_BISECTION_STEPS,
            "gradient_tolerance": NULL_OPTIMIZATION_GRADIENT_TOLERANCE,
            "bracket_tolerance": NULL_OPTIMIZATION_BRACKET_TOLERANCE,
            "checkpoint_selection": (
                "minimum recomputed full-board CE, then minimum absolute state, "
                "then lexicographic float32 state bytes"
            ),
            "nonconverged_policy": "fail_closed_without_final_iterate_fallback",
        },
        "optimization_access": (
            "complete fit board on every objective and gradient evaluation; no "
            "minibatch noise and no final-iterate fallback"
        ),
        "confirmation_fit_access": False,
        "width8_selection_access": False,
    },
    "selection_boundary": {
        "model_family": "singleton saturated_fit_cell_v1",
        "model_family_frozen_by": "reviewed recovery source",
        "checkpoint_objective": "fit-board full-vocabulary cross entropy",
        "fit_width_development_role": "audit_only_no_reselection",
        "width8_role": "post-freeze evaluation_only",
        "confirmation_role": "post-freeze evaluation_only",
        "fallback_after_nonconvergence": False,
    },
    "required_case_coverage": {
        "matched_positive_cases": 9,
        "matched_negative_cases": 9,
        "carry_commit_rescue_cases": 6,
        "development_episodes_per_arm": 300,
        "development_boundary_cycle_cases_per_arm": 50,
        "development_direct_interactions_per_arm": 12,
        "confirmation_episodes_per_arm": 256,
        "missing_arm_or_case_policy": "fail_closed",
    },
    "required_scoring": {
        "matched_oracles": "full-vocabulary greedy decode with exact grammar gate",
        "development": "full-vocabulary autonomous greedy decode",
        "confirmation": "full-vocabulary autonomous greedy decode",
        "report_per_regime_and_width": True,
        "confirmation_transition_strata": ["operation", "width", "position"],
        "confirmation_episode_strata": ["regime", "operation", "width"],
        "all_manifest_strata_required": True,
        "all_nonempty_fit_development_confirmation_strata_required": True,
        "treatment_must_beat_both_calibration_nulls_in_every_stratum": True,
        "pairwise_c0_c1_probe_may_satisfy_gate": False,
        "teacher_forced_metric_may_satisfy_gate": False,
    },
    "raw_margin_diagnostic": {
        "margin": "m=logit(c1)-logit(c0)",
        "constant_shift": "m_after=m+delta",
        "all_positive_and_negative_feasibility_condition": (
            "-min(m_positive) < delta < -max(m_negative)"
        ),
        "interval_endpoints": "exclusive",
        "report_binary_accuracy_optimal_delta": True,
        "fit_diagnostic_is_capability_evidence": False,
    },
    "recovery_v4_calibration_boundary": {
        "artifact": (
            "artifacts/eval_history/drs_carry_nuisance_audit_20260718_v4.json"
        ),
        "artifact_sha256": (
            "94bf0b4b61b239601a7677f7badca03ac9b507c3aad6616b80d37f11072c7f68"
        ),
        "operation_width_fit_optimal_value_ood_sensitivity": {
            "before_correct": 11,
            "after_correct": 16,
            "denominator": 16,
        },
        "binary_margin_cross_entropy": {"correct": 14, "denominator": 16},
        "production_full_vocabulary_objective": False,
        "candidate_fit_or_selection_authority": False,
        "downstream_gate_authority": False,
        "claim_boundary": "immutable calibration boundary only",
    },
    "pre_fit_pairwise_constant_bias_audit": {
        "artifact": (
            "artifacts/eval_history/drs_carry_constant_bias_audit_20260718.json"
        ),
        "artifact_sha256": (
            "7f2eef8843eb686c2b63683ab7f11a248b5e1b8c8a4358c936a6c2d49326b7b3"
        ),
        "source_probe_sha256": (
            "c3c2d0b037852cb57d54e1f147d445d27093a8548b965c41466e81bcc1a27778"
        ),
        "layer": 29,
        "raw_pairwise": {
            "correct": 32,
            "total": 40,
            "target_0_correct": 13,
            "target_0_total": 20,
            "target_1_correct": 19,
            "target_1_total": 20,
        },
        "favorable_constant": {
            "representative_delta": -0.7841806411743164,
            "correct": 35,
            "total": 40,
            "target_0_correct": 18,
            "target_0_total": 20,
            "target_1_correct": 17,
            "target_1_total": 20,
        },
        "perfect_constant_feasibility": {
            "feasible": False,
            "lower_open": 0.6561751365661621,
            "upper_open": -1.1492173671722412,
        },
        "bind_probe_code_bytes": False,
        "sufficient_autonomous_gate": False,
        "claim_boundary": (
            "Pairwise grammar-gated c0/c1 constant-calibration audit only; not "
            "full-vocabulary decoding, a hidden-state reader, autonomous execution, "
            "or reasoning."
        ),
    },
    "unreviewed_width_calibration_diagnostic": {
        "decision_authority": False,
        "independent_exact_reproduction_required": True,
        "oracle_per_width": {
            "uses_ood_labels": True,
            "admissible_control": False,
            "correct": 38,
            "total": 40,
            "fit_w4_correct": 15,
            "fit_w4_total": 16,
            "fit_w6_correct": 15,
            "fit_w6_total": 16,
            "width_ood_w8_correct": 8,
            "width_ood_w8_total": 8,
        },
        "fit_only_width_affine_guidance": {
            "fit_w4_delta": -0.2242728621,
            "fit_w6_delta": -1.4803290665,
            "affine_extrapolated_w8_delta": -2.736385271,
            "fit_w4_correct": 8,
            "fit_w4_total": 8,
            "fit_w6_correct": 8,
            "fit_w6_total": 8,
            "value_ood_w4_correct": 7,
            "value_ood_w4_total": 8,
            "value_ood_w6_correct": 7,
            "value_ood_w6_total": 8,
            "width_ood_w8_correct": 5,
            "width_ood_w8_total": 8,
            "correct": 35,
            "total": 40,
        },
        "fresh_fit_only_nuisance_audit": {
            "selection_rows": ["fit_w4", "fit_w6"],
            "ood_labels_used_for_selection": False,
            "global_constant": {
                "delta": -0.7778145075,
                "fit_correct": 15,
                "fit_total": 16,
                "regime_correct_of_8": [8, 7, 6, 7, 7],
                "correct": 35,
                "total": 40,
            },
            "op_only": {
                "add_delta": -0.7778145075,
                "sub_delta": -0.4535870552,
                "add_fit_correct": 10,
                "add_fit_total": 10,
                "sub_fit_correct": 5,
                "sub_fit_total": 6,
                "width_ood_w8_correct": 7,
                "width_ood_w8_total": 8,
                "correct": 35,
                "total": 40,
            },
            "op_by_width": {
                "add_w4_delta": 0.0,
                "add_w6_delta": -0.7778145075,
                "sub_w4_delta": 0.0,
                "sub_w6_delta_rule": "nextafter(-1.1492173672,-inf)",
                "fit_correct": 16,
                "fit_total": 16,
                "value_ood_w4_correct": 7,
                "value_ood_w4_total": 8,
                "value_ood_w6_correct": 8,
                "value_ood_w6_total": 8,
                "linear_w8_diagnostic_correct": 6,
                "linear_w8_diagnostic_total": 8,
                "public_w8_extrapolator_was_preregistered": False,
            },
        },
        "public_width_ood_is_now_unopened": False,
        "may_satisfy_confirmation_gate": False,
        "frozen_arm_rule_before_confirmation_reveal": True,
        "claim_boundary": (
            "Unreviewed pairwise calibration guidance only. Oracle per-width deltas "
            "are forbidden as a deployed control; public width OOD is no longer an "
            "unopened scientific gate."
        ),
    },
    "development": {
        "episodes": 300,
        "boundary_cycle_cases": 50,
        "researcher_direct_interactions": 12,
        "selection": "frozen a0c258e source-order development selection",
    },
    "confirmation": {
        "episodes": 256,
        "reveals": 1,
        "generator": "frozen a0c258e secret-derived confirmation generator",
    },
    "carry_commit_rescue_set": {
        "source_replay_sha256": (
            "756911f568c12093f3a303a42525a2519c38187c8eac71f5da3ca06ac1ce3b20"
        ),
        "cases": [
            {
                "id": "width_ood_w8-00175",
                "branch": "counterfactual",
                "operation": "add",
                "expected_answer": 177453123,
            },
            {
                "id": "width_ood_w8-00207",
                "branch": "normal",
                "operation": "add",
                "expected_answer": 176477219,
            },
            {
                "id": "width_ood_w8-00209",
                "branch": "normal",
                "operation": "add",
                "expected_answer": 169264069,
            },
            {
                "id": "width_ood_w8-00219",
                "branch": "normal",
                "operation": "add",
                "expected_answer": 187969887,
            },
            {
                "id": "width_ood_w8-00219",
                "branch": "counterfactual",
                "operation": "add",
                "expected_answer": 187969888,
            },
            {
                "id": "width_ood_w8-00242",
                "branch": "normal",
                "operation": "add",
                "expected_answer": 164377525,
            },
        ],
        "required_treatment_rescues": 6,
        "required_treatment_gain_over_constant_bias_cases": 2,
        "required_treatment_gain_over_nuisance_only_cases": 2,
        "allowed_shuffled_rescues": 0,
        "allowed_new_prefix_divergences": 0,
    },
    "terminal_width_sweep_oracle": {
        "artifact": (
            "artifacts/eval_history/"
            "drs_terminal_width_sweep_v2_w2_w10_20260718_mps.json"
        ),
        "artifact_sha256": (
            "db6056e66310ed7d56509403d40f7549d016294a014c0c4527173b4005210520"
        ),
        "superseded_diagnostics": [
            {
                "artifact": (
                    "artifacts/eval_history/"
                    "drs_terminal_width_sweep_w2_w10_20260718_mps.json"
                ),
                "artifact_sha256": (
                    "c9670853040349cce4eb4f89c5d5d8381d7b25494ff4428fd873fc2b7be6098d"
                ),
                "reason": "strict-parser nulls overcounted field errors",
                "decision_authority": False,
            },
            {
                "artifact": (
                    "artifacts/eval_history/"
                    "manual_drs_carry_serializer_probe_v2_20260718_mps.json"
                ),
                "artifact_sha256": (
                    "b1cafe345bad726517e4c426596c691bf3ae1133d93619af581927ca7a336806"
                ),
                "reason": "narrow historical probe superseded by matched widths",
                "decision_authority": False,
            },
        ],
        "protocol": "frozen raw-field DRS transition and serializer reanalysis",
        "construction": {
            "widths": [2, 3, 4, 5, 6, 7, 8, 9, 10],
            "lower_digit_history_matched_within_width": True,
            "only_terminal_operand_digits_change_within_width": True,
            "positive_terminal_digits_lsf": ["9", "8"],
            "negative_terminal_digits_lsf": ["2", "3"],
        },
        "cases": [
            {
                "id": "w2_positive",
                "arm": "positive",
                "width": 2,
                "carry_class": "11",
                "expected_state": "dws:op=add;w=2;p=2;c=1;a=69;b=78;r=38;z=1",
                "expected_answer": 183,
            },
            {
                "id": "w2_negative",
                "arm": "negative",
                "width": 2,
                "carry_class": "10",
                "expected_state": "dws:op=add;w=2;p=2;c=0;a=62;b=73;r=36;z=1",
                "expected_answer": 63,
            },
            {
                "id": "w3_positive",
                "arm": "positive",
                "width": 3,
                "carry_class": "11",
                "expected_state": "dws:op=add;w=3;p=3;c=1;a=679;b=748;r=328;z=1",
                "expected_answer": 1823,
            },
            {
                "id": "w3_negative",
                "arm": "negative",
                "width": 3,
                "carry_class": "10",
                "expected_state": "dws:op=add;w=3;p=3;c=0;a=672;b=743;r=326;z=1",
                "expected_answer": 623,
            },
            {
                "id": "w4_positive",
                "arm": "positive",
                "width": 4,
                "carry_class": "11",
                "expected_state": "dws:op=add;w=4;p=4;c=1;a=6799;b=7418;r=3218;z=1",
                "expected_answer": 18123,
            },
            {
                "id": "w4_negative",
                "arm": "negative",
                "width": 4,
                "carry_class": "10",
                "expected_state": "dws:op=add;w=4;p=4;c=0;a=6792;b=7413;r=3216;z=1",
                "expected_answer": 6123,
            },
            {
                "id": "w5_positive",
                "arm": "positive",
                "width": 5,
                "carry_class": "01",
                "expected_state": "dws:op=add;w=5;p=5;c=1;a=67909;b=74128;r=32137;z=1",
                "expected_answer": 173123,
            },
            {
                "id": "w5_negative",
                "arm": "negative",
                "width": 5,
                "carry_class": "00",
                "expected_state": "dws:op=add;w=5;p=5;c=0;a=67902;b=74123;r=32135;z=1",
                "expected_answer": 53123,
            },
            {
                "id": "w6_positive",
                "arm": "positive",
                "width": 6,
                "carry_class": "11",
                "expected_state": "dws:op=add;w=6;p=6;c=1;a=679099;b=741268;r=321358;z=1",
                "expected_answer": 1853123,
            },
            {
                "id": "w6_negative",
                "arm": "negative",
                "width": 6,
                "carry_class": "10",
                "expected_state": "dws:op=add;w=6;p=6;c=0;a=679092;b=741263;r=321356;z=1",
                "expected_answer": 653123,
            },
            {
                "id": "w7_positive",
                "arm": "positive",
                "width": 7,
                "carry_class": "01",
                "expected_state": "dws:op=add;w=7;p=7;c=1;a=6790939;b=7412608;r=3213547;z=1",
                "expected_answer": 17453123,
            },
            {
                "id": "w7_negative",
                "arm": "negative",
                "width": 7,
                "carry_class": "00",
                "expected_state": "dws:op=add;w=7;p=7;c=0;a=6790932;b=7412603;r=3213545;z=1",
                "expected_answer": 5453123,
            },
            {
                "id": "w8_positive",
                "arm": "positive",
                "width": 8,
                "carry_class": "01",
                "expected_state": (
                    "dws:op=add;w=8;p=8;c=1;a=67909319;b=74126068;r=32135477;z=1"
                ),
                "expected_answer": 177453123,
            },
            {
                "id": "w8_negative",
                "arm": "negative",
                "width": 8,
                "carry_class": "00",
                "expected_state": (
                    "dws:op=add;w=8;p=8;c=0;a=67909312;b=74126063;r=32135475;z=1"
                ),
                "expected_answer": 57453123,
            },
            {
                "id": "w9_positive",
                "arm": "positive",
                "width": 9,
                "carry_class": "11",
                "expected_state": (
                    "dws:op=add;w=9;p=9;c=1;a=679093149;b=741260688;r=321354728;z=1"
                ),
                "expected_answer": 1827453123,
            },
            {
                "id": "w9_negative",
                "arm": "negative",
                "width": 9,
                "carry_class": "10",
                "expected_state": (
                    "dws:op=add;w=9;p=9;c=0;a=679093142;b=741260683;r=321354726;z=1"
                ),
                "expected_answer": 627453123,
            },
            {
                "id": "w10_positive",
                "arm": "positive",
                "width": 10,
                "carry_class": "11",
                "expected_state": (
                    "dws:op=add;w=10;p=10;c=1;a=6790931469;b=7412606838;"
                    "r=3213547208;z=1"
                ),
                "expected_answer": 18027453123,
            },
            {
                "id": "w10_negative",
                "arm": "negative",
                "width": 10,
                "carry_class": "10",
                "expected_state": (
                    "dws:op=add;w=10;p=10;c=0;a=6790931462;b=7412606833;"
                    "r=3213547206;z=1"
                ),
                "expected_answer": 6027453123,
            },
        ],
        "frozen_parent_observation": {
            "positive_transition_exact": 0,
            "positive_transition_denominator": 9,
            "positive_serializer_exact": 0,
            "positive_serializer_denominator": 9,
            "positive_raw_carry_exact_widths": [2, 3],
            "positive_raw_carry_failed_widths": [4, 5, 6, 7, 8, 9, 10],
            "positive_failure_classes": {
                "z_only": [2, 3],
                "c_only": [4, 5, 7],
                "c_and_r": [6, 8, 9],
                "broad_p_c_r_z": [10],
            },
            "common_positive_answer_error": "leading-1 omission",
            "negative_raw_carry_exact_widths": [2, 3, 4, 5, 6, 7, 8, 9, 10],
            "negative_serializer_exact_widths": [2, 3, 4, 5, 6],
            "negative_serializer_failed_widths": [7, 8, 9, 10],
            "negative_transition_exact_widths": [2, 3, 4, 5, 7],
            "negative_transition_failed_widths": [6, 8, 9, 10],
        },
        "residual_swap_diagnostic": {
            "artifact": (
                "artifacts/eval_history/"
                "drs_terminal_carry_residual_swap_w2_w10_20260718_mps.json"
            ),
            "artifact_sha256": (
                "4183b8c381e559b23c41b88c8c8cc3b3d0e0b41c03b3dea4786df98a7676590f"
            ),
            "layer": 29,
            "teacher_forced_class_separation_widths": [2, 3, 4, 5, 7, 8, 9, 10],
            "teacher_forced_inverted_widths": [6],
            "ordinary_positive_c1_logit_over_c0_widths": [2, 3],
            "ordinary_positive_c1_logit_not_over_c0_widths": [4, 5, 6, 7, 8, 9, 10],
            "evidence_boundary": "teacher-forced calibration hypothesis only",
            "autonomous_motor_or_reasoning_claim": False,
        },
        "carry_motor_decision": {
            "positive_widths": [2, 3, 4, 5, 6, 7, 8, 9, 10],
            "required_positive_transition_exact": 9,
            "positive_transition_denominator": 9,
            "required_positive_failure_class_exact": {
                "z_only": {"widths": [2, 3], "fields": ["z"], "required": 2},
                "c_only": {"widths": [4, 5, 7], "fields": ["c"], "required": 3},
                "c_and_r": {
                    "widths": [6, 8, 9],
                    "fields": ["c", "r"],
                    "required": 3,
                },
                "broad_p_c_r_z": {
                    "widths": [10],
                    "fields": ["p", "c", "r", "z"],
                    "required": 1,
                },
            },
            "negative_carry_preservation_widths": [2, 3, 4, 5, 6, 7, 8, 9, 10],
            "required_negative_carry_preservation_exact": 9,
            "negative_carry_preservation_denominator": 9,
            "negative_serializer_preservation_widths": [2, 3, 4, 5, 6],
            "required_negative_serializer_preservation_exact": 5,
            "negative_serializer_preservation_denominator": 5,
            "allowed_shuffled_positive_transition_rescues": 0,
            "required_treatment_gain_over_constant_bias_transitions": 2,
            "required_treatment_gain_over_nuisance_only_transitions": 2,
            "requires_no_new_matched_negative_divergence": True,
            "serializer_transfer_metrics_may_satisfy_carry_gate": False,
        },
        "heldout_motor_calibration": {
            "widths": [2, 3, 4, 5, 6, 7, 8, 9, 10],
            "oracle_exclusion_proof": (
                "derive zero source-input and prompt-receipt overlap between the "
                "canonical fit split and every frozen matched oracle"
            ),
            "report_fields_separately": ["p", "c", "r", "z"],
            "report_each_width_separately": True,
            "pooled_field_score_may_satisfy_gate": False,
            "teacher_forced_swap_may_satisfy_gate": False,
            "fit_metric_may_satisfy_gate": False,
            "width_6_inversion_may_be_pooled_away": False,
            "recompute_constant_and_nuisance_on_all_18_terminal_cases": True,
            "one_off_terminal_pairwise_values_may_be_embedded_as_gate": False,
            "full_vocabulary_per_width_evaluation_required": True,
            "public_width_ood_is_unopened": False,
            "confirmation_width_ood_must_remain_unopened_until_reveal": True,
            "requires_development_and_confirmation": True,
        },
        "serializer_decision": {
            "positive_readout_widths": [2, 3, 4, 5, 6, 7, 8, 9, 10],
            "negative_preservation_widths": [2, 3, 4, 5, 6],
            "negative_transfer_readout_widths": [7, 8, 9, 10],
            "report_each_width_separately": True,
            "required_negative_preservation_exact": 5,
            "negative_preservation_denominator": 5,
            "length_transfer_required_positive_exact": 9,
            "length_transfer_positive_denominator": 9,
            "length_transfer_required_negative_exact": 9,
            "length_transfer_negative_denominator": 9,
            "carry_transition_metrics_may_satisfy_serializer_gate": False,
        },
    },
    "preservation": {
        "non_dws_prompts": "frozen upstream non-DWS preservation set",
        "matched_arms": [
            "base",
            "treatment",
            "constant_bias",
            "nuisance_only",
            "shuffled",
        ],
        "allowed_false_fires": 0,
        "required_gate_off_token_and_logit_identity": "exact",
        "allowed_fit_width_regression_points": 2,
    },
    "mechanism_go": {
        "next_carry_accuracy_min": 0.95,
        "next_carry_gain_over_base_points_min": 15,
        "next_carry_gain_over_shuffled_points_min": 15,
        "next_carry_gain_over_constant_bias_points_min": 15,
        "next_carry_gain_over_nuisance_only_points_min": 15,
        "one_step_state_gain_over_base_points_min": 15,
        "one_step_state_gain_over_shuffled_points_min": 15,
        "one_step_state_gain_over_constant_bias_points_min": 15,
        "one_step_state_gain_over_nuisance_only_points_min": 15,
        "autonomous_episode_gain_over_base_points_min": 20,
        "autonomous_episode_gain_over_shuffled_points_min": 20,
        "autonomous_episode_gain_over_constant_bias_points_min": 20,
        "autonomous_episode_gain_over_nuisance_only_points_min": 20,
        "unseen_width_gain_over_base_points_min": 15,
        "unseen_width_gain_over_shuffled_points_min": 15,
        "unseen_width_gain_over_constant_bias_points_min": 15,
        "unseen_width_gain_over_nuisance_only_points_min": 15,
        "boundary_cycle_min_correct": 25,
        "boundary_cycle_denominator": 50,
        "boundary_cycle_gain_over_constant_bias_min_correct": 8,
        "boundary_cycle_gain_over_nuisance_only_min_correct": 8,
        "direct_interactions_min_exact": 8,
        "direct_interactions_denominator": 12,
        "direct_interactions_gain_over_constant_bias_min_exact": 2,
        "direct_interactions_gain_over_nuisance_only_min_exact": 2,
        "shuffled_may_meet_treatment_thresholds": False,
        "constant_bias_may_match_treatment_mechanism_metrics": False,
        "nuisance_only_may_match_treatment_mechanism_metrics": False,
        "treatment_constant_bias_comparisons_are_noncompensatory": True,
        "treatment_nuisance_only_comparisons_are_noncompensatory": True,
        "confirmation_every_transition_stratum_must_beat_both_nulls": True,
        "confirmation_every_episode_stratum_must_beat_both_nulls": True,
        "requires_development_and_confirmation": True,
        "requires_all_preservation_gates": True,
    },
    "decision_labels": {
        "fit_only": "fit-mechanics-only",
        "carry_only": "writer-actuator-repair-only",
        "serializer_only": "serializer-length-transfer-only",
        "rescue_only": "targeted-carry-commit-repair-only",
        "full_pass": "mechanism-go-not-general-reasoning-proof",
    },
}
RECOVERY_PLAN_CLAIM_BOUNDARY = (
    "This recovery plan binds an already sealed upstream feature lineage and one "
    "reviewed mechanical normalization executor. It establishes no fitted motor, "
    "evaluation result, mechanism, capability, or reasoning claim."
)
RECOVERY_FIT_CLAIM_BOUNDARY = (
    "This v11 recovery fit is not a v8 canonical artifact and establishes no reasoning "
    "result. It reuses exact sealed a0c258e features under explicit dual provenance and "
    "adds restricted constant-bias and saturated nuisance-only calibration nulls; heldout "
    "development and separately reviewed confirmation recovery remain required."
)
REVIEW_CLAIM_BOUNDARY = (
    "A valid external signature attests only that the exact recovery source, runtime, "
    "reserved output identity, normalization amendment, and v11 claim boundary are "
    "eligible to publish the frozen fit. It is not a capability or evaluation result."
)

CONSTANT_BIAS_PAYLOAD_KEYS = frozenset(
    {
        "schema",
        "control_id",
        "parameterization",
        "training_rows",
        "training_feature_payload_sha256",
        "label_source",
        "initial_state_sha256",
        "state",
        "state_sha256",
        "fit",
        "raw_margin_diagnostic",
        "claim_boundary",
    }
)
NUISANCE_ONLY_PAYLOAD_KEYS = frozenset(
    {
        "schema",
        "control_id",
        "parameterization",
        "training_rows",
        "training_feature_payload_sha256",
        "training_metadata_receipt",
        "capacity_ledger",
        "label_source",
        "initial_state_sha256",
        "state",
        "state_sha256",
        "fit",
        "width_ood_policy",
        "claim_boundary",
    }
)


class ConstantBiasMotor(torch.nn.Module):
    """One zero-sum carry-logit delta with no feature-dependent input."""

    def __init__(self):
        super().__init__()
        self.delta = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, hidden):
        if type(hidden) is not torch.Tensor or hidden.ndim < 1:
            raise ValueError("constant-bias deployment requires a tensor batch shape")
        pair = REVIEWED_STACK((-self.delta / 2.0, self.delta / 2.0))
        return pair.expand(*hidden.shape[:-1], 2)

    def parameter_count(self):
        return 1


def constant_bias_parameterization():
    return canonical_json_document(
        {
            "control_id": CONSTANT_BIAS_CONTROL_ID,
            "trainable_parameters": 1,
            "state_key": "delta",
            "identifiable_parameter": "delta=b1-b0",
            "deployed_vector": ["-delta/2", "+delta/2"],
            "hidden_value_dependence": False,
            "width_dependence": False,
            "prompt_dependence": False,
            "token_history_dependence": False,
            "batch_shape_and_device_only_from_hidden": True,
            "deployment_site": "upstream.apply_motor_logits",
            "grammar_gate": "upstream.CarryRouter",
            "gate_off_full_logit_identity": "exact",
        }
    )


class NuisanceOnlyMotor(torch.nn.Module):
    """Saturated fit-cell threshold with frozen width extrapolation."""

    def __init__(self):
        super().__init__()
        self.cell_delta = torch.nn.Parameter(
            torch.zeros(NUISANCE_PARAMETER_COUNT, dtype=torch.float32)
        )

    def forward(self, metadata):
        if (
            type(metadata) is not torch.Tensor
            or metadata.ndim != 2
            or metadata.shape[1] != NUISANCE_PARAMETER_COUNT
        ):
            raise ValueError("nuisance-only deployment requires frozen design rows")
        delta = metadata @ self.cell_delta
        return REVIEWED_STACK((-delta / 2.0, delta / 2.0), dim=-1)

    def parameter_count(self):
        return NUISANCE_PARAMETER_COUNT


def _nuisance_position_weights(width, position, fit_width):
    fraction = float(position) / float(width - 1)
    coordinate = fraction * float(fit_width - 1)
    lower = min(int(math.floor(coordinate)), fit_width - 1)
    upper = min(lower + 1, fit_width - 1)
    upper_weight = coordinate - float(lower)
    lower_weight = 1.0 - upper_weight
    return ((lower, lower_weight), (upper, upper_weight))


def _nuisance_design_row(operation, width, position):
    if (
        operation not in {"add", "sub"}
        or type(width) is not int
        or width < 2
        or type(position) is not int
        or not 0 <= position < width
    ):
        raise ValueError("nuisance metadata row is malformed")
    width_coordinate = (float(width) - 4.0) / 2.0
    fit_width_weights = ((4, 1.0 - width_coordinate), (6, width_coordinate))
    cell_to_index = {cell: index for index, cell in enumerate(NUISANCE_FIT_CELLS)}
    design = [0.0] * NUISANCE_PARAMETER_COUNT
    for fit_width, width_weight in fit_width_weights:
        for fit_position, position_weight in _nuisance_position_weights(
            width, position, fit_width
        ):
            design[cell_to_index[(operation, fit_width, fit_position)]] += (
                width_weight * position_weight
            )
    return design


def nuisance_metadata_from_rows(rows):
    """Build the only deployment inputs allowed for the nuisance-only arm."""
    if type(rows) is not list or not rows:
        raise ValueError("nuisance metadata requires a nonempty row list")
    values = []
    for row in rows:
        if type(row) is not dict:
            raise ValueError("nuisance metadata row must be a dictionary")
        values.append(
            _nuisance_design_row(
                row.get("operation"), row.get("width"), row.get("position")
            )
        )
    return REVIEWED_AS_TENSOR(values, dtype=torch.float32)


def nuisance_metadata_receipt(rows):
    metadata = nuisance_metadata_from_rows(rows)
    identities = [
        {
            "operation": row["operation"],
            "width": row["width"],
            "position": row["position"],
        }
        for row in rows
    ]
    return canonical_json_document(
        {
            "schema": NUISANCE_METADATA_SCHEMA,
            "source_fields": ["operation", "width", "position"],
            "forbidden_source_fields": [
                "hidden",
                "prompt",
                "prompt_ids",
                "prefix_ids",
                "style",
                "current_carry",
                "target",
                "operand_digits",
            ],
            "feature_names": list(NUISANCE_METADATA_FEATURE_NAMES),
            "fit_cells": [list(cell) for cell in NUISANCE_FIT_CELLS],
            "rows": len(rows),
            "row_identity_sha256": stable_json_sha256(identities),
            "metadata_sha256": scientific_tree_sha256(metadata),
        }
    )


def nuisance_only_parameterization():
    return canonical_json_document(
        {
            "control_id": NUISANCE_ONLY_CONTROL_ID,
            "trainable_parameters": NUISANCE_PARAMETER_COUNT,
            "state_keys": ["cell_delta"],
            "identifiable_parameter": (
                "one independent delta per operation/fit-width/position cell"
            ),
            "deployed_vector": ["-delta(metadata)/2", "+delta(metadata)/2"],
            "allowed_metadata": ["operation", "width", "position"],
            "feature_names": list(NUISANCE_METADATA_FEATURE_NAMES),
            "fit_cells": [list(cell) for cell in NUISANCE_FIT_CELLS],
            "fit_widths": list(NUISANCE_TRAIN_WIDTHS),
            "fit_design_rank": NUISANCE_PARAMETER_COUNT,
            "width_extrapolation": (
                "piecewise-linear normalized-position interpolation within widths "
                "4 and 6, followed by the immutable linear width rule "
                "d(w)=d4+(w-4)*(d6-d4)/2"
            ),
            "hidden_residual_dependence": False,
            "prompt_text_dependence": False,
            "token_history_dependence": False,
            "style_dependence": False,
            "current_carry_dependence": False,
            "operand_digit_dependence": False,
            "deployment_site": "upstream.apply_motor_logits",
            "grammar_gate": "upstream.CarryRouter",
            "gate_off_full_logit_identity": "exact",
        }
    )


def nuisance_capacity_ledger():
    return canonical_json_document(NUISANCE_CAPACITY_LEDGER)


def nuisance_width_ood_policy():
    return canonical_json_document(
        {
            "fit_widths": list(NUISANCE_TRAIN_WIDTHS),
            "public_diagnostic_widths": list(NUISANCE_PUBLIC_OOD_WIDTHS),
            "confirmation_ood_width_rule": "every width outside fit widths 4 and 6",
            "fit_may_read_public_or_confirmation_ood_rows": False,
            "refit_after_public_or_confirmation_ood": False,
            "extrapolation": (
                "evaluate the frozen saturated fit-cell state through the immutable "
                "normalized-position and linear-width rule"
            ),
            "public_width_ood_is_unopened": False,
            "confirmation_width_ood_must_remain_unopened_until_reveal": True,
            "model_family_frozen_before_width8": True,
            "checkpoint_frozen_before_width8": True,
            "selection_widths": list(NUISANCE_TRAIN_WIDTHS),
            "selection_may_read_width8": False,
            "selection_may_read_confirmation": False,
            "capacity_ledger": nuisance_capacity_ledger(),
        }
    )


def deployment_parameter_ledger(checkpoint_cfg, rank):
    """Derive every unique deployed parameter from the bound model architecture."""
    if type(checkpoint_cfg) is not dict or type(rank) is not int or rank <= 0:
        raise ValueError("deployment parameter ledger inputs are malformed")
    observed = {name: checkpoint_cfg.get(name) for name in EXPECTED_BASE_PARAMETER_CONFIG}
    if any(type(observed[name]) is not type(expected) for name, expected in EXPECTED_BASE_PARAMETER_CONFIG.items()):
        raise ValueError("deployment parameter configuration is not type-strict")
    if observed != EXPECTED_BASE_PARAMETER_CONFIG:
        raise ValueError("deployment parameter configuration differs from the frozen base")

    vocab_size = observed["vocab_size"]
    layers = observed["n_layer"]
    heads = observed["n_head"]
    kv_heads = observed["n_kv_head"]
    d_model = observed["d_model"]
    d_ff = observed["d_ff"]
    if d_model % heads != 0 or heads % kv_heads != 0:
        raise ValueError("deployment attention dimensions are not integral")
    head_dim = d_model // heads
    embedding_and_tied_head = vocab_size * d_model
    attention_per_layer = (
        d_model * d_model
        + 2 * d_model * kv_heads * head_dim
        + d_model * d_model
    )
    qk_norm_per_layer = 2 * head_dim if observed["qk_norm"] else 0
    block_norms_per_layer = 2 * d_model
    mlp_per_layer = 3 * d_model * d_ff
    transformer_blocks = layers * (
        attention_per_layer
        + qk_norm_per_layer
        + block_norms_per_layer
        + mlp_per_layer
    )
    final_norm = d_model
    untied_output_head = 0 if observed["tie_embeddings"] else vocab_size * d_model
    base_total = (
        embedding_and_tied_head
        + transformer_blocks
        + final_norm
        + untied_output_head
    )
    treatment = d_model * rank + rank + 2 * rank + 2
    constant_bias = 1
    nuisance_only = NUISANCE_PARAMETER_COUNT
    combined_total = base_total + treatment + constant_bias + nuisance_only
    remaining = STRICT_PARAMETER_CAP - combined_total
    if (
        base_total != EXPECTED_BASE_PARAMETER_COUNT
        or treatment != d_model * rank + 3 * rank + 2
        or combined_total >= STRICT_PARAMETER_CAP
        or remaining <= 0
    ):
        raise ValueError("deployment parameter ledger violates the strict cap")
    return canonical_json_document(
        {
            "schema": "carry_motor_exact_deployment_parameter_ledger_v1",
            "strict_cap_exclusive": STRICT_PARAMETER_CAP,
            "base_configuration": observed,
            "base": {
                "embedding_and_tied_output_head": embedding_and_tied_head,
                "transformer_blocks": transformer_blocks,
                "final_norm": final_norm,
                "untied_output_head": untied_output_head,
                "unique_trainable_parameters": base_total,
                "tied_embedding_counted_once": True,
            },
            "treatment": {
                "rank": rank,
                "formula": "rank*d_model + 3*rank + 2",
                "trainable_parameters": treatment,
            },
            "null_heads": {
                "constant_bias": constant_bias,
                "nuisance_only": nuisance_only,
                "trainable_parameters": constant_bias + nuisance_only,
            },
            "combined_unique_trainable_parameters": combined_total,
            "strictly_below_cap": True,
            "remaining_headroom": remaining,
        }
    )


def validate_deployment_parameter_ledger(ledger, checkpoint_cfg, rank):
    expected = deployment_parameter_ledger(checkpoint_cfg, rank)
    if (
        type(ledger) is not dict
        or not type_strict_equal(ledger, expected)
        or type(ledger.get("combined_unique_trainable_parameters")) is not int
        or ledger["combined_unique_trainable_parameters"] >= STRICT_PARAMETER_CAP
        or ledger.get("strictly_below_cap") is not True
    ):
        raise ValueError("deployment parameter ledger is inexact or exceeds the cap")
    return expected


REVIEW_TRUST_BOUNDARY = (
    "Signature verification proves control of the frozen external private key and exact "
    "receipt bytes. It cannot prove reviewer independence, diligence, or honesty; those "
    "remain explicit human-governance assumptions."
)

FEATURE_READING_EVALUATION_SCHEMA = "carry_motor_canonical_case_evaluation_v4"
FEATURE_READING_CASE_SCHEMA = "carry_motor_canonical_case_v2"
FEATURE_READING_RECORD_SCHEMA = "carry_motor_canonical_case_result_v2"
FEATURE_READING_DECISION_SCHEMA = "carry_motor_feature_reading_decision_v4"


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_sha256(value):
    return sha256_bytes(canonical_json_payload(value).encode("ascii"))


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _parse_finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON float: {value}")
    return parsed


def _validate_finite_json_tree(
    value,
    label,
    *,
    allow_integer_mapping_keys=False,
    _active=None,
):
    """Require an acyclic tree of exact finite JSON primitive types."""
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite float")
        return
    if type(value) not in {list, dict}:
        raise ValueError(f"{label} contains a non-JSON type: {type(value).__name__}")
    active = set() if _active is None else _active
    identity = id(value)
    if identity in active:
        raise ValueError(f"{label} contains a recursive JSON container")
    active.add(identity)
    try:
        if type(value) is list:
            for item in value:
                _validate_finite_json_tree(
                    item,
                    label,
                    allow_integer_mapping_keys=allow_integer_mapping_keys,
                    _active=active,
                )
            return
        for key, item in value.items():
            if type(key) is not str and not (
                allow_integer_mapping_keys and type(key) is int
            ):
                raise ValueError(f"{label} contains a non-string JSON key")
            _validate_finite_json_tree(
                item,
                label,
                allow_integer_mapping_keys=allow_integer_mapping_keys,
                _active=active,
            )
    finally:
        active.remove(identity)


def enforce_secure_creation_umask():
    """Set and verify the process-wide creation mask used by this executor."""
    os.umask(SECURE_CREATION_UMASK)
    observed = os.umask(SECURE_CREATION_UMASK)
    if observed != SECURE_CREATION_UMASK:
        raise RuntimeError("recovery executor could not enforce umask 0077")
    return "0077"


def require_secure_creation_umask():
    observed = os.umask(SECURE_CREATION_UMASK)
    os.umask(observed)
    if observed != SECURE_CREATION_UMASK:
        raise RuntimeError("recovery executor creation umask is not 0077")
    return "0077"


def load_exact_json(text, label):
    try:
        value = json.loads(
            text,
            object_pairs_hook=_json_pairs,
            parse_float=_parse_finite_float,
            parse_constant=_reject_constant,
        )
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    _validate_finite_json_tree(value, label)
    return value


def canonical_json_payload(value):
    _validate_finite_json_tree(value, "canonical JSON value")
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "value is not finite or contains non-finite strict JSON"
        ) from exc


def canonical_json_document(value):
    return load_exact_json(canonical_json_payload(value), "canonical JSON document")


def canonical_json_receipt_bytes(value):
    return (canonical_json_payload(value) + "\n").encode("ascii")


def decode_canonical_base64(value, expected_bytes, label):
    if type(value) is not str or not value or not value.isascii():
        raise ValueError(f"{label} is not canonical base64")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} is not canonical base64") from exc
    canonical = base64.b64encode(decoded).decode("ascii")
    if canonical != value or len(decoded) != int(expected_bytes):
        raise ValueError(f"{label} has non-canonical padding or length")
    return decoded


def normalize_generated_board_document(value):
    """Apply the one preregistered integer-key-to-JSON-string normalization."""
    _validate_finite_json_tree(
        value,
        "generated fit board",
        allow_integer_mapping_keys=True,
    )
    try:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "generated fit board is not finite JSON-compatible data"
        ) from exc
    return load_exact_json(payload, "normalized generated fit board")


def type_strict_equal(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is torch.Tensor:
        return (
            left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and REVIEWED_TORCH_EQUAL(left, right)
        )
    if type(left) in {dict, collections.OrderedDict}:
        if len(left) != len(right):
            return False
        unmatched = list(right)
        pairs = []
        for left_key in left:
            matches = [
                right_key
                for right_key in unmatched
                if type(left_key) is type(right_key) and left_key == right_key
            ]
            if len(matches) != 1:
                return False
            right_key = matches[0]
            unmatched.remove(right_key)
            pairs.append((left_key, right_key))
        return not unmatched and all(
            type_strict_equal(left[left_key], right[right_key])
            for left_key, right_key in pairs
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            type_strict_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def initial_constant_bias_state():
    motor = ConstantBiasMotor()
    state = collections.OrderedDict(
        (name, tensor.detach().cpu().clone())
        for name, tensor in motor.state_dict().items()
    )
    return state, REVIEWED_TENSOR_STATE_SHA256(state)


def initial_nuisance_only_state():
    motor = NuisanceOnlyMotor()
    state = collections.OrderedDict(
        (name, tensor.detach().cpu().clone())
        for name, tensor in motor.state_dict().items()
    )
    return state, REVIEWED_TENSOR_STATE_SHA256(state)


def raw_carry_margin_diagnostic(base01, labels, fitted_delta):
    """Describe the strongest hidden-independent binary threshold explanation."""
    if (
        type(base01) is not torch.Tensor
        or type(labels) is not torch.Tensor
        or base01.ndim != 2
        or base01.shape[1] != 2
        or labels.ndim != 1
        or len(base01) != len(labels)
        or type(fitted_delta) is not float
        or not math.isfinite(fitted_delta)
    ):
        raise ValueError("raw carry margin diagnostic inputs are malformed")
    margins_tensor = (base01[:, 1].float() - base01[:, 0].float()).detach().cpu()
    labels_tensor = labels.detach().cpu()
    if not bool(REVIEWED_ISFINITE(margins_tensor).all()) or not bool(
        ((labels_tensor == 0) | (labels_tensor == 1)).all()
    ):
        raise ValueError("raw carry margins or labels are invalid")
    margins = [float(value) for value in margins_tensor]
    target_values = [int(value) for value in labels_tensor]
    positive = [margin for margin, target in zip(margins, target_values) if target == 1]
    negative = [margin for margin, target in zip(margins, target_values) if target == 0]
    if not positive or not negative:
        raise ValueError("raw carry margin diagnostic requires both classes")

    lower = -min(positive)
    upper = -max(negative)
    breakpoints = sorted(set(-margin for margin in margins))
    candidates = {0.0}
    candidates.add(math.nextafter(breakpoints[0], -math.inf))
    candidates.add(math.nextafter(breakpoints[-1], math.inf))
    for left, right in zip(breakpoints, breakpoints[1:]):
        midpoint = left + (right - left) / 2.0
        if left < midpoint < right:
            candidates.add(midpoint)
        else:
            candidates.add(math.nextafter(left, right))

    scored = []
    for delta in candidates:
        correct = sum(
            int((1 if margin + delta > 0.0 else 0) == target)
            for margin, target in zip(margins, target_values)
        )
        scored.append((correct, float(delta)))
    best_correct, best_delta = min(
        scored,
        key=lambda item: (-item[0], abs(item[1]), item[1]),
    )
    return canonical_json_document(
        {
            "schema": "carry_motor_raw_margin_threshold_v1",
            "margin_definition": "m=logit(c1)-logit(c0)",
            "constant_shift": "m_after=m+delta",
            "feasibility_condition": ("-min(m_positive) < delta < -max(m_negative)"),
            "positive_rows": len(positive),
            "negative_rows": len(negative),
            "min_positive_margin": min(positive),
            "max_negative_margin": max(negative),
            "feasible_delta_interval": {
                "lower_exclusive": lower,
                "upper_exclusive": upper,
                "nonempty": lower < upper,
            },
            "binary_accuracy_optimum": {
                "delta": best_delta,
                "correct": best_correct,
                "denominator": len(margins),
                "accuracy": best_correct / len(margins),
                "tie_break": "max_correct_then_min_abs_delta_then_lowest_delta",
            },
            "fitted_delta": fitted_delta,
            "fitted_delta_in_feasible_interval": lower < fitted_delta < upper,
            "claim_boundary": CONSTANT_BIAS_CLAIM_BOUNDARY,
        }
    )


def _feature_case_expected_keys():
    return {
        "schema",
        "case_id",
        "phase",
        "suite",
        "regime",
        "operation",
        "width",
        "position",
        "target",
        "expected_token_ids",
        "expected_transition",
        "expected_serializer_token_ids",
        "expected_episode_output",
        "source_input",
        "source_input_sha256",
        "prompt",
        "prompt_sha256",
        "generator_binding",
        "generator_receipt_sha256",
        "split_membership",
        "split_membership_receipt_sha256",
    }


def _feature_record_expected_keys():
    return {
        "schema",
        "case_id",
        "arm",
        "actual_token_ids",
        "full_vocab_logits",
        "actual_transition",
        "actual_serializer_token_ids",
        "actual_episode_output",
        "motor_gate_trace",
        "gate_off_base_logits",
        "gate_off_arm_logits",
        "deployment_vocabulary_sha256",
    }


def _validate_deployment_vocabulary(binding):
    required = {
        "schema",
        "base_checkpoint_sha256",
        "tokenizer_sha256",
        "output_vocab_width",
        "token_id_order",
        "token_id_order_sha256",
        "tokenizer_id_to_token_sha256",
        "zero_id",
        "one_id",
        "deployment_logit_dtype",
    }
    if type(binding) is not dict or set(binding) != required:
        raise ValueError("deployment vocabulary binding schema mismatch")
    width = binding["output_vocab_width"]
    token_order = binding["token_id_order"]
    if (
        binding["schema"] != DEPLOYMENT_VOCABULARY_SCHEMA
        or binding["base_checkpoint_sha256"] != upstream.EXPECTED_CHECKPOINT_SHA256
        or binding["tokenizer_sha256"] != upstream.EXPECTED_TOKENIZER_SHA256
        or type(width) is not int
        or width < 2
        or type(token_order) is not list
        or any(type(token_id) is not int for token_id in token_order)
        or token_order != list(range(width))
        or binding["token_id_order_sha256"] != stable_json_sha256(token_order)
        or not re.fullmatch(
            r"[0-9a-f]{64}", binding["tokenizer_id_to_token_sha256"]
        )
        or type(binding["zero_id"]) is not int
        or type(binding["one_id"]) is not int
        or binding["zero_id"] == binding["one_id"]
        or not 0 <= binding["zero_id"] < width
        or not 0 <= binding["one_id"] < width
        or binding["deployment_logit_dtype"] != "torch.bfloat16"
    ):
        raise ValueError("deployment vocabulary is not model-bound and ordered")
    return canonical_json_document(binding)


def _deployment_other_token_ids(deployment_vocabulary):
    return REVIEWED_TENSOR(
        [
            token_id
            for token_id in range(deployment_vocabulary["output_vocab_width"])
            if token_id
            not in {
                deployment_vocabulary["zero_id"],
                deployment_vocabulary["one_id"],
            }
        ],
        dtype=torch.int64,
    )


def _expected_case_split(case):
    if case["phase"] == "matched":
        return "matched"
    if case["phase"] == "confirmation":
        return "confirmation"
    match = re.fullmatch(r"development-episode-([0-9]{3})", case["case_id"])
    if case["suite"] == "episode" and match is not None and int(match.group(1)) < 40:
        return "fit_width_audit"
    return "development"


def _expected_case_generator_id(phase):
    return {
        "matched": "frozen_matched_generator_v1",
        "development": "frozen_development_generator_v1",
        "confirmation": "a0c258e_confirmation_generator_v1",
    }[phase]


def _validate_case_provenance(
    item,
    *,
    expected_split,
    expected_generator_id,
    required_source_fields,
):
    source_input = item.get("source_input")
    prompt = item.get("prompt")
    generator = item.get("generator_binding")
    if type(source_input) is not dict or not source_input:
        raise ValueError("canonical source input is missing")
    _validate_finite_json_tree(source_input, "canonical source input")
    if (
        type(required_source_fields) is not tuple
        or not required_source_fields
        or any(type(name) is not str for name in required_source_fields)
        or any(name not in source_input or name not in item for name in required_source_fields)
    ):
        raise ValueError("canonical source input is missing required identity fields")
    for name in required_source_fields:
        if not type_strict_equal(source_input[name], item[name]):
            raise ValueError("canonical case differs from its bound source input")
    if type(prompt) is not str or not prompt:
        raise ValueError("canonical prompt is missing")
    if (
        type(generator) is not dict
        or set(generator)
        != {"schema", "generator_id", "source_contract_sha256"}
        or generator["schema"] != CASE_GENERATOR_BINDING_SCHEMA
        or generator["generator_id"] != expected_generator_id
        or not re.fullmatch(r"[0-9a-f]{64}", generator["source_contract_sha256"])
    ):
        raise ValueError("canonical prompt generator binding is malformed")
    source_sha256 = stable_json_sha256(source_input)
    prompt_sha256 = sha256_bytes(prompt.encode("utf-8"))
    generator_receipt = stable_json_sha256(
        {
            "generator_binding": generator,
            "source_input_sha256": source_sha256,
            "prompt_sha256": prompt_sha256,
        }
    )
    membership = {
        "case_id": item.get("case_id"),
        "split": expected_split,
        "source_input_sha256": source_sha256,
        "prompt_sha256": prompt_sha256,
        "generator_receipt_sha256": generator_receipt,
    }
    if (
        item.get("source_input_sha256") != source_sha256
        or item.get("prompt_sha256") != prompt_sha256
        or item.get("generator_receipt_sha256") != generator_receipt
        or item.get("split_membership") != expected_split
        or item.get("split_membership_receipt_sha256")
        != stable_json_sha256(membership)
    ):
        raise ValueError("canonical source or split receipt is forged")
    return canonical_json_document(membership)


def canonical_fit_selection_rows(context):
    rows = context["board_context"]["rows"]
    labels = context["features"]["labels"]
    if type(rows) is not list or len(rows) != len(labels):
        raise ValueError("canonical fit rows and labels differ in length")
    generator = {
        "schema": CASE_GENERATOR_BINDING_SCHEMA,
        "generator_id": "a0c258e_fit_generator_v1",
        "source_contract_sha256": stable_json_sha256(
            context["upstream_source_contract"]
        ),
    }
    result = []
    for index, (row, label) in enumerate(zip(rows, labels)):
        if type(row) is not dict:
            raise ValueError("canonical fit source row is malformed")
        _validate_finite_json_tree(row, "canonical fit source row")
        source_input = copy.deepcopy(row)
        source_sha256 = stable_json_sha256(source_input)
        prompt_identity = {
            "prompt_ids": row.get("prompt_ids"),
            "prompt_sha256": row.get("prompt_sha256"),
            "prefix_ids": row.get("prefix_ids"),
            "prefix_sha256": row.get("prefix_sha256"),
        }
        _validate_finite_json_tree(prompt_identity, "canonical fit prompt identity")
        prompt = canonical_json_payload(prompt_identity)
        prompt_sha256 = sha256_bytes(prompt.encode("ascii"))
        generator_receipt = stable_json_sha256(
            {
                "generator_binding": generator,
                "source_input_sha256": source_sha256,
                "prompt_sha256": prompt_sha256,
            }
        )
        case_id = f"fit-row-{index:06d}-{source_sha256[:16]}"
        membership = {
            "case_id": case_id,
            "split": "fit",
            "source_input_sha256": source_sha256,
            "prompt_sha256": prompt_sha256,
            "generator_receipt_sha256": generator_receipt,
        }
        result.append(
            {
                "case_id": case_id,
                "operation": row.get("operation"),
                "width": row.get("width"),
                "position": row.get("position"),
                "target": int(label),
                "source_input": source_input,
                "source_input_sha256": source_sha256,
                "prompt": prompt,
                "prompt_sha256": prompt_sha256,
                "generator_binding": generator,
                "generator_receipt_sha256": generator_receipt,
                "split_membership": "fit",
                "split_membership_receipt_sha256": stable_json_sha256(membership),
            }
        )
    return canonical_json_document(result)


def _validate_fit_selection_rows(rows):
    required = {
        "case_id",
        "operation",
        "width",
        "position",
        "target",
        "source_input",
        "source_input_sha256",
        "prompt",
        "prompt_sha256",
        "generator_binding",
        "generator_receipt_sha256",
        "split_membership",
        "split_membership_receipt_sha256",
    }
    if type(rows) is not list or not rows:
        raise ValueError("model-bound fit rows are missing")
    identifiers = set()
    memberships = []
    for row in rows:
        if type(row) is not dict or set(row) != required:
            raise ValueError("model-bound fit row schema mismatch")
        if (
            type(row["case_id"]) is not str
            or not row["case_id"]
            or row["case_id"] in identifiers
            or row["operation"] not in {"add", "sub"}
            or type(row["width"]) is not int
            or row["width"] not in NUISANCE_TRAIN_WIDTHS
            or type(row["position"]) is not int
            or not 0 <= row["position"] < row["width"]
            or row["target"] not in {0, 1}
        ):
            raise ValueError("model-bound fit row identity is malformed")
        identifiers.add(row["case_id"])
        memberships.append(
            _validate_case_provenance(
                row,
                expected_split="fit",
                expected_generator_id="a0c258e_fit_generator_v1",
                required_source_fields=("operation", "width", "position", "target"),
            )
        )
    return memberships


def _validate_token_ids(value, label, *, nonempty):
    if (
        type(value) is not list
        or (nonempty and not value)
        or any(type(token) is not int or token < 0 for token in value)
    ):
        raise ValueError(f"{label} token ids are malformed")


def _validate_transition(value, label, *, required):
    if value is None and not required:
        return
    if (
        type(value) is not dict
        or set(value) != {"p", "c", "r", "z"}
        or any(type(value[name]) is not int for name in ("p", "c", "r", "z"))
    ):
        raise ValueError(f"{label} transition is malformed")


def _validate_full_vocab_logits(value, label, *, expected_width=None):
    if type(value) is not list or not value:
        raise ValueError(f"{label} full-vocabulary logits are missing")
    width = None
    for row in value:
        if (
            type(row) is not list
            or not row
            or any(type(item) is not float or not math.isfinite(item) for item in row)
        ):
            raise ValueError(f"{label} full-vocabulary logits are malformed")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(f"{label} full-vocabulary logit widths differ")
    if expected_width is not None and width != expected_width:
        raise ValueError(f"{label} logits differ from deployed vocabulary width")


def _tokens_from_full_vocab_logits(value):
    return [
        min(range(len(row)), key=lambda token_id: (-row[token_id], token_id))
        for row in value
    ]


def _terminal_transition_from_state(value):
    match = re.fullmatch(
        r"dws:op=(?:add|sub);w=\d+;p=(\d+);c=(\d+);a=\d+;b=\d+;"
        r"r=(\d+);z=(\d+)",
        value,
    )
    if match is None:
        raise AssertionError("frozen terminal state is malformed")
    position, carry, result, done = (int(item) for item in match.groups())
    return {"p": position, "c": carry, "r": result, "z": done}


def _validate_frozen_matched_manifest(cases):
    rescue_expected = {
        f"rescue:{item['id']}:{item['branch']}": item
        for item in DOWNSTREAM_EVALUATION_CONTRACT["carry_commit_rescue_set"]["cases"]
    }
    terminal_expected = {
        f"terminal:{item['id']}": item
        for item in DOWNSTREAM_EVALUATION_CONTRACT["terminal_width_sweep_oracle"][
            "cases"
        ]
    }
    rescue_observed = {
        case["case_id"]: case
        for case in cases
        if case["phase"] == "matched" and case["suite"] == "rescue"
    }
    terminal_observed = {
        case["case_id"]: case
        for case in cases
        if case["phase"] == "matched" and case["suite"] == "terminal"
    }
    if set(rescue_observed) != set(rescue_expected):
        raise ValueError("canonical manifest has a missing or extra rescue case")
    if set(terminal_observed) != set(terminal_expected):
        raise ValueError("canonical manifest has a missing or extra terminal case")
    for case_id, expected in rescue_expected.items():
        case = rescue_observed[case_id]
        if (
            case["regime"] != expected["branch"]
            or case["operation"] != expected["operation"]
            or case["expected_episode_output"] != expected["expected_answer"]
        ):
            raise ValueError("canonical rescue case differs from frozen oracle")
    for case_id, expected in terminal_expected.items():
        case = terminal_observed[case_id]
        target = 1 if expected["arm"] == "positive" else 0
        if (
            case["regime"] != expected["arm"]
            or case["operation"] != "add"
            or case["width"] != expected["width"]
            or case["position"] != expected["width"] - 1
            or case["target"] != target
            or case["expected_transition"]
            != _terminal_transition_from_state(expected["expected_state"])
            or case["expected_episode_output"] != expected["expected_answer"]
        ):
            raise ValueError("canonical terminal case differs from frozen oracle")


def _validate_fit_model_binding(
    binding,
    deployment_vocabulary,
    expected_fit_model_binding_sha256,
    fit_rows,
    raw_feature_payload,
):
    required = {
        "schema",
        "base_checkpoint_sha256",
        "tokenizer_sha256",
        "upstream_plan_sha256",
        "upstream_source_contract_sha256",
        "upstream_shard_receipts_sha256",
        "feature_merge_sha256",
        "fit_rows_sha256",
        "raw_feature_payload_sha256",
        "deployment_vocabulary_sha256",
        "deployment_logit_dtype",
        "parameter_ledger",
    }
    if type(binding) is not dict or set(binding) != required:
        raise ValueError("fit model binding schema mismatch")
    expected_ledger = deployment_parameter_ledger(
        dict(EXPECTED_BASE_PARAMETER_CONFIG), upstream.RANK
    )
    if (
        binding["schema"] != FIT_MODEL_BINDING_SCHEMA
        or binding["base_checkpoint_sha256"]
        != upstream.EXPECTED_CHECKPOINT_SHA256
        or binding["tokenizer_sha256"] != upstream.EXPECTED_TOKENIZER_SHA256
        or binding["upstream_plan_sha256"] != UPSTREAM_PLAN_SHA256
        or any(
            not re.fullmatch(r"[0-9a-f]{64}", binding[name])
            for name in (
                "upstream_source_contract_sha256",
                "upstream_shard_receipts_sha256",
                "feature_merge_sha256",
                "raw_feature_payload_sha256",
            )
        )
        or binding["fit_rows_sha256"] != stable_json_sha256(fit_rows)
        or binding["raw_feature_payload_sha256"]
        != scientific_tree_sha256(raw_feature_payload)
        or binding["deployment_vocabulary_sha256"]
        != stable_json_sha256(deployment_vocabulary)
        or binding["deployment_logit_dtype"] != "torch.bfloat16"
        or not type_strict_equal(binding["parameter_ledger"], expected_ledger)
        or stable_json_sha256(binding) != expected_fit_model_binding_sha256
    ):
        raise ValueError("fit model binding differs from sealed raw evidence")


def _validate_fit_selection_evidence(
    evidence,
    raw_evidence,
    deployment_vocabulary,
    expected_fit_model_binding_sha256,
):
    required_raw = {
        "schema",
        "fit_model_binding",
        "fit_rows",
        "base01",
        "other_lse",
        "labels",
        "other_token_ids",
        "deployed_state",
    }
    if type(raw_evidence) is not dict or set(raw_evidence) != required_raw:
        raise ValueError("raw fit selection evidence schema mismatch")
    fit_rows = raw_evidence["fit_rows"]
    _validate_fit_selection_rows(fit_rows)
    base01 = raw_evidence["base01"]
    other_lse = raw_evidence["other_lse"]
    labels = raw_evidence["labels"]
    other_token_ids = raw_evidence["other_token_ids"]
    deployed_state = raw_evidence["deployed_state"]
    expected_other_ids = _deployment_other_token_ids(deployment_vocabulary)
    if (
        raw_evidence["schema"] != FIT_SELECTION_RAW_EVIDENCE_SCHEMA
        or type(base01) is not torch.Tensor
        or base01.dtype != torch.bfloat16
        or tuple(base01.shape) != (len(fit_rows), 2)
        or type(other_lse) is not torch.Tensor
        or other_lse.dtype != torch.float32
        or tuple(other_lse.shape) != (len(fit_rows),)
        or type(labels) is not torch.Tensor
        or labels.dtype != torch.int64
        or tuple(labels.shape) != (len(fit_rows),)
        or type(other_token_ids) is not torch.Tensor
        or other_token_ids.dtype != torch.int64
        or tuple(other_token_ids.shape)
        != (deployment_vocabulary["output_vocab_width"] - 2,)
        or not REVIEWED_TORCH_EQUAL(other_token_ids.cpu(), expected_other_ids)
        or not bool(REVIEWED_ISFINITE(base01.float()).all())
        or not bool(REVIEWED_ISFINITE(other_lse).all())
        or [int(value) for value in labels] != [row["target"] for row in fit_rows]
        or type(deployed_state) is not collections.OrderedDict
        or tuple(deployed_state) != ("cell_delta",)
        or type(deployed_state["cell_delta"]) is not torch.Tensor
        or deployed_state["cell_delta"].dtype != torch.float32
        or tuple(deployed_state["cell_delta"].shape)
        != (NUISANCE_PARAMETER_COUNT,)
        or not bool(REVIEWED_ISFINITE(deployed_state["cell_delta"]).all())
    ):
        raise ValueError("raw fit evidence is not deployed bfloat16 model output")
    raw_feature_payload = {
        "base01": base01,
        "other_lse": other_lse,
        "labels": labels,
        "other_token_ids": other_token_ids,
    }
    _validate_fit_model_binding(
        raw_evidence["fit_model_binding"],
        deployment_vocabulary,
        expected_fit_model_binding_sha256,
        fit_rows,
        raw_feature_payload,
    )
    groups = []
    for operation, width, position in NUISANCE_FIT_CELLS:
        indices = [
            index
            for index, row in enumerate(fit_rows)
            if (row["operation"], row["width"], row["position"])
            == (operation, width, position)
        ]
        groups.append((f"{operation}-w{width}-p{position}", indices))
    selected, recomputed = _solve_full_board_scalar_groups(
        {"base01": base01, "other_lse": other_lse},
        labels,
        groups,
        "nuisance_only",
        fit_rows,
        raw_evidence["fit_model_binding"],
    )
    selected_state = collections.OrderedDict(
        (("cell_delta", REVIEWED_TENSOR(selected, dtype=torch.float32)),)
    )
    expected = _finalize_null_optimization_evidence(recomputed, selected_state)
    if not type_strict_equal(deployed_state, selected_state):
        raise ValueError("fit selection did not deploy the recomputed checkpoint")
    if not type_strict_equal(evidence, expected):
        raise ValueError("fit selection evidence differs from raw-evidence replay")
    return expected


def _validate_canonical_feature_manifest(cases):
    if type(cases) is not list or not cases:
        raise ValueError("canonical case manifest is missing")
    identifiers = set()
    for case in cases:
        if type(case) is not dict or set(case) != _feature_case_expected_keys():
            raise ValueError("canonical case schema mismatch")
        if (
            case["schema"] != FEATURE_READING_CASE_SCHEMA
            or type(case["case_id"]) is not str
            or not case["case_id"]
            or case["case_id"] in identifiers
            or case["phase"] not in {"matched", "development", "confirmation"}
            or case["suite"]
            not in {
                "rescue",
                "terminal",
                "episode",
                "boundary",
                "direct",
                "preservation",
            }
            or type(case["regime"]) is not str
            or not case["regime"]
        ):
            raise ValueError("canonical case identity is malformed or duplicated")
        identifiers.add(case["case_id"])
        expected_split = _expected_case_split(case)
        _validate_case_provenance(
            case,
            expected_split=expected_split,
            expected_generator_id=_expected_case_generator_id(case["phase"]),
            required_source_fields=(
                "case_id",
                "phase",
                "suite",
                "regime",
                "operation",
                "width",
                "position",
                "target",
            ),
        )
        _validate_token_ids(case["expected_token_ids"], "expected", nonempty=True)
        if case["operation"] is not None and case["operation"] not in {"add", "sub"}:
            raise ValueError("canonical case operation is malformed")
        if case["width"] is not None and (
            type(case["width"]) is not int or case["width"] < 2
        ):
            raise ValueError("canonical case width is malformed")
        if case["position"] is not None and (
            type(case["position"]) is not int
            or case["width"] is None
            or not 0 <= case["position"] < case["width"]
        ):
            raise ValueError("canonical case position is malformed")
        if case["target"] is not None and case["target"] not in {0, 1}:
            raise ValueError("canonical case target is malformed")
        _validate_transition(
            case["expected_transition"],
            "expected",
            required=case["suite"] in {"terminal", "episode"},
        )
        serializer_required = case["suite"] in {"terminal", "episode"}
        if case["expected_serializer_token_ids"] is None and not serializer_required:
            pass
        else:
            _validate_token_ids(
                case["expected_serializer_token_ids"],
                "expected serializer",
                nonempty=serializer_required,
            )
        if case["suite"] in {"rescue", "terminal", "episode", "boundary", "direct"}:
            if type(case["expected_episode_output"]) is not int:
                raise ValueError("canonical episode output is malformed")
        elif case["expected_episode_output"] is not None:
            raise ValueError("preservation case has an unexpected episode output")
        if expected_split == "fit_width_audit" and (
            case["phase"] != "development"
            or case["suite"] != "episode"
            or case["width"] not in NUISANCE_TRAIN_WIDTHS
        ):
            raise ValueError("width8 or confirmation entered model selection")

    _validate_frozen_matched_manifest(cases)
    expected_ids = {
        "development_episode": {
            f"development-episode-{index:03d}" for index in range(300)
        },
        "development_boundary": {
            f"development-boundary-{index:03d}" for index in range(50)
        },
        "development_direct": {
            f"development-direct-{index:03d}" for index in range(12)
        },
        "confirmation_episode": {
            f"confirmation-episode-{index:03d}" for index in range(256)
        },
    }
    for label, expected in expected_ids.items():
        phase, suite = label.split("_", 1)
        observed = {
            case["case_id"]
            for case in cases
            if case["phase"] == phase and case["suite"] == suite
        }
        if observed != expected:
            raise ValueError(f"canonical manifest {label} coverage mismatch")
    for phase in ("matched", "development", "confirmation"):
        if not any(
            case["phase"] == phase and case["suite"] == "preservation" for case in cases
        ):
            raise ValueError(f"canonical manifest lacks {phase} preservation cases")

    selection = [
        case for case in cases if case["split_membership"] == "fit_width_audit"
    ]
    if {case["case_id"] for case in selection} != {
        f"development-episode-{index:03d}" for index in range(40)
    }:
        raise ValueError("fit-width audit split membership is incomplete")
    selection_cells = collections.defaultdict(set)
    for case in selection:
        selection_cells[(case["operation"], case["width"], case["position"])].add(
            case["target"]
        )
    if set(selection_cells) != set(NUISANCE_FIT_CELLS) or any(
        targets != {0, 1} for targets in selection_cells.values()
    ):
        raise ValueError("model selection has a missing fit metadata cell")

    for phase in ("development", "confirmation"):
        transition_targets = collections.defaultdict(set)
        serializer_targets = collections.defaultdict(set)
        for case in cases:
            if case["phase"] != phase or case["suite"] != "episode":
                continue
            transition_targets[
                (case["operation"], case["width"], case["position"])
            ].add(case["target"])
            serializer_targets[case["width"]].add(case["target"])
        if len(transition_targets) < 2 or any(
            targets != {0, 1} for targets in transition_targets.values()
        ):
            raise ValueError(f"{phase} positive/negative transition strata incomplete")
        if len(serializer_targets) < 2 or any(
            targets != {0, 1} for targets in serializer_targets.values()
        ):
            raise ValueError(f"{phase} serializer strata incomplete")
        if not any(width not in NUISANCE_TRAIN_WIDTHS for width in serializer_targets):
            raise ValueError(f"{phase} lacks post-freeze unseen-width cases")


def _derive_split_receipt(cases, fit_rows):
    memberships = []
    for case in cases:
        memberships.append(
            _validate_case_provenance(
                case,
                expected_split=_expected_case_split(case),
                expected_generator_id=_expected_case_generator_id(case["phase"]),
                required_source_fields=(
                    "case_id",
                    "phase",
                    "suite",
                    "regime",
                    "operation",
                    "width",
                    "position",
                    "target",
                ),
            )
        )
    memberships.extend(_validate_fit_selection_rows(fit_rows))
    memberships.sort(key=lambda item: (item["split"], item["case_id"]))
    by_split = collections.defaultdict(lambda: {"source": set(), "prompt": set()})
    seen_sources = {}
    seen_prompts = {}
    for membership in memberships:
        split = membership["split"]
        source = membership["source_input_sha256"]
        prompt = membership["prompt_sha256"]
        previous_source = seen_sources.setdefault(source, split)
        previous_prompt = seen_prompts.setdefault(prompt, split)
        if previous_source != split or previous_prompt != split:
            raise ValueError("canonical source or prompt crosses sealed splits")
        by_split[split]["source"].add(source)
        by_split[split]["prompt"].add(prompt)
    required_splits = {"fit", "fit_width_audit", "matched", "development", "confirmation"}
    if set(by_split) != required_splits:
        raise ValueError("canonical split membership is incomplete")
    oracle_ids = sorted(
        case["case_id"]
        for case in cases
        if case["phase"] == "matched" and case["suite"] in {"rescue", "terminal"}
    )
    oracle_memberships = {
        membership["case_id"]: membership
        for membership in memberships
        if membership["case_id"] in oracle_ids
    }
    fit_sources = by_split["fit"]["source"]
    fit_prompts = by_split["fit"]["prompt"]
    oracle_source_overlap = sorted(
        item["source_input_sha256"]
        for item in oracle_memberships.values()
        if item["source_input_sha256"] in fit_sources
    )
    oracle_prompt_overlap = sorted(
        item["prompt_sha256"]
        for item in oracle_memberships.values()
        if item["prompt_sha256"] in fit_prompts
    )
    if oracle_source_overlap or oracle_prompt_overlap:
        raise ValueError("matched oracle source or prompt entered fit")
    disjointness = {
        "splits": sorted(required_splits),
        "matched_oracle_case_ids": oracle_ids,
        "fit_case_ids": sorted(
            item["case_id"] for item in memberships if item["split"] == "fit"
        ),
        "oracle_fit_source_overlap": oracle_source_overlap,
        "oracle_fit_prompt_overlap": oracle_prompt_overlap,
        "oracle_excluded_from_fit": True,
    }
    return canonical_json_document(
        {
            "schema": CASE_SPLIT_RECEIPT_SCHEMA,
            "memberships": memberships,
            "memberships_sha256": stable_json_sha256(memberships),
            "disjointness": disjointness,
            "disjointness_sha256": stable_json_sha256(disjointness),
        }
    )


def _validate_supplied_split_receipt(supplied, derived):
    if (
        type(supplied) is not dict
        or set(supplied)
        != {
            "schema",
            "memberships",
            "memberships_sha256",
            "disjointness_sha256",
        }
        or supplied["schema"] != CASE_SPLIT_RECEIPT_SCHEMA
        or supplied["memberships"] != derived["memberships"]
        or supplied["memberships_sha256"] != derived["memberships_sha256"]
        or supplied["disjointness_sha256"] != derived["disjointness_sha256"]
    ):
        raise ValueError("canonical split receipt is missing or forged")


def _derive_feature_record(case, record, deployment_vocabulary):
    if type(record) is not dict or set(record) != _feature_record_expected_keys():
        raise ValueError("canonical result record schema mismatch")
    if (
        record["schema"] != FEATURE_READING_RECORD_SCHEMA
        or record["case_id"] != case["case_id"]
        or record["arm"] not in DOWNSTREAM_EVALUATION_CONTRACT["arms"]
        or record["deployment_vocabulary_sha256"]
        != stable_json_sha256(deployment_vocabulary)
    ):
        raise ValueError("canonical result record identity is malformed")
    _validate_token_ids(record["actual_token_ids"], "actual", nonempty=True)
    vocab_width = deployment_vocabulary["output_vocab_width"]
    _validate_full_vocab_logits(
        record["full_vocab_logits"], "actual", expected_width=vocab_width
    )
    if any(token_id >= vocab_width for token_id in case["expected_token_ids"]):
        raise ValueError("expected token id is outside the full vocabulary")
    if len(record["actual_token_ids"]) != len(record["full_vocab_logits"]):
        raise ValueError("actual token and full-vocabulary output lengths differ")
    if record["actual_token_ids"] != _tokens_from_full_vocab_logits(
        record["full_vocab_logits"]
    ):
        raise ValueError("actual tokens were not derived from full-vocabulary logits")
    _validate_transition(
        record["actual_transition"],
        "actual",
        required=case["expected_transition"] is not None,
    )
    if record["actual_serializer_token_ids"] is None:
        if case["expected_serializer_token_ids"] is not None:
            raise ValueError("required serializer output is missing")
    else:
        _validate_token_ids(
            record["actual_serializer_token_ids"],
            "actual serializer",
            nonempty=True,
        )
    if (
        type(record["motor_gate_trace"]) is not list
        or not record["motor_gate_trace"]
        or any(type(item) is not bool for item in record["motor_gate_trace"])
    ):
        raise ValueError("motor gate trace is malformed")
    for name in ("gate_off_base_logits", "gate_off_arm_logits"):
        _validate_full_vocab_logits(record[name], name, expected_width=vocab_width)
    if len(record["gate_off_base_logits"]) != len(record["gate_off_arm_logits"]):
        raise ValueError("gate-off full-vocabulary output lengths differ")
    base_widths = [len(row) for row in record["gate_off_base_logits"]]
    arm_widths = [len(row) for row in record["gate_off_arm_logits"]]
    if base_widths != arm_widths:
        raise ValueError("gate-off full-vocabulary logit widths differ")
    gate_off_exact = type_strict_equal(
        record["gate_off_base_logits"], record["gate_off_arm_logits"]
    )
    gate_off_base_tokens = _tokens_from_full_vocab_logits(
        record["gate_off_base_logits"]
    )
    gate_off_arm_tokens = _tokens_from_full_vocab_logits(record["gate_off_arm_logits"])
    transition_fields = {
        field: (
            record["actual_transition"][field] == case["expected_transition"][field]
            if case["expected_transition"] is not None
            else None
        )
        for field in ("p", "c", "r", "z")
    }
    derived = {
        "canonical_case_sha256": stable_json_sha256(case),
        "canonical_record_sha256": stable_json_sha256(record),
        "deployment_vocabulary_sha256": stable_json_sha256(
            deployment_vocabulary
        ),
        "actual_token_ids": record["actual_token_ids"],
        "full_vocab_exact": type_strict_equal(
            record["actual_token_ids"], case["expected_token_ids"]
        ),
        "full_vocab_logits_sha256": stable_json_sha256(record["full_vocab_logits"]),
        "transition_fields": transition_fields,
        "actual_transition": record["actual_transition"],
        "transition_exact": (
            all(transition_fields.values())
            if case["expected_transition"] is not None
            else None
        ),
        "serializer_exact": (
            type_strict_equal(
                record["actual_serializer_token_ids"],
                case["expected_serializer_token_ids"],
            )
            if case["expected_serializer_token_ids"] is not None
            else None
        ),
        "actual_serializer_token_ids": record["actual_serializer_token_ids"],
        "serializer_output_sha256": stable_json_sha256(
            {"value": record["actual_serializer_token_ids"]}
        ),
        "episode_exact": (
            type_strict_equal(
                record["actual_episode_output"], case["expected_episode_output"]
            )
            if case["expected_episode_output"] is not None
            else None
        ),
        "actual_episode_output": record["actual_episode_output"],
        "motor_fired": any(record["motor_gate_trace"]),
        "gate_off_exact": gate_off_exact
        and gate_off_base_tokens == gate_off_arm_tokens,
        "gate_off_base_logits_sha256": stable_json_sha256(
            record["gate_off_base_logits"]
        ),
        "gate_off_arm_logits_sha256": stable_json_sha256(record["gate_off_arm_logits"]),
    }
    derived["derived_output_sha256"] = stable_json_sha256(derived)
    return canonical_json_document(derived)


def evaluate_feature_reading_decision(
    evaluation,
    *,
    expected_deployment_vocabulary_sha256,
    expected_fit_model_binding_sha256,
):
    """Derive the complete decision from canonical case and raw output records."""
    if (
        type(evaluation) is not dict
        or set(evaluation)
        != {
            "schema",
            "deployment_vocabulary",
            "canonical_cases",
            "records",
            "fit_selection_raw_evidence",
            "fit_selection_evidence",
            "split_receipt",
        }
        or evaluation["schema"] != FEATURE_READING_EVALUATION_SCHEMA
        or not re.fullmatch(
            r"[0-9a-f]{64}", expected_deployment_vocabulary_sha256
        )
        or not re.fullmatch(r"[0-9a-f]{64}", expected_fit_model_binding_sha256)
    ):
        raise ValueError("canonical feature-reading evaluation schema mismatch")
    deployment_vocabulary = _validate_deployment_vocabulary(
        evaluation["deployment_vocabulary"]
    )
    if (
        stable_json_sha256(deployment_vocabulary)
        != expected_deployment_vocabulary_sha256
    ):
        raise ValueError("deployment vocabulary differs from the sealed fit")
    cases = evaluation["canonical_cases"]
    records = evaluation["records"]
    _validate_canonical_feature_manifest(cases)
    fit_evidence = _validate_fit_selection_evidence(
        evaluation["fit_selection_evidence"],
        evaluation["fit_selection_raw_evidence"],
        deployment_vocabulary,
        expected_fit_model_binding_sha256,
    )
    split_receipt = _derive_split_receipt(
        cases, evaluation["fit_selection_raw_evidence"]["fit_rows"]
    )
    _validate_supplied_split_receipt(evaluation["split_receipt"], split_receipt)
    if type(records) is not list or not records:
        raise ValueError("canonical result records are missing")
    arms = tuple(DOWNSTREAM_EVALUATION_CONTRACT["arms"])
    case_by_id = {case["case_id"]: case for case in cases}
    expected_keys = {(case_id, arm) for case_id in case_by_id for arm in arms}
    record_by_key = {}
    derived_by_key = {}
    for record in records:
        if type(record) is not dict:
            raise ValueError("canonical result record must be a dictionary")
        key = (record.get("case_id"), record.get("arm"))
        if key in record_by_key:
            raise ValueError("duplicate canonical case result")
        if key not in expected_keys:
            raise ValueError("extra or unknown canonical case result")
        record_by_key[key] = record
        derived_by_key[key] = _derive_feature_record(
            case_by_id[key[0]], record, deployment_vocabulary
        )
    if set(record_by_key) != expected_keys:
        raise ValueError("missing canonical case result")

    selection_case_ids = sorted(
        case["case_id"]
        for case in cases
        if case["split_membership"] == "fit_width_audit"
    )
    selection_records = [
        record_by_key[(case_id, arm)] for case_id in selection_case_ids for arm in arms
    ]
    model_selection = canonical_json_document(
        {
            "schema": "carry_motor_nuisance_model_selection_v1",
            "family": NUISANCE_ONLY_CONTROL_ID,
            "capacity_ledger": nuisance_capacity_ledger(),
            "fit_selected_state_sha256": fit_evidence["selected_state_sha256"],
            "fit_selection_evidence_sha256": stable_json_sha256(fit_evidence),
            "fit_model_binding_sha256": expected_fit_model_binding_sha256,
            "deployment_vocabulary_sha256": (
                expected_deployment_vocabulary_sha256
            ),
            "fit_width_development_case_ids": selection_case_ids,
            "fit_width_development_records_sha256": stable_json_sha256(
                selection_records
            ),
            "family_or_checkpoint_changed_by_development": False,
            "width8_selection_access": False,
            "confirmation_selection_access": False,
            "frozen_before_width8_and_confirmation": True,
        }
    )

    def select_cases(*, phase=None, suite=None, predicate=None):
        selected = []
        for case in cases:
            if phase is not None and case["phase"] != phase:
                continue
            if suite is not None and case["suite"] != suite:
                continue
            if predicate is not None and not predicate(case):
                continue
            selected.append(case)
        return selected

    def accuracy(selected_cases, arm, metric, field=None):
        values = []
        for case in selected_cases:
            derived = derived_by_key[(case["case_id"], arm)]
            value = (
                derived["transition_fields"][field]
                if metric == "transition_field"
                else derived[metric]
            )
            if type(value) is not bool:
                raise ValueError(f"required derived metric is absent: {metric}")
            values.append(value)
        if not values:
            raise ValueError(f"required derived metric has no cases: {metric}")
        return sum(values) / len(values), sum(values), len(values)

    checks = {}
    summaries = {}

    def summarize(label, selected_cases, metric, *, field=None):
        arm_values = {
            arm: accuracy(selected_cases, arm, metric, field=field) for arm in arms
        }
        summaries[label] = {
            "case_ids": [case["case_id"] for case in selected_cases],
            "denominator": len(selected_cases),
            "arms": {
                arm: {
                    "correct": value[1],
                    "denominator": value[2],
                    "accuracy": value[0],
                }
                for arm, value in arm_values.items()
            },
        }
        return arm_values

    def strongest_null_gain(label, arm_values, minimum):
        treatment = arm_values["treatment"][0]
        strongest = max(arm_values["constant_bias"][0], arm_values["nuisance_only"][0])
        checks[f"{label}_treatment_over_strongest_null"] = (
            treatment - strongest >= minimum
        )
        checks[f"{label}_constant_bias_noncompensatory"] = (
            treatment - arm_values["constant_bias"][0] >= minimum
        )
        checks[f"{label}_nuisance_only_noncompensatory"] = (
            treatment - arm_values["nuisance_only"][0] >= minimum
        )

    rescue = select_cases(phase="matched", suite="rescue")
    rescue_values = summarize("matched_rescue", rescue, "episode_exact")
    checks["matched_treatment_rescue_exact"] = rescue_values["treatment"][1] == 6
    checks["matched_shuffled_rescue_zero"] = rescue_values["shuffled"][1] == 0
    strongest_null_gain("matched_rescue", rescue_values, 2.0 / 6.0)

    terminal_positive = select_cases(
        phase="matched",
        suite="terminal",
        predicate=lambda case: case["target"] == 1,
    )
    terminal_negative = select_cases(
        phase="matched",
        suite="terminal",
        predicate=lambda case: case["target"] == 0,
    )
    positive_transition = summarize(
        "matched_positive_transition", terminal_positive, "transition_exact"
    )
    checks["matched_positive_transition_exact"] = (
        positive_transition["treatment"][1] == 9
    )
    strongest_null_gain("matched_positive_transition", positive_transition, 2.0 / 9.0)
    positive_episode = summarize(
        "matched_positive_episode", terminal_positive, "episode_exact"
    )
    checks["matched_positive_episode_treatment_exact"] = (
        positive_episode["treatment"][1] == 9
    )
    strongest_null_gain("matched_positive_episode", positive_episode, 2.0 / 9.0)
    negative_episode = summarize(
        "matched_negative_episode", terminal_negative, "episode_exact"
    )
    checks["matched_negative_episode_treatment_exact"] = (
        negative_episode["treatment"][1] == 9
    )
    for field in ("p", "c", "r", "z"):
        for width in range(2, 11):
            selected = [case for case in terminal_positive if case["width"] == width]
            values = summarize(
                f"matched_positive_w{width}_{field}",
                selected,
                "transition_field",
                field=field,
            )
            strongest_null_gain(f"matched_positive_w{width}_{field}", values, 0.15)
    for arm in arms:
        negative_carry = accuracy(terminal_negative, arm, "transition_field", field="c")
        checks[f"matched_negative_carry_preservation_{arm}"] = negative_carry[1] == 9
        established_serializer = [
            case for case in terminal_negative if case["width"] <= 6
        ]
        preserved = accuracy(established_serializer, arm, "serializer_exact")
        checks[f"matched_negative_serializer_preservation_{arm}"] = preserved[1] == 5
    for field in ("p", "c", "r", "z"):
        for width in range(2, 11):
            selected = [case for case in terminal_negative if case["width"] == width]
            values = summarize(
                f"matched_negative_w{width}_{field}",
                selected,
                "transition_field",
                field=field,
            )
            checks[f"matched_negative_w{width}_{field}_treatment_exact"] = (
                values["treatment"][1] == 1
            )
    for target, label in ((1, "positive"), (0, "negative")):
        selected_target = terminal_positive if target == 1 else terminal_negative
        for width in range(2, 11):
            selected = [case for case in selected_target if case["width"] == width]
            values = summarize(
                f"matched_serializer_{label}_w{width}", selected, "serializer_exact"
            )
            if target == 1 or width >= 7:
                strongest_null_gain(
                    f"matched_serializer_{label}_w{width}", values, 0.15
                )
            checks[f"matched_treatment_serializer_{label}_w{width}_exact"] = (
                values["treatment"][1] == 1
            )

    thresholds = DOWNSTREAM_EVALUATION_CONTRACT["mechanism_go"]
    for phase in ("development", "confirmation"):
        phase_episodes = select_cases(phase=phase, suite="episode")
        metric_specs = (
            ("next_carry", "transition_field", "c", 0.15),
            ("one_step_state", "transition_exact", None, 0.15),
            ("autonomous_episode", "episode_exact", None, 0.20),
            ("full_vocab_output", "full_vocab_exact", None, 0.15),
            ("serializer_transfer", "serializer_exact", None, 0.15),
        )
        for label, metric, field, minimum in metric_specs:
            values = summarize(f"{phase}_{label}", phase_episodes, metric, field=field)
            strongest_null_gain(f"{phase}_{label}", values, minimum)
            for comparator in ("base", "shuffled"):
                checks[f"{phase}_{label}_gain_over_{comparator}"] = (
                    values["treatment"][0] - values[comparator][0] >= minimum
                )
            if label == "next_carry":
                checks[f"{phase}_next_carry_absolute"] = (
                    values["treatment"][0] >= thresholds["next_carry_accuracy_min"]
                )
        unseen = [
            case
            for case in phase_episodes
            if case["width"] not in NUISANCE_TRAIN_WIDTHS
        ]
        unseen_values = summarize(f"{phase}_unseen_width", unseen, "episode_exact")
        strongest_null_gain(f"{phase}_unseen_width", unseen_values, 0.15)
        for comparator in ("base", "shuffled"):
            checks[f"{phase}_unseen_width_gain_over_{comparator}"] = (
                unseen_values["treatment"][0] - unseen_values[comparator][0] >= 0.15
            )

        transition_groups = collections.defaultdict(list)
        episode_groups = collections.defaultdict(list)
        serializer_groups = collections.defaultdict(list)
        full_vocab_groups = collections.defaultdict(list)
        for case in phase_episodes:
            transition_groups[
                (case["operation"], case["width"], case["position"], case["target"])
            ].append(case)
            episode_groups[
                (case["regime"], case["operation"], case["width"], case["target"])
            ].append(case)
            serializer_groups[(case["width"], case["target"])].append(case)
            full_vocab_groups[(case["regime"], case["width"], case["target"])].append(
                case
            )
        for identity, selected in sorted(transition_groups.items()):
            suffix = "-".join(str(item) for item in identity)
            for field in ("p", "c", "r", "z"):
                values = summarize(
                    f"{phase}_transition_{suffix}_{field}",
                    selected,
                    "transition_field",
                    field=field,
                )
                strongest_null_gain(
                    f"{phase}_transition_{suffix}_{field}", values, 0.15
                )
        for identity, selected in sorted(episode_groups.items()):
            suffix = "-".join(str(item) for item in identity)
            values = summarize(f"{phase}_episode_{suffix}", selected, "episode_exact")
            strongest_null_gain(f"{phase}_episode_{suffix}", values, 0.20)
        for identity, selected in sorted(serializer_groups.items()):
            suffix = "-".join(str(item) for item in identity)
            values = summarize(
                f"{phase}_serializer_{suffix}", selected, "serializer_exact"
            )
            strongest_null_gain(f"{phase}_serializer_{suffix}", values, 0.15)
        for identity, selected in sorted(full_vocab_groups.items()):
            suffix = "-".join(str(item) for item in identity)
            values = summarize(
                f"{phase}_full_vocab_{suffix}", selected, "full_vocab_exact"
            )
            strongest_null_gain(f"{phase}_full_vocab_{suffix}", values, 0.15)

    boundary = select_cases(phase="development", suite="boundary")
    boundary_values = summarize("development_boundary_cycle", boundary, "episode_exact")
    checks["development_boundary_cycle_absolute"] = (
        boundary_values["treatment"][1] >= thresholds["boundary_cycle_min_correct"]
    )
    strongest_null_gain(
        "development_boundary_cycle",
        boundary_values,
        thresholds["boundary_cycle_gain_over_nuisance_only_min_correct"] / 50.0,
    )
    direct = select_cases(phase="development", suite="direct")
    direct_values = summarize(
        "development_direct_interactions", direct, "episode_exact"
    )
    checks["development_direct_absolute"] = (
        direct_values["treatment"][1] >= thresholds["direct_interactions_min_exact"]
    )
    strongest_null_gain(
        "development_direct_interactions",
        direct_values,
        thresholds["direct_interactions_gain_over_nuisance_only_min_exact"] / 12.0,
    )

    fit_width_cases = [case_by_id[case_id] for case_id in selection_case_ids]
    fit_treatment = accuracy(fit_width_cases, "treatment", "full_vocab_exact")[0]
    fit_base = accuracy(fit_width_cases, "base", "full_vocab_exact")[0]
    checks["fit_width_regression_within_two_points"] = fit_treatment + 0.02 >= fit_base

    for phase in ("matched", "development", "confirmation"):
        preservation = select_cases(phase=phase, suite="preservation")
        for arm in arms:
            exact = accuracy(preservation, arm, "full_vocab_exact")
            gate = accuracy(preservation, arm, "gate_off_exact")
            checks[f"{phase}_preservation_output_{arm}"] = exact[1] == exact[2]
            checks[f"{phase}_preservation_gate_off_{arm}"] = gate[1] == gate[2]
            checks[f"{phase}_preservation_no_false_fire_{arm}"] = all(
                derived_by_key[(case["case_id"], arm)]["motor_fired"] is False
                for case in preservation
            )
    for arm in arms:
        all_gate = accuracy(cases, arm, "gate_off_exact")
        checks[f"all_case_gate_off_full_vocab_identity_{arm}"] = (
            all_gate[1] == all_gate[2]
        )

    coverage = {
        "case_count": len(cases),
        "record_count": len(records),
        "arms": list(arms),
        "phase_case_counts": {
            phase: sum(case["phase"] == phase for case in cases)
            for phase in ("matched", "development", "confirmation")
        },
        "suite_case_counts": {
            f"{phase}:{suite}": sum(
                case["phase"] == phase and case["suite"] == suite for case in cases
            )
            for phase in ("matched", "development", "confirmation")
            for suite in (
                "rescue",
                "terminal",
                "episode",
                "boundary",
                "direct",
                "preservation",
            )
        },
    }
    derived_outputs = [
        {
            "case_id": case_id,
            "arm": arm,
            **derived_by_key[(case_id, arm)],
        }
        for case_id in sorted(case_by_id)
        for arm in arms
    ]
    canonical_cases = sorted(cases, key=lambda case: case["case_id"])
    canonical_records = [
        record_by_key[(case["case_id"], arm)]
        for case in canonical_cases
        for arm in arms
    ]
    evidence = canonical_json_document(
        {
            "schema": FEATURE_READING_EVALUATION_SCHEMA,
            "coverage": coverage,
            "canonical_case_manifest_sha256": stable_json_sha256(canonical_cases),
            "canonical_record_matrix_sha256": stable_json_sha256(canonical_records),
            "deployment_vocabulary": deployment_vocabulary,
            "deployment_vocabulary_sha256": (
                expected_deployment_vocabulary_sha256
            ),
            "fit_selection_raw_evidence_sha256": scientific_tree_sha256(
                evaluation["fit_selection_raw_evidence"]
            ),
            "split_receipt": split_receipt,
            "per_case_evidence": derived_outputs,
            "derived_outputs_sha256": stable_json_sha256(derived_outputs),
            "model_selection": model_selection,
            "strata": summaries,
        }
    )
    return canonical_json_document(
        {
            "schema": FEATURE_READING_DECISION_SCHEMA,
            "passed": all(checks.values()),
            "checks": checks,
            "evidence": evidence,
            "claim_boundary": (
                "A pass supports only a residual-dependent carry-actuator hypothesis "
                "beyond the frozen constant and saturated metadata-only nulls; it is "
                "not a general or autonomous reasoning proof."
            ),
        }
    )


def scientific_tree_sha256(value):
    """Hash a complete type-strict tensor/tree payload without pickle."""
    digest = hashlib.sha256()

    def update(item):
        if item is None:
            digest.update(b"N")
        elif type(item) is bool:
            digest.update(b"B1" if item else b"B0")
        elif type(item) is int:
            digest.update(b"I" + str(item).encode("ascii") + b"\0")
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("scientific tree contains non-finite float")
            digest.update(b"F" + item.hex().encode("ascii") + b"\0")
        elif type(item) is str:
            payload = item.encode("utf-8")
            digest.update(b"S" + len(payload).to_bytes(8, "big") + payload)
        elif type(item) is torch.Tensor:
            tensor = item.detach().cpu().contiguous()
            digest.update(b"T")
            update(str(tensor.dtype))
            update(list(tensor.shape))
            if tensor.dtype == torch.bfloat16:
                payload = tensor.view(torch.uint16).numpy().tobytes()
            else:
                payload = tensor.numpy().tobytes()
            digest.update(len(payload).to_bytes(8, "big") + payload)
        elif type(item) is list:
            digest.update(b"L" + len(item).to_bytes(8, "big"))
            for child in item:
                update(child)
        elif type(item) is tuple:
            digest.update(b"U" + len(item).to_bytes(8, "big"))
            for child in item:
                update(child)
        elif type(item) in {dict, collections.OrderedDict}:
            digest.update(
                (b"O" if type(item) is collections.OrderedDict else b"D")
                + len(item).to_bytes(8, "big")
            )
            keys = sorted(
                item, key=lambda key: canonical_json_payload(_typed_tree(key))
            )
            for key in keys:
                update(key)
                update(item[key])
        else:
            raise ValueError(
                f"scientific tree contains unsupported type: {type(item).__name__}"
            )

    update(value)
    return digest.hexdigest()


def _typed_tree(value):
    if value is None:
        return ["none", None]
    if type(value) is bool:
        return ["bool", value]
    if type(value) is int:
        return ["int", str(value)]
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("typed tree contains non-finite float")
        return ["float", value.hex()]
    if type(value) is str:
        return ["str", value]
    if type(value) is list:
        return ["list", [_typed_tree(item) for item in value]]
    if type(value) is dict:
        entries = []
        for key, item in value.items():
            if type(key) not in {str, int}:
                raise ValueError("typed tree contains unsupported mapping key")
            entries.append([_typed_tree(key), _typed_tree(item)])
        entries.sort(key=canonical_json_payload)
        return ["dict", entries]
    raise ValueError(f"typed tree contains unsupported type: {type(value).__name__}")


def typed_tree_sha256(value):
    return stable_json_sha256(_typed_tree(value))


def deployment_vocabulary_binding(tokenizer, plan, deployment_logit_dtype):
    if (
        type(plan) is not dict
        or plan.get("vocab_size") != EXPECTED_BASE_PARAMETER_CONFIG["vocab_size"]
        or plan.get("d_model") != EXPECTED_BASE_PARAMETER_CONFIG["d_model"]
        or deployment_logit_dtype != "torch.bfloat16"
    ):
        raise ValueError("deployment vocabulary differs from the frozen model")
    vocab = tokenizer.get_vocab(with_added_tokens=True)
    if type(vocab) is not dict or len(vocab) != plan["vocab_size"]:
        raise ValueError("tokenizer vocabulary width differs from the deployed head")
    ordered_tokens = [None] * plan["vocab_size"]
    for token, token_id in vocab.items():
        if (
            type(token) is not str
            or type(token_id) is not int
            or not 0 <= token_id < len(ordered_tokens)
            or ordered_tokens[token_id] is not None
        ):
            raise ValueError("tokenizer vocabulary ordering is malformed")
        ordered_tokens[token_id] = token
    if any(token is None for token in ordered_tokens):
        raise ValueError("tokenizer vocabulary ordering has a missing token id")
    binding = {
        "schema": DEPLOYMENT_VOCABULARY_SCHEMA,
        "base_checkpoint_sha256": upstream.EXPECTED_CHECKPOINT_SHA256,
        "tokenizer_sha256": upstream.EXPECTED_TOKENIZER_SHA256,
        "output_vocab_width": plan["vocab_size"],
        "token_id_order": list(range(plan["vocab_size"])),
        "token_id_order_sha256": stable_json_sha256(
            list(range(plan["vocab_size"]))
        ),
        "tokenizer_id_to_token_sha256": stable_json_sha256(ordered_tokens),
        "zero_id": plan["zero_id"],
        "one_id": plan["one_id"],
        "deployment_logit_dtype": deployment_logit_dtype,
    }
    return _validate_deployment_vocabulary(binding)


def build_fit_model_binding(context, deployment_vocabulary):
    deployment_vocabulary = _validate_deployment_vocabulary(
        deployment_vocabulary
    )
    plan = context["upstream_plan"]
    rows = canonical_fit_selection_rows(context)
    features = context["features"]
    expected_other_ids = _deployment_other_token_ids(deployment_vocabulary)
    binding = {
        "schema": FIT_MODEL_BINDING_SCHEMA,
        "base_checkpoint_sha256": context["expected_bindings"][
            "base_checkpoint_sha256"
        ],
        "tokenizer_sha256": context["expected_bindings"]["tokenizer_sha256"],
        "upstream_plan_sha256": UPSTREAM_PLAN_SHA256,
        "upstream_source_contract_sha256": stable_json_sha256(
            context["upstream_source_contract"]
        ),
        "upstream_shard_receipts_sha256": stable_json_sha256(
            context["shard_receipts"]
        ),
        "feature_merge_sha256": stable_json_sha256(context["feature_merge"]),
        "fit_rows_sha256": stable_json_sha256(rows),
        "raw_feature_payload_sha256": scientific_tree_sha256(
            {
                "base01": features["base01"],
                "other_lse": features["other_lse"],
                "labels": features["labels"],
                "other_token_ids": features["other_token_ids"],
            }
        ),
        "deployment_vocabulary_sha256": stable_json_sha256(
            deployment_vocabulary
        ),
        "deployment_logit_dtype": features["deployment_logit_dtype"],
        "parameter_ledger": deployment_parameter_ledger(
            dict(EXPECTED_BASE_PARAMETER_CONFIG), plan["fit_budget"]["rank"]
        ),
    }
    if (
        binding["base_checkpoint_sha256"]
        != upstream.EXPECTED_CHECKPOINT_SHA256
        or binding["tokenizer_sha256"] != upstream.EXPECTED_TOKENIZER_SHA256
        or binding["deployment_logit_dtype"] != "torch.bfloat16"
        or features["base01"].dtype != torch.bfloat16
        or features["other_lse"].dtype != torch.float32
        or features["labels"].dtype != torch.int64
        or features["other_token_ids"].dtype != torch.int64
        or not REVIEWED_TORCH_EQUAL(
            features["other_token_ids"].cpu(), expected_other_ids
        )
    ):
        raise ValueError("fit evidence is not bound to the deployed bfloat16 model")
    return canonical_json_document(binding)


def _context_fit_model_binding(context):
    try:
        deployment_vocabulary = context["expected_recovery_plan"]["fit_contract"][
            "deployment_vocabulary"
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError("recovery context lacks the bound deployment vocabulary") from exc
    return build_fit_model_binding(context, deployment_vocabulary)


def normalization_contract():
    return {
        "schema": NORMALIZATION_SCHEMA,
        "upstream_board_rows_sha256": UPSTREAM_BOARD_ROWS_SHA256,
        "canonical_board_sha256": UPSTREAM_CANONICAL_BOARD_SHA256,
        "mismatch_ledger_sha256": NORMALIZATION_MISMATCH_LEDGER_SHA256,
        "expected_mismatches": list(EXPECTED_NORMALIZATION_MISMATCHES),
        "allowed_transformation": ALLOWED_TRANSFORMATION,
    }


def build_normalization_proof(generated_board, sealed_board, rows):
    """Prove that strict JSON key normalization is the sole board difference."""
    generated = copy.deepcopy(generated_board)
    generated["rows_sha256"] = REVIEWED_STABLE_JSON_SHA256(rows)
    if generated["rows_sha256"] != UPSTREAM_BOARD_ROWS_SHA256:
        raise ValueError("generated fit rows differ from sealed upstream identity")
    if stable_json_sha256(sealed_board) != UPSTREAM_CANONICAL_BOARD_SHA256:
        raise ValueError("sealed upstream board identity mismatch")
    if set(generated) != set(sealed_board):
        raise ValueError("generated and sealed board schemas differ")

    histogram_fields = {
        "prompt_length_histogram",
        "token_length_histogram",
    }
    for key in sorted(set(generated) - histogram_fields):
        if not type_strict_equal(generated[key], sealed_board[key]):
            raise ValueError(f"non-histogram board difference: {key}")

    observed = []
    for field in ("prompt_length_histogram", "token_length_histogram"):
        raw = generated[field]
        sealed = sealed_board[field]
        if type(raw) is not dict or type(sealed) is not dict:
            raise ValueError(f"{field} is not a mapping")
        if not raw or any(type(key) is not int for key in raw):
            raise ValueError(f"{field} generated keys are not all integers")
        if not sealed or any(type(key) is not str for key in sealed):
            raise ValueError(f"{field} sealed keys are not all strings")
        if any(type(value) is not int for value in (*raw.values(), *sealed.values())):
            raise ValueError(f"{field} counts are not exact integers")
        if {str(key): value for key, value in raw.items()} != sealed:
            raise ValueError(f"{field} differs beyond JSON key typing")
        observed.append(
            {
                "path": f"board.{field}",
                "generated_key_type": "int",
                "sealed_key_type": "str",
                "generated_keys": sorted(raw),
                "sealed_keys": sorted(sealed, key=int),
            }
        )
    if not type_strict_equal(observed, list(EXPECTED_NORMALIZATION_MISMATCHES)):
        raise ValueError(
            "normalization mismatch ledger is not the frozen two-entry ledger"
        )
    if stable_json_sha256(observed) != NORMALIZATION_MISMATCH_LEDGER_SHA256:
        raise ValueError("normalization mismatch ledger hash mismatch")

    normalized = normalize_generated_board_document(generated)
    if not type_strict_equal(normalized, sealed_board):
        raise ValueError("strict JSON normalization does not reproduce sealed board")
    if stable_json_sha256(normalized) != UPSTREAM_CANONICAL_BOARD_SHA256:
        raise ValueError("normalized board hash mismatch")
    return {
        "schema": NORMALIZATION_SCHEMA,
        "generated_rows": len(rows),
        "generated_rows_sha256": generated["rows_sha256"],
        "generated_board_typed_sha256": typed_tree_sha256(generated),
        "canonical_board_sha256": stable_json_sha256(normalized),
        "mismatch_count": len(observed),
        "mismatches": observed,
        "mismatch_ledger_sha256": stable_json_sha256(observed),
        "canonical_board_equal": True,
        "allowed_transformation": ALLOWED_TRANSFORMATION,
    }, normalized


def _stable_directory_identity(observed):
    return {
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "mode": stat.S_IMODE(observed.st_mode),
        "uid": observed.st_uid,
        "gid": observed.st_gid,
    }


def _bound_directory_identity(observed):
    return {
        **_stable_directory_identity(observed),
        "links": observed.st_nlink,
    }


def _open_physical_directory_chain(path, label):
    """Open every lexical ancestor without following a symlink component."""
    raw = os.fspath(path)
    expected = Path(raw)
    if (
        not expected.is_absolute()
        or os.path.normpath(raw) != raw
        or ".." in expected.parts
        or "." in expected.parts
    ):
        raise ValueError(f"{label} is not an exact absolute lexical path")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    opened = []
    current_path = Path(expected.anchor)
    current_fd = os.open(expected.anchor, flags)
    try:
        current_stat = os.fstat(current_fd)
        if not stat.S_ISDIR(current_stat.st_mode):
            raise ValueError(f"{label} filesystem root is not a directory")
        opened.append(
            {
                "path": str(current_path),
                "fd": current_fd,
                "identity": _stable_directory_identity(current_stat),
            }
        )
        current_fd = None
        for component in expected.parts[1:]:
            parent_fd = opened[-1]["fd"]
            linked = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISDIR(linked.st_mode) or stat.S_ISLNK(linked.st_mode):
                raise ValueError(f"{label} has a symlinked or non-directory ancestor")
            child_fd = os.open(component, flags, dir_fd=parent_fd)
            child = os.fstat(child_fd)
            if _stable_directory_identity(child) != _stable_directory_identity(linked):
                os.close(child_fd)
                raise RuntimeError(f"{label} ancestor changed during descriptor walk")
            current_path /= component
            opened.append(
                {
                    "path": str(current_path),
                    "fd": child_fd,
                    "identity": _stable_directory_identity(child),
                }
            )
        if current_path != expected:
            raise AssertionError("physical directory walk did not reach expected path")
        return opened
    except Exception:
        if current_fd is not None:
            os.close(current_fd)
        for item in reversed(opened):
            os.close(item["fd"])
        raise


def _verify_physical_directory_chain(chain, label):
    for item in chain:
        linked = os.stat(item["path"], follow_symlinks=False)
        opened = os.fstat(item["fd"])
        if (
            not stat.S_ISDIR(linked.st_mode)
            or stat.S_ISLNK(linked.st_mode)
            or _stable_directory_identity(linked) != item["identity"]
            or _stable_directory_identity(opened) != item["identity"]
        ):
            raise RuntimeError(f"{label} ancestor chain changed after binding")


def capture_physical_directory_binding(path, mode, label, children=None):
    chain = _open_physical_directory_chain(path, label)
    try:
        leaf = chain[-1]
        if mode is not None and leaf["identity"]["mode"] != int(mode):
            raise ValueError(f"{label} directory mode mismatch")
        observed_children = sorted(os.listdir(leaf["fd"]))
        if children is not None and observed_children != sorted(children):
            raise ValueError(f"{label} directory is not closed-world")
        _verify_physical_directory_chain(chain, label)
        return canonical_json_document(
            {
                "path": str(Path(path)),
                "identity": leaf["identity"],
                "descriptor_identity": _bound_directory_identity(os.fstat(leaf["fd"])),
                "ancestor_chain": [
                    {"path": item["path"], "identity": item["identity"]}
                    for item in chain
                ],
                "children": observed_children,
            }
        )
    finally:
        for item in reversed(chain):
            os.close(item["fd"])


class BoundFile:
    """Exact lexical path plus open-descriptor identity and digest binding."""

    def __init__(
        self,
        path,
        expected_path,
        expected_sha256,
        label,
        *,
        required_mode=None,
        required_parent_mode=None,
    ):
        raw = os.fspath(path)
        expected = Path(expected_path)
        if raw != str(expected) or not expected.is_absolute():
            raise ValueError(f"{label} path aliases or differs from frozen path")
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected_sha256)):
            raise ValueError(f"{label} receipt is invalid")
        self.path = expected
        self.label = label
        self.handle = None
        self._ancestor_chain = _open_physical_directory_chain(expected.parent, label)
        try:
            parent = self._ancestor_chain[-1]
            if required_parent_mode is not None and parent["identity"]["mode"] != int(
                required_parent_mode
            ):
                raise ValueError(f"{label} parent mode mismatch")
            path_stat = os.stat(
                expected.name, dir_fd=parent["fd"], follow_symlinks=False
            )
            if (
                not stat.S_ISREG(path_stat.st_mode)
                or stat.S_ISLNK(path_stat.st_mode)
                or path_stat.st_nlink != 1
            ):
                raise ValueError(f"{label} is not a one-link regular file")
            if required_mode is not None and stat.S_IMODE(path_stat.st_mode) != int(
                required_mode
            ):
                raise ValueError(f"{label} mode mismatch")
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(expected.name, flags, dir_fd=parent["fd"])
            self.handle = os.fdopen(descriptor, "rb")
            opened = os.fstat(self.handle.fileno())
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or (
                    required_mode is not None
                    and stat.S_IMODE(opened.st_mode) != int(required_mode)
                )
            ):
                raise ValueError(f"{label} opened descriptor identity mismatch")
            self.identity = _stat_identity(opened)
            if self.identity != _stat_identity(path_stat):
                raise ValueError(f"{label} changed during binding")
            self.sha256 = self._hash_handle()
            if self.sha256 != expected_sha256:
                raise ValueError(f"{label} artifact hash mismatch")
            _verify_physical_directory_chain(self._ancestor_chain, label)
        except Exception:
            self.close()
            raise

    def _hash_handle(self):
        digest = hashlib.sha256()
        self.handle.seek(0)
        for block in iter(lambda: self.handle.read(1024 * 1024), b""):
            digest.update(block)
        self.handle.seek(0)
        return digest.hexdigest()

    def bytes(self):
        self.handle.seek(0)
        payload = self.handle.read()
        self.handle.seek(0)
        return payload

    def text(self):
        return self.bytes().decode("utf-8")

    def verify(self):
        _verify_physical_directory_chain(self._ancestor_chain, self.label)
        parent = self._ancestor_chain[-1]
        observed = os.stat(self.path.name, dir_fd=parent["fd"], follow_symlinks=False)
        linked_identity = _stat_identity(observed)
        opened_identity = _stat_identity(os.fstat(self.handle.fileno()))
        if (
            linked_identity != self.identity
            or opened_identity != self.identity
            or self._hash_handle() != self.sha256
        ):
            raise RuntimeError(f"{self.label} changed after binding")

    def close(self):
        if self.handle is not None and not self.handle.closed:
            self.handle.close()
        for item in reversed(getattr(self, "_ancestor_chain", ())):
            try:
                os.close(item["fd"])
            except OSError:
                pass
        self._ancestor_chain = []


def _code_constant_document(value):
    if inspect.iscode(value):
        return {"type": "code", "value": _code_object_document(value)}
    if value is None:
        return {"type": "none"}
    if value is Ellipsis:
        return {"type": "ellipsis"}
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "int", "value": str(value)}
    if type(value) is float:
        return {"type": "float", "value": value.hex()}
    if type(value) is complex:
        return {
            "type": "complex",
            "real": value.real.hex(),
            "imag": value.imag.hex(),
        }
    if type(value) is str:
        return {"type": "str", "value": value}
    if type(value) is bytes:
        return {"type": "bytes", "value": value.hex()}
    if type(value) is tuple:
        return {
            "type": "tuple",
            "value": [_code_constant_document(item) for item in value],
        }
    if type(value) is frozenset:
        members = [_code_constant_document(item) for item in value]
        members.sort(key=canonical_json_payload)
        return {"type": "frozenset", "value": members}
    raise TypeError(f"unsupported Python code constant: {type(value).__qualname__}")


def _code_object_document(code):
    return {
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "code_hex": code.co_code.hex(),
        "constants": [_code_constant_document(value) for value in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
        "filename": code.co_filename,
        "name": code.co_name,
        "qualname": code.co_qualname,
        "firstlineno": code.co_firstlineno,
        "linetable_hex": code.co_linetable.hex(),
        "exceptiontable_hex": code.co_exceptiontable.hex(),
    }


def _code_sha256(value):
    code = getattr(value, "__code__", None)
    if code is None:
        return None
    return sha256_bytes(
        canonical_json_payload(_code_object_document(code)).encode("ascii")
    )


def _callable_descriptor(value):
    module_name = getattr(value, "__module__", None)
    module = sys.modules.get(module_name)
    module_path = getattr(module, "__file__", None)
    descriptor = {
        "module": module_name,
        "qualname": getattr(value, "__qualname__", getattr(value, "__name__", None)),
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "code_sha256": _code_sha256(value),
        "module_path": str(Path(module_path).resolve(strict=True))
        if module_path is not None
        else None,
    }
    if inspect.isclass(value):
        methods = {}
        for name, member in sorted(vars(value).items()):
            if isinstance(member, (staticmethod, classmethod)):
                member = member.__func__
            digest = _code_sha256(member)
            if digest is not None:
                methods[name] = digest
        descriptor["class_method_code_sha256"] = methods
    return canonical_json_document(descriptor)


def _reviewed_export_registry():
    registry = {
        "torch.optim.AdamW": (lambda: torch.optim.AdamW, REVIEWED_ADAMW),
        "torch._C": (lambda: torch._C, torch_c_module),
        "tokenizers.tokenizers": (
            lambda: tokenizers_module.tokenizers,
            tokenizers_native_module,
        ),
        "torch.load": (lambda: torch.load, REVIEWED_TORCH_LOAD),
        "torch.save": (lambda: torch.save, REVIEWED_TORCH_SAVE),
        "torch.serialization.safe_globals": (
            lambda: torch.serialization.safe_globals,
            REVIEWED_SAFE_GLOBALS,
        ),
        "torch.torch_version.TorchVersion": (
            lambda: torch.torch_version.TorchVersion,
            REVIEWED_TORCH_VERSION_TYPE,
        ),
        "torch.nn.Linear": (lambda: torch.nn.Linear, REVIEWED_LINEAR),
        "torch.nn.functional.cross_entropy": (
            lambda: torch.nn.functional.cross_entropy,
            REVIEWED_CROSS_ENTROPY,
        ),
        "upstream.CarryMotor": (lambda: upstream.CarryMotor, REVIEWED_CARRY_MOTOR),
        "upstream._batch_schedule": (
            lambda: upstream._batch_schedule,
            REVIEWED_BATCH_SCHEDULE,
        ),
        "upstream.full_vocab_motor_loss": (
            lambda: upstream.full_vocab_motor_loss,
            REVIEWED_FULL_VOCAB_MOTOR_LOSS,
        ),
        "upstream.initial_motor_state": (
            lambda: upstream.initial_motor_state,
            REVIEWED_INITIAL_MOTOR_STATE,
        ),
        "upstream.fit_linear_diagnostic": (
            lambda: upstream.fit_linear_diagnostic,
            REVIEWED_FIT_LINEAR_DIAGNOSTIC,
        ),
        "upstream.canonical_fit_teacher_forced_evidence": (
            lambda: upstream.canonical_fit_teacher_forced_evidence,
            REVIEWED_CANONICAL_FIT_EVIDENCE,
        ),
        "upstream.merge_feature_shards": (
            lambda: upstream.merge_feature_shards,
            REVIEWED_MERGE_FEATURE_SHARDS,
        ),
        "upstream._validate_motor_bundle_against_replayed_features": (
            lambda: upstream._validate_motor_bundle_against_replayed_features,
            REVIEWED_VALIDATE_MOTOR_BUNDLE,
        ),
        "upstream.generate_fit_rows": (
            lambda: upstream.generate_fit_rows,
            REVIEWED_GENERATE_FIT_ROWS,
        ),
        "upstream.permuted_control_labels": (
            lambda: upstream.permuted_control_labels,
            REVIEWED_PERMUTED_CONTROL_LABELS,
        ),
        "upstream.tensor_state_sha256": (
            lambda: upstream.tensor_state_sha256,
            REVIEWED_TENSOR_STATE_SHA256,
        ),
        "model.GPTConfig": (lambda: model_module.GPTConfig, REVIEWED_GPT_CONFIG),
        "upstream.require_canonical_cuda_runtime": (
            lambda: upstream.require_canonical_cuda_runtime,
            REVIEWED_REQUIRE_CANONICAL_CUDA,
        ),
        "torch.manual_seed": (lambda: torch.manual_seed, REVIEWED_MANUAL_SEED),
        "torch.cuda.manual_seed_all": (
            lambda: torch.cuda.manual_seed_all,
            REVIEWED_CUDA_MANUAL_SEED_ALL,
        ),
        "torch.as_tensor": (lambda: torch.as_tensor, REVIEWED_AS_TENSOR),
        "torch.tensor": (lambda: torch.tensor, REVIEWED_TENSOR),
        "torch.isfinite": (lambda: torch.isfinite, REVIEWED_ISFINITE),
        "torch.randperm": (lambda: torch.randperm, REVIEWED_RANDPERM),
        "torch.logsumexp": (lambda: torch.logsumexp, REVIEWED_LOGSUMEXP),
        "torch.logaddexp": (lambda: torch.logaddexp, REVIEWED_LOGADDEXP),
        "torch.stack": (lambda: torch.stack, REVIEWED_STACK),
        "torch.nextafter": (lambda: torch.nextafter, REVIEWED_NEXTAFTER),
        "torch.equal": (lambda: torch.equal, REVIEWED_TORCH_EQUAL),
        "torch.Generator": (lambda: torch.Generator, REVIEWED_TORCH_GENERATOR),
        "torch.no_grad": (lambda: torch.no_grad, REVIEWED_TORCH_NO_GRAD),
        "torch.get_default_dtype": (
            lambda: torch.get_default_dtype,
            REVIEWED_GET_DEFAULT_DTYPE,
        ),
        "torch.get_float32_matmul_precision": (
            lambda: torch.get_float32_matmul_precision,
            REVIEWED_GET_FLOAT32_MATMUL_PRECISION,
        ),
        "torch.are_deterministic_algorithms_enabled": (
            lambda: torch.are_deterministic_algorithms_enabled,
            REVIEWED_ARE_DETERMINISTIC_ALGORITHMS_ENABLED,
        ),
        "torch.is_deterministic_algorithms_warn_only_enabled": (
            lambda: torch.is_deterministic_algorithms_warn_only_enabled,
            REVIEWED_IS_DETERMINISTIC_WARN_ONLY_ENABLED,
        ),
        "torch.get_num_threads": (
            lambda: torch.get_num_threads,
            REVIEWED_GET_NUM_THREADS,
        ),
        "torch.get_num_interop_threads": (
            lambda: torch.get_num_interop_threads,
            REVIEWED_GET_NUM_INTEROP_THREADS,
        ),
        "torch.cuda.device_count": (
            lambda: torch.cuda.device_count,
            REVIEWED_CUDA_DEVICE_COUNT,
        ),
        "torch.cuda.get_device_name": (
            lambda: torch.cuda.get_device_name,
            REVIEWED_CUDA_GET_DEVICE_NAME,
        ),
        "upstream.stable_json_sha256": (
            lambda: upstream.stable_json_sha256,
            REVIEWED_STABLE_JSON_SHA256,
        ),
        "tokenizers.Tokenizer.from_str": (
            lambda: Tokenizer.from_str,
            REVIEWED_TOKENIZER_FROM_STR,
        ),
        "upstream.rollout_episode": (
            lambda: upstream.rollout_episode,
            REVIEWED_ROLLOUT_EPISODE,
        ),
        "digitwise_controller.rollout_episode": (
            lambda: digitwise_controller_module.rollout_episode,
            REVIEWED_ROLLOUT_EPISODE,
        ),
        "torch.optim.adamw.adamw": (
            lambda: torch_adamw_module.adamw,
            REVIEWED_ADAMW_FUNCTIONAL,
        ),
        "torch.optim.adam.adam": (
            lambda: torch_adam_module.adam,
            REVIEWED_ADAM_FUNCTIONAL,
        ),
        "torch.optim.AdamW.step": (
            lambda: torch.optim.AdamW.step,
            REVIEWED_ADAMW_STEP,
        ),
        "torch.optim.Optimizer.zero_grad": (
            lambda: torch.optim.Optimizer.zero_grad,
            REVIEWED_OPTIMIZER_ZERO_GRAD,
        ),
        "torch.nn.Module.__call__": (
            lambda: torch.nn.Module.__call__,
            REVIEWED_MODULE_CALL,
        ),
        "torch.nn.Module.to": (
            lambda: torch.nn.Module.to,
            REVIEWED_MODULE_TO,
        ),
        "torch.nn.Module.cpu": (
            lambda: torch.nn.Module.cpu,
            REVIEWED_MODULE_CPU,
        ),
        "torch.nn.Module.eval": (
            lambda: torch.nn.Module.eval,
            REVIEWED_MODULE_EVAL,
        ),
        "torch.nn.Module.parameters": (
            lambda: torch.nn.Module.parameters,
            REVIEWED_MODULE_PARAMETERS,
        ),
        "torch.nn.Module.load_state_dict": (
            lambda: torch.nn.Module.load_state_dict,
            REVIEWED_MODULE_LOAD_STATE_DICT,
        ),
        "torch.nn.Module.state_dict": (
            lambda: torch.nn.Module.state_dict,
            REVIEWED_MODULE_STATE_DICT,
        ),
        "torch.Tensor.backward": (
            lambda: torch.Tensor.backward,
            REVIEWED_TENSOR_BACKWARD,
        ),
    }
    for helper_name, captured in REVIEWED_SERIALIZATION_HELPERS.items():
        registry[f"torch.serialization.{helper_name}"] = (
            lambda name=helper_name: torch_serialization_module.__dict__[name],
            captured,
        )
    return registry


_SEMANTIC_RECURSION_MODULES = frozenset(
    {
        upstream.__name__,
        digitwise_controller_module.__name__,
        digitwise_protocol_module.__name__,
        eval_suite_module.__name__,
        model_module.__name__,
        probe_digitwise_workspace_module.__name__,
        torch_adam_module.__name__,
        torch_adamw_module.__name__,
        torch_functional_module.__name__,
        torch_linear_module.__name__,
        torch_module_module.__name__,
        torch_optimizer_module.__name__,
        torch_serialization_module.__name__,
    }
)


def _semantic_data_document(value, active):
    if inspect.ismodule(value):
        module_path = getattr(value, "__file__", None)
        if module_path is not None:
            resolved = Path(module_path).resolve(strict=True)
            path = str(resolved)
            digest = sha256_file(resolved)
        else:
            path = None
            digest = None
        return {
            "kind": "module",
            "name": getattr(value, "__name__", None),
            "path": path,
            "sha256": digest,
        }
    if isinstance(value, functools.partial):
        identity = id(value)
        if identity in active:
            raise ValueError("reviewed semantic dependency contains a cycle")
        active.add(identity)
        try:
            return {
                "kind": "partial",
                "function": _semantic_data_document(value.func, active),
                "args": _semantic_data_document(value.args, active),
                "keywords": _semantic_data_document(value.keywords or {}, active),
            }
        finally:
            active.remove(identity)
    if inspect.ismethod(value):
        return {
            "kind": "bound_method",
            "descriptor": _callable_descriptor(value),
            "self": _semantic_data_document(value.__self__, active),
        }
    if callable(value):
        return {"kind": "callable", "descriptor": _callable_descriptor(value)}
    if value is None:
        return {"kind": "scalar", "type": "none"}
    if type(value) is bool:
        return {"kind": "scalar", "type": "bool", "value": value}
    if type(value) is int:
        return {"kind": "scalar", "type": "int", "value": str(value)}
    if type(value) is float:
        return {"kind": "scalar", "type": "float", "value": value.hex()}
    if type(value) is complex:
        return {
            "kind": "scalar",
            "type": "complex",
            "real": value.real.hex(),
            "imag": value.imag.hex(),
        }
    if type(value) is str:
        return {"kind": "scalar", "type": "str", "value": value}
    if type(value) is bytes:
        return {"kind": "scalar", "type": "bytes", "value": value.hex()}
    if isinstance(value, re.Pattern):
        return {
            "kind": "regular_expression",
            "pattern": value.pattern,
            "flags": value.flags,
        }
    if type(value).__module__ == "typing" and type(value).__qualname__ in {
        "ParamSpec",
        "TypeVar",
    }:
        return {
            "kind": "typing_parameter",
            "type": type(value).__qualname__,
            "name": value.__name__,
            "representation": repr(value),
        }
    if type(value) in {list, tuple, set, frozenset, dict, collections.OrderedDict}:
        identity = id(value)
        if identity in active:
            raise ValueError("reviewed semantic dependency contains a cycle")
        active.add(identity)
        try:
            if type(value) in {dict, collections.OrderedDict}:
                return {
                    "kind": "mapping",
                    "type": f"{type(value).__module__}.{type(value).__qualname__}",
                    "entries": [
                        {
                            "key": _semantic_data_document(key, active),
                            "value": _semantic_data_document(item, active),
                        }
                        for key, item in value.items()
                    ],
                }
            members = [_semantic_data_document(item, active) for item in value]
            if type(value) in {set, frozenset}:
                members.sort(key=canonical_json_payload)
            return {
                "kind": "collection",
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "members": members,
            }
        finally:
            active.remove(identity)
    try:
        state = vars(value)
    except TypeError as exc:
        raise TypeError(
            "unsupported reviewed semantic dependency: "
            f"{type(value).__module__}.{type(value).__qualname__}"
        ) from exc
    identity = id(value)
    if identity in active:
        raise ValueError("reviewed semantic dependency contains a cycle")
    active.add(identity)
    try:
        return {
            "kind": "state_object",
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "state": _semantic_data_document(state, active),
        }
    finally:
        active.remove(identity)


def _semantic_value_descriptor(value):
    return canonical_json_document(_semantic_data_document(value, set()))


def _nested_semantic_functions(value, active=None):
    if inspect.isfunction(value):
        return (value,)
    if inspect.ismethod(value):
        return (value.__func__,)
    if active is None:
        active = set()
    if type(value) not in {
        functools.partial,
        list,
        tuple,
        set,
        frozenset,
        dict,
        collections.OrderedDict,
    }:
        return ()
    identity = id(value)
    if identity in active:
        return ()
    active.add(identity)
    try:
        if isinstance(value, functools.partial):
            children = (value.func, *value.args, *(value.keywords or {}).values())
        elif type(value) in {dict, collections.OrderedDict}:
            children = (*value.keys(), *value.values())
        else:
            children = tuple(value)
        return tuple(
            function
            for child in children
            for function in _nested_semantic_functions(child, active)
        )
    finally:
        active.remove(identity)


_SEMANTIC_RUNTIME_STATE_GLOBALS = frozenset(
    {
        ("torch.cuda", "_cached_device_count"),
        ("torch.cuda", "_initialized"),
    }
)


def _collect_reviewed_semantic_bindings():
    queue = collections.deque()
    registry = _reviewed_export_registry()
    for export_name in sorted(registry):
        _lookup, value = registry[export_name]
        if inspect.isfunction(value):
            queue.append((f"export.{export_name}", value))
        elif inspect.isclass(value):
            for base in value.__mro__:
                if base.__module__ not in _SEMANTIC_RECURSION_MODULES:
                    continue
                for member_name, member in sorted(vars(base).items()):
                    if isinstance(member, (staticmethod, classmethod)):
                        member = member.__func__
                    if inspect.isfunction(member):
                        queue.append(
                            (
                                f"export.{export_name}.mro."
                                f"{base.__module__}.{base.__qualname__}.{member_name}",
                                member,
                            )
                        )
    bindings = {}
    seen = set()
    while queue:
        provenance, function = queue.popleft()
        if not inspect.isfunction(function) or id(function) in seen:
            continue
        seen.add(id(function))
        prefix = f"{provenance}.callable.{function.__module__}.{function.__qualname__}"
        for name in function.__code__.co_names:
            if (
                name not in function.__globals__
                or (
                    function.__module__,
                    name,
                )
                in _SEMANTIC_RUNTIME_STATE_GLOBALS
            ):
                continue
            value = function.__globals__[name]
            label = f"{prefix}.__globals__.{name}"
            existing = bindings.get(label)
            if existing is not None and existing[2] is not value:
                raise RuntimeError(f"reviewed semantic global label collision: {label}")
            bindings[label] = (function.__globals__, name, value)
            for nested in _nested_semantic_functions(value):
                if nested.__module__ in _SEMANTIC_RECURSION_MODULES:
                    queue.append((label, nested))
        closure = function.__closure__ or ()
        for name, cell in zip(function.__code__.co_freevars, closure):
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            label = f"{prefix}.__closure__.{name}"
            existing = bindings.get(label)
            if existing is not None and existing[2] is not value:
                raise RuntimeError(
                    f"reviewed semantic closure label collision: {label}"
                )
            bindings[label] = (cell, None, value)
            for nested in _nested_semantic_functions(value):
                if nested.__module__ in _SEMANTIC_RECURSION_MODULES:
                    queue.append((label, nested))
    return bindings


REVIEWED_SEMANTIC_BINDINGS = _collect_reviewed_semantic_bindings()
REVIEWED_SEMANTIC_DESCRIPTORS = canonical_json_document(
    {
        name: _semantic_value_descriptor(binding[2])
        for name, binding in sorted(REVIEWED_SEMANTIC_BINDINGS.items())
    }
)
REVIEWED_SEMANTIC_CLOSURE_SHA256 = stable_json_sha256(REVIEWED_SEMANTIC_DESCRIPTORS)


REVIEWED_CALLABLE_DESCRIPTORS = canonical_json_document(
    {
        name: _callable_descriptor(value)
        for name, (_lookup, value) in _reviewed_export_registry().items()
    }
)


def assert_reviewed_callable_exports():
    observed = {}
    for name, (lookup, captured) in _reviewed_export_registry().items():
        if lookup() is not captured:
            raise RuntimeError(f"reviewed callable export was monkeypatched: {name}")
        observed[name] = _callable_descriptor(captured)
    if not type_strict_equal(observed, REVIEWED_CALLABLE_DESCRIPTORS):
        raise RuntimeError("reviewed callable descriptor changed after import")
    semantic_descriptors = {}
    for name, (owner, key, captured) in REVIEWED_SEMANTIC_BINDINGS.items():
        current = owner.cell_contents if key is None else owner.get(key)
        if current is not captured:
            raise RuntimeError(
                f"reviewed semantic dependency was monkeypatched: {name}"
            )
        semantic_descriptors[name] = _semantic_value_descriptor(captured)
    if not type_strict_equal(semantic_descriptors, REVIEWED_SEMANTIC_DESCRIPTORS):
        raise RuntimeError("reviewed semantic dependency descriptor changed")
    return canonical_json_document(
        {
            "schema": RECOVERY_CALLABLE_CONTRACT_SCHEMA,
            "exports": observed,
            "exports_sha256": stable_json_sha256(observed),
            "semantic_dependencies": semantic_descriptors,
            "semantic_dependencies_sha256": stable_json_sha256(semantic_descriptors),
        }
    )


def safe_torch_load(bound):
    assert_reviewed_callable_exports()
    for name in DESERIALIZATION_CONTRACT["ambient_override_environment_forbidden"]:
        if name in os.environ:
            raise RuntimeError(
                f"ambient torch deserialization override is forbidden: {name}"
            )
    bound.handle.seek(0)
    with REVIEWED_SAFE_GLOBALS([REVIEWED_TORCH_VERSION_TYPE]):
        value = REVIEWED_TORCH_LOAD(bound.handle, map_location="cpu", weights_only=True)
    bound.handle.seek(0)
    bound.verify()
    return value


def _capture_optional_regular_file(path, label):
    expected = Path(path)
    try:
        observed = os.stat(expected, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if stat.S_ISDIR(observed.st_mode) and not stat.S_ISLNK(observed.st_mode):
        return {
            "kind": "directory",
            **capture_physical_directory_binding(expected, None, label),
        }
    return _capture_executable(expected, label)


def capture_git_repository_contract(repo_root):
    """Bind controls while all reads use a synthetic, configuration-free GIT_DIR."""
    root = Path(repo_root)
    repository = _ConfigExcludedGit(root, copy_index=True)
    try:
        layout = repository.layout
        controls = {
            "worktree_git_pointer": _capture_optional_regular_file(
                root / ".git", "Git worktree pointer"
            ),
            "worktree_head": _capture_optional_regular_file(
                layout["git_dir"] / "HEAD", "Git worktree HEAD"
            ),
            "worktree_index": _capture_optional_regular_file(
                layout["git_dir"] / "index", "Git worktree index"
            ),
            "worktree_commondir": _capture_optional_regular_file(
                layout["git_dir"] / "commondir", "Git worktree commondir"
            ),
            "worktree_config": _capture_optional_regular_file(
                layout["git_dir"] / "config.worktree", "Git worktree config"
            ),
            "common_config": _capture_optional_regular_file(
                layout["common_dir"] / "config", "Git common config"
            ),
            "common_packed_refs": _capture_optional_regular_file(
                layout["common_dir"] / "packed-refs", "Git packed refs"
            ),
            "common_info_attributes": _capture_optional_regular_file(
                layout["common_dir"] / "info" / "attributes",
                "Git info attributes",
            ),
        }
        if controls["worktree_head"] is None or controls["worktree_index"] is None:
            raise ValueError("Git repository control files are incomplete")
        return canonical_json_document(
            {
                "schema": RECOVERY_GIT_CONTRACT_SCHEMA,
                "worktree_root": str(root),
                "worktree_root_identity": _stable_directory_identity(os.lstat(root)),
                "absolute_git_dir": str(layout["git_dir"]),
                "common_git_dir": str(layout["common_dir"]),
                "object_directory": str(layout["objects"]),
                "head_commit": repository.head_commit,
                "object_format": "sha1",
                "configuration_exclusion": GIT_CONFIGURATION_EXCLUSION,
                "control_files_observed_but_never_interpreted": controls,
            }
        )
    finally:
        repository.close()


def validate_loaded_module_paths(repo_root):
    root = Path(repo_root)
    expected = {
        "recovery": root / "train" / "causal_carry_motor_recovery.py",
        "upstream": root / "train" / "causal_carry_motor.py",
        "model": root / "train" / "model.py",
        "digitwise_controller": root / "train" / "digitwise_controller.py",
        "digitwise_protocol": root / "train" / "digitwise_protocol.py",
        "eval_suite": root / "train" / "eval_suite.py",
        "probe_digitwise_workspace": (root / "train" / "probe_digitwise_workspace.py"),
    }
    observed = {
        "recovery": Path(__file__),
        "upstream": Path(upstream.__file__),
        "model": Path(model_module.__file__),
        "digitwise_controller": Path(digitwise_controller_module.__file__),
        "digitwise_protocol": Path(digitwise_protocol_module.__file__),
        "eval_suite": Path(eval_suite_module.__file__),
        "probe_digitwise_workspace": Path(probe_digitwise_workspace_module.__file__),
    }
    for name in expected:
        if (
            os.fspath(observed[name]) != str(expected[name])
            or observed[name].resolve(strict=True) != expected[name]
        ):
            raise ValueError(f"loaded {name} module is shadowed or aliased")
    return {name: str(path) for name, path in expected.items()}


def validate_recovery_commit_topology(
    recovery_source_commit, *, repo_root=None, repository=None
):
    root = Path(repo_root or Path(__file__).resolve().parents[1])
    owned_repository = repository is None
    repository = repository or _ConfigExcludedGit(root, copy_index=False)
    try:
        _tree, parents = _preimport_commit_headers(
            repository.commit(recovery_source_commit), "recovery"
        )
        if parents != [UPSTREAM_SOURCE_COMMIT]:
            raise ValueError(
                "recovery commit must have a0c258e as its sole direct parent"
            )
        recovery_tree = repository.tree(recovery_source_commit)
        upstream_tree = repository.tree(UPSTREAM_SOURCE_COMMIT)
        additions = sorted(set(recovery_tree) - set(upstream_tree))
        removed = set(upstream_tree) - set(recovery_tree)
        modified = {
            path
            for path in set(recovery_tree) & set(upstream_tree)
            if recovery_tree[path] != upstream_tree[path]
        }
        observed = tuple(f"A\t{name}" for name in additions)
        if (
            observed != RECOVERY_NAME_STATUS_DIFF
            or removed
            or modified
            or any(recovery_tree[name][0] != "100644" for name in additions)
        ):
            raise ValueError(
                "recovery commit diff must be exactly four added recovery files"
            )
        return {
            "parent_commit": UPSTREAM_SOURCE_COMMIT,
            "name_status_diff": list(observed),
        }
    finally:
        if owned_repository:
            repository.close()


def validate_checkout_closed_world(repo_root, repository, expected_tree):
    root = Path(repo_root)
    index = repository.index()
    if index != expected_tree:
        raise ValueError("recovery source index differs from reviewed commit")
    _preimport_hash_checkout(root, expected_tree)
    tracked = set(expected_tree)
    return {
        "tracked_file_count": len(tracked),
        "tracked_paths_sha256": stable_json_sha256(sorted(tracked)),
        "tree_entries_sha256": stable_json_sha256(
            [
                {"path": path, "mode": mode, "oid": oid}
                for path, (mode, oid) in sorted(expected_tree.items())
            ]
        ),
        "index_sha256": repository.index_binding["sha256"],
    }


def _capture_executable(path, label):
    expected = Path(path)
    chain = _open_physical_directory_chain(expected.parent, label)
    parent_fd = chain[-1]["fd"]
    observed = os.stat(expected.name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISREG(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
        for item in reversed(chain):
            os.close(item["fd"])
        raise ValueError(f"{label} is not a regular non-symlink file")
    descriptor = os.open(
        expected.name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_fd,
    )
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (observed.st_dev, observed.st_ino):
            raise ValueError(f"{label} changed during binding")
        digest = hashlib.sha256()
        with os.fdopen(os.dup(descriptor), "rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        current = os.stat(expected.name, dir_fd=parent_fd, follow_symlinks=False)
        identity = {
            "device": opened.st_dev,
            "inode": opened.st_ino,
            "mode": stat.S_IMODE(opened.st_mode),
            "links": opened.st_nlink,
            "uid": opened.st_uid,
            "gid": opened.st_gid,
            "size": opened.st_size,
            "mtime_ns": opened.st_mtime_ns,
            "ctime_ns": opened.st_ctime_ns,
        }
        current_identity = {
            "device": current.st_dev,
            "inode": current.st_ino,
            "mode": stat.S_IMODE(current.st_mode),
            "links": current.st_nlink,
            "uid": current.st_uid,
            "gid": current.st_gid,
            "size": current.st_size,
            "mtime_ns": current.st_mtime_ns,
            "ctime_ns": current.st_ctime_ns,
        }
        if identity != current_identity:
            raise RuntimeError(f"{label} changed while hashing")
        _verify_physical_directory_chain(chain, label)
        return {
            "path": str(expected),
            "sha256": digest.hexdigest(),
            "identity": identity,
            "ancestor_chain": [
                {"path": item["path"], "identity": item["identity"]} for item in chain
            ],
        }
    finally:
        os.close(descriptor)
        for item in reversed(chain):
            os.close(item["fd"])


def forbidden_executor_environment_names(environment=None):
    values = os.environ if environment is None else environment
    forbidden = {name for name in FORBIDDEN_EXECUTOR_ENVIRONMENT if name in values}
    for name in values:
        if name in PREFIX_ENVIRONMENT_EXCEPTIONS:
            continue
        if any(
            name.startswith(prefix)
            for prefix in FORBIDDEN_EXECUTOR_ENVIRONMENT_PREFIXES
        ):
            forbidden.add(name)
    return sorted(forbidden)


def capture_numerical_runtime_contract():
    matmul = torch.backends.cuda.matmul
    return canonical_json_document(
        {
            "creation_umask": require_secure_creation_umask(),
            "default_dtype": str(REVIEWED_GET_DEFAULT_DTYPE()),
            "float32_matmul_precision": REVIEWED_GET_FLOAT32_MATMUL_PRECISION(),
            "deterministic_algorithms": (
                REVIEWED_ARE_DETERMINISTIC_ALGORITHMS_ENABLED()
            ),
            "deterministic_warn_only": REVIEWED_IS_DETERMINISTIC_WARN_ONLY_ENABLED(),
            "cuda_matmul_allow_tf32": bool(matmul.allow_tf32),
            "cuda_matmul_allow_fp16_reduced_precision_reduction": bool(
                matmul.allow_fp16_reduced_precision_reduction
            ),
            "cuda_matmul_allow_bf16_reduced_precision_reduction": bool(
                matmul.allow_bf16_reduced_precision_reduction
            ),
            "cudnn_enabled": bool(torch.backends.cudnn.enabled),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "intraop_threads": REVIEWED_GET_NUM_THREADS(),
            "interop_threads": REVIEWED_GET_NUM_INTEROP_THREADS(),
        }
    )


def capture_dependency_manifest():
    modules = (
        ("fcntl", fcntl),
        ("torch", torch),
        ("torch._C", torch_c_module),
        ("torch._weights_only_unpickler", torch_weights_only_module),
        ("torch.nn.functional", torch_functional_module),
        ("torch.nn.modules.linear", torch_linear_module),
        ("torch.nn.modules.module", torch_module_module),
        ("torch.optim.adam", torch_adam_module),
        ("torch.optim.adamw", torch_adamw_module),
        ("torch.optim.optimizer", torch_optimizer_module),
        ("torch.serialization", torch_serialization_module),
        ("tokenizers", tokenizers_module),
        ("tokenizers.tokenizers", tokenizers_native_module),
    )
    files = {}
    portable_receipts = {}
    prefix = Path(sys.prefix).resolve(strict=True)
    for name, module in modules:
        module_path = Path(module.__file__).resolve(strict=True)
        module_spec = getattr(module, "__spec__", None)
        spec_origin = getattr(module_spec, "origin", None)
        if (
            getattr(module, "__name__", None) != name
            or getattr(module_spec, "name", None) != name
            or spec_origin is None
            or Path(spec_origin).resolve(strict=True) != module_path
        ):
            raise ValueError(f"loaded dependency module identity mismatch: {name}")
        if prefix not in module_path.parents:
            raise ValueError(f"loaded dependency escapes pinned Python: {name}")
        descriptor = _capture_executable(module_path, f"loaded dependency {name}")
        files[name] = descriptor
        portable_receipts[name] = {
            "path": descriptor["path"],
            "sha256": descriptor["sha256"],
            "bytes": descriptor["identity"]["size"],
            "mode": descriptor["identity"]["mode"],
        }
    for native_name in ("torch._C", "tokenizers.tokenizers"):
        if Path(files[native_name]["path"]).suffix not in {".so", ".pyd", ".dylib"}:
            raise ValueError(
                f"native dependency is not an extension module: {native_name}"
            )
    return canonical_json_document(
        {
            "schema": RECOVERY_DEPENDENCY_MANIFEST_SCHEMA,
            "files": files,
            "portable_receipts": portable_receipts,
            "manifest_sha256": stable_json_sha256(portable_receipts),
            "identity_manifest_sha256": stable_json_sha256(files),
        }
    )


def reviewed_python_paths(repo_root):
    root = Path(repo_root)
    candidates = (
        root / "train",
        Path(sysconfig.get_path("stdlib")),
        Path(sysconfig.get_config_var("DESTSHARED")),
        Path(sysconfig.get_path("purelib")),
        Path(sysconfig.get_path("platlib")),
    )
    result = []
    for candidate in candidates:
        resolved = candidate.resolve(strict=True)
        value = str(resolved)
        if value not in result:
            result.append(value)
    return result


def capture_executor_runtime_contract(*, repo_root=None):
    root = Path(repo_root or Path(__file__).resolve().parents[1])
    launcher = PINNED_PYTHON_LAUNCHER
    if not root.is_absolute() or root.is_symlink() or root.resolve(strict=True) != root:
        raise ValueError("recovery source root is not an exact physical directory")
    for name, value in EXECUTOR_ENVIRONMENT.items():
        if os.environ.get(name) != value:
            raise ValueError(f"recovery executor environment mismatch: {name}")
    forbidden_environment = forbidden_executor_environment_names()
    if forbidden_environment:
        raise ValueError(
            "forbidden recovery executor environment: "
            + ",".join(forbidden_environment)
        )
    if "PYTHONPATH" in os.environ:
        raise ValueError("recovery isolated startup forbids PYTHONPATH")
    resolved_launcher = launcher.resolve(strict=True)
    resolved_executable = Path(sys.executable).resolve(strict=True)
    if resolved_launcher != resolved_executable:
        raise ValueError("running Python does not resolve from the pinned launcher")
    expected_flags = {
        "dont_write_bytecode": 1,
        "hash_randomization": 1,
        "no_user_site": 1,
        "no_site": 1,
        "ignore_environment": 1,
        "isolated": 1,
        "safe_path": True,
        "optimize": 0,
    }
    observed_flags = {name: getattr(sys.flags, name) for name in expected_flags}
    if observed_flags != expected_flags:
        raise ValueError("recovery Python startup flags mismatch")
    expected_sys_path = reviewed_python_paths(root)
    if list(sys.path) != expected_sys_path:
        raise ValueError("recovery sys.path is not the exact reviewed isolated path")
    if (
        "site" in sys.modules
        or "sitecustomize" in sys.modules
        or "usercustomize" in sys.modules
    ):
        raise ValueError("site startup module loaded in isolated recovery runtime")
    module_paths = validate_loaded_module_paths(root)
    dependency_manifest = capture_dependency_manifest()
    callable_contract = assert_reviewed_callable_exports()
    git_contract = capture_git_repository_contract(root)
    return canonical_json_document(
        {
            "schema": RECOVERY_EXECUTOR_RUNTIME_SCHEMA,
            "source_root": str(root),
            "launcher_path": str(launcher),
            "resolved_executable": _capture_executable(
                resolved_executable, "resolved Python interpreter"
            ),
            "git_executable": _capture_executable(PINNED_GIT, "pinned git"),
            "python": {
                "version": sys.version,
                "implementation": sys.implementation.name,
                "cache_tag": sys.implementation.cache_tag,
                "soabi": sysconfig.get_config_var("SOABI"),
                "executable": sys.executable,
                "sys_path": list(sys.path),
                "flags": observed_flags,
            },
            "packages": {
                "torch": {
                    "version": str(torch.__version__),
                },
                "tokenizers": {
                    "version": str(tokenizers_module.__version__),
                },
            },
            "dependency_manifest": dependency_manifest,
            "callable_contract": callable_contract,
            "git_repository_contract": git_contract,
            "explicit_unbound_runtime_boundary": [
                "host_kernel_image_and_configuration",
                "NVIDIA_kernel_driver_binary_and_firmware",
                "transitive_dynamic_shared_objects_not_named_in_dependency_manifest",
            ],
            "numerical_runtime": capture_numerical_runtime_contract(),
            "module_paths": module_paths,
            "environment": {
                **EXECUTOR_ENVIRONMENT,
            },
            "forbidden_environment": list(FORBIDDEN_EXECUTOR_ENVIRONMENT),
            "forbidden_environment_prefixes": list(
                FORBIDDEN_EXECUTOR_ENVIRONMENT_PREFIXES
            ),
        }
    )


def build_recovery_executor_source_contract(
    recovery_source_commit,
    expected_manifest_sha256,
    *,
    repo_root=None,
):
    root = Path(repo_root or Path(__file__).resolve().parents[1])
    if not re.fullmatch(r"[0-9a-f]{40}", str(recovery_source_commit)):
        raise ValueError("recovery source commit must be lowercase 40-hex")
    if recovery_source_commit == UPSTREAM_SOURCE_COMMIT:
        raise ValueError("recovery executor may not alias the upstream source identity")
    repository = _ConfigExcludedGit(root, copy_index=True)
    try:
        if repository.head_commit != recovery_source_commit:
            raise ValueError("reviewed recovery source commit is not checked out")
        topology = validate_recovery_commit_topology(
            recovery_source_commit,
            repo_root=root,
            repository=repository,
        )
        recovery_tree = repository.tree(recovery_source_commit)
        checkout = validate_checkout_closed_world(root, repository, recovery_tree)
        module_paths = validate_loaded_module_paths(root)
        git_repository_contract = capture_git_repository_contract(root)
        sources = {}
        source_files = {}
        for name in RECOVERY_SOURCE_PATHS:
            mode, oid = recovery_tree.get(name, (None, None))
            if mode != "100644" or not re.fullmatch(r"[0-9a-f]{40}", str(oid)):
                raise ValueError(f"recovery source Git mode mismatch: {name}")
            committed = repository.blob(oid)
            path = root / name
            before = os.stat(path, follow_symlinks=False)
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_ISLNK(before.st_mode)
                or stat.S_IMODE(before.st_mode) != 0o644
                or before.st_nlink != 1
            ):
                raise ValueError(f"recovery source file identity mismatch: {name}")
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                opened = os.fstat(descriptor)
                with os.fdopen(os.dup(descriptor), "rb") as source:
                    working = source.read()
                after = os.stat(path, follow_symlinks=False)
            finally:
                os.close(descriptor)
            if not (
                _stat_identity(before)
                == _stat_identity(opened)
                == _stat_identity(after)
            ):
                raise RuntimeError(f"recovery source changed during binding: {name}")
            if working != committed:
                raise ValueError(
                    f"recovery executor source differs from commit: {name}"
                )
            sources[name] = sha256_bytes(committed)
            source_files[name] = {
                "git_mode": "100644",
                "checkout_identity": _stat_identity(opened),
            }
        manifest = stable_json_sha256(sources)
        if manifest != expected_manifest_sha256:
            raise ValueError("recovery executor source manifest mismatch")
        return {
            "schema": RECOVERY_EXECUTOR_SOURCE_SCHEMA,
            "git_commit": recovery_source_commit,
            **topology,
            "checkout": checkout,
            "sources": sources,
            "source_files": source_files,
            "manifest_sha256": manifest,
            "loaded_module_paths": module_paths,
            "git_repository_contract": git_repository_contract,
        }
    finally:
        repository.close()


def verify_upstream_source_snapshot(plan, *, repo_root=None):
    root = Path(repo_root or Path(__file__).resolve().parents[1])
    expected_contract = {
        "git_commit": UPSTREAM_SOURCE_COMMIT,
        "manifest_sha256": UPSTREAM_SOURCE_MANIFEST_SHA256,
    }
    if not type_strict_equal(plan.get("source_contract"), expected_contract):
        raise ValueError("upstream plan source contract mismatch")
    plan_sources = plan.get("scientific_source_sha256")
    if type(plan_sources) is not dict or set(plan_sources) != set(
        upstream.SCIENTIFIC_SOURCE_PATHS
    ):
        raise ValueError("upstream plan scientific source schema mismatch")
    repository = _ConfigExcludedGit(root, copy_index=False)
    try:
        upstream_tree = repository.tree(UPSTREAM_SOURCE_COMMIT)
        observed = {}
        for name in upstream.SCIENTIFIC_SOURCE_PATHS:
            mode, oid = upstream_tree.get(name, (None, None))
            if mode not in {"100644", "100755"}:
                raise ValueError(f"upstream source Git mode mismatch: {name}")
            committed = repository.blob(oid)
            digest = sha256_bytes(committed)
            if plan_sources[name] != digest:
                raise ValueError(f"upstream plan source hash mismatch: {name}")
            required_mode = 0o755 if mode == "100755" else 0o644
            working, _identity = _preimport_read_regular(
                root / name,
                f"upstream source {name}",
                required_mode=required_mode,
            )
            if working != committed:
                raise ValueError(
                    f"loaded upstream dependency differs from a0c258e: {name}"
                )
            observed[name] = digest
        if stable_json_sha256(observed) != UPSTREAM_SOURCE_MANIFEST_SHA256:
            raise ValueError("upstream scientific source manifest mismatch")
        return expected_contract, observed
    finally:
        repository.close()


def validate_confirmation_generator_contract(plan, source_contract, source_hashes):
    commitment = plan.get("confirmation_commitment")
    document = commitment.get("document") if isinstance(commitment, dict) else None
    generator = (
        document.get("generator_source_contract")
        if isinstance(document, dict)
        else None
    )
    expected_sources = {
        name: source_hashes[name]
        for name in upstream.CANONICAL_CONFIRMATION_GENERATOR_SOURCES
    }
    expected_generator = {
        "schema": upstream.CANONICAL_CONFIRMATION_GENERATOR_SCHEMA,
        "entrypoint": upstream.CANONICAL_CONFIRMATION_GENERATOR_ENTRYPOINT,
        "sources": expected_sources,
        "manifest_sha256": REVIEWED_STABLE_JSON_SHA256(expected_sources),
    }
    if (
        not type_strict_equal(generator, expected_generator)
        or document.get("source_contract") != source_contract
    ):
        raise ValueError("upstream confirmation generator substitution detected")
    return generator


def recovery_root(recovery_source_commit):
    if not re.fullmatch(r"[0-9a-f]{40}", str(recovery_source_commit)):
        raise ValueError("recovery source commit must be lowercase 40-hex")
    if recovery_source_commit == UPSTREAM_SOURCE_COMMIT:
        raise ValueError("recovery root cannot alias the upstream commit")
    return RECOVERY_PARENT / (
        f"upstream_{UPSTREAM_PLAN_SHA256}_executor_{recovery_source_commit}"
    )


def recovery_review_path(recovery_source_commit):
    return REVIEW_PARENT / f"review_{recovery_source_commit}" / "hostile_review.json"


def _layout_identity(observed):
    return {
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "mode": stat.S_IMODE(observed.st_mode),
        "links": observed.st_nlink,
        "uid": observed.st_uid,
        "gid": observed.st_gid,
    }


def _expected_layout_receipt_from_identities(
    root,
    source_contract,
    root_identity,
    directory_identities,
    regular_file_nlink_delta,
):
    return canonical_json_document(
        {
            "audit": RECOVERY_LAYOUT_RECEIPT_AUDIT,
            "root": str(root),
            "root_identity": root_identity,
            "root_reserved_mode": 0o700,
            "root_planned_mode": 0o555,
            "directory_regular_file_nlink_delta": regular_file_nlink_delta,
            "directories": {
                name: {
                    "path": str(root / name),
                    "identity": identity,
                    "reserved_mode": 0o700,
                    "fit_sealed_mode": 0o555 if name == "fit" else 0o700,
                }
                for name, identity in sorted(directory_identities.items())
            },
            "installer_source_binding": _recovery_parent_installer_binding(
                source_contract
            ),
            "required_umask": "0077",
            "claim_boundary": (
                "This receipt reserves directory identities only. It establishes no "
                "plan, fitted trajectory, evaluation, capability, or reasoning claim."
            ),
        }
    )


def _expected_layout_receipt(
    root, source_contract, root_stat, directories, regular_file_nlink_delta
):
    return _expected_layout_receipt_from_identities(
        root,
        source_contract,
        _layout_identity(root_stat),
        {name: _layout_identity(observed) for name, observed in directories.items()},
        regular_file_nlink_delta,
    )


def _require_reserved_layout_identity(identity, label):
    keys = {"device", "inode", "mode", "links", "uid", "gid"}
    if (
        type(identity) is not dict
        or set(identity) != keys
        or any(type(identity[name]) is not int for name in keys)
        or identity["device"] < 0
        or identity["inode"] <= 0
        or identity["mode"] != 0o700
        or identity["links"] < 2
        or identity["uid"] < 0
        or identity["gid"] < 0
    ):
        raise ValueError(f"{label} reserved identity is malformed")


def _matches_reserved_layout_identity(
    current,
    reserved,
    required_mode,
    *,
    regular_file_children=0,
    regular_file_nlink_delta=0,
):
    _require_reserved_layout_identity(reserved, "recovery layout")
    return (
        current.get("device") == reserved["device"]
        and current.get("inode") == reserved["inode"]
        and current.get("links")
        == reserved["links"]
        + int(regular_file_children) * int(regular_file_nlink_delta)
        and current.get("uid") == reserved["uid"]
        and current.get("gid") == reserved["gid"]
        and current.get("mode") == int(required_mode)
    )


def reserve_recovery_layout(root, source_contract, parent_binding):
    """Exclusively reserve root/subdirectory inodes before external review."""
    enforce_secure_creation_umask()
    root = Path(root)
    if root.parent != RECOVERY_PARENT or not root.is_absolute():
        raise ValueError("recovery reservation root is not exact")
    verify_recovery_parent_binding(parent_binding, source_contract)
    parent_chain = _open_physical_directory_chain(root.parent, "recovery parent")
    root_fd = None
    try:
        parent_fd = parent_chain[-1]["fd"]
        if parent_chain[-1]["identity"]["mode"] != 0o700:
            raise ValueError("recovery parent mode changed before reservation")
        try:
            os.mkdir(root.name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            return capture_recovery_layout_binding(
                root, source_contract, phase="reserved"
            )
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        root_fd = os.open(root.name, flags, dir_fd=parent_fd)
        root_stat = os.fstat(root_fd)
        root_initial_links = root_stat.st_nlink
        if (
            stat.S_IMODE(root_stat.st_mode) != 0o700
            or root_stat.st_uid != os.getuid()
            or root_stat.st_gid != os.getgid()
        ):
            raise RuntimeError("recovery reservation root identity mismatch")
        directories = {}
        for name in ("fit", "development_eval", "confirmation_eval"):
            os.mkdir(name, 0o700, dir_fd=root_fd)
            child_fd = os.open(name, flags, dir_fd=root_fd)
            try:
                observed = os.fstat(child_fd)
                linked = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                if (
                    not stat.S_ISDIR(observed.st_mode)
                    or stat.S_ISLNK(linked.st_mode)
                    or _layout_identity(observed) != _layout_identity(linked)
                    or stat.S_IMODE(observed.st_mode) != 0o700
                    or observed.st_uid != os.getuid()
                    or observed.st_gid != os.getgid()
                ):
                    raise RuntimeError(
                        "secure recovery subdirectory reservation failed"
                    )
                directories[name] = observed
            finally:
                os.close(child_fd)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        receipt_fd = os.open(RECOVERY_LAYOUT_RECEIPT_NAME, flags, 0o600, dir_fd=root_fd)
        try:
            root_stat = os.fstat(root_fd)
            regular_file_nlink_delta = (
                root_stat.st_nlink - root_initial_links - len(directories)
            )
            if regular_file_nlink_delta not in {0, 1}:
                raise RuntimeError("unsupported directory link-count semantics")
            receipt = _expected_layout_receipt(
                root,
                source_contract,
                root_stat,
                directories,
                regular_file_nlink_delta,
            )
            payload = canonical_json_receipt_bytes(receipt)
            with os.fdopen(os.dup(receipt_fd), "wb") as sink:
                sink.write(payload)
                sink.flush()
                os.fsync(sink.fileno())
            os.fchmod(receipt_fd, 0o444)
            os.fsync(receipt_fd)
        finally:
            os.close(receipt_fd)
        if sorted(os.listdir(root_fd)) != [
            "confirmation_eval",
            "development_eval",
            "fit",
            RECOVERY_LAYOUT_RECEIPT_NAME,
        ]:
            raise RuntimeError("recovery reservation root is not closed-world")
        os.fsync(root_fd)
        os.fsync(parent_fd)
    finally:
        if root_fd is not None:
            os.close(root_fd)
        for item in reversed(parent_chain):
            os.close(item["fd"])
    return capture_recovery_layout_binding(root, source_contract, phase="reserved")


def capture_recovery_layout_binding(root, source_contract, *, phase, fit_state="empty"):
    root = Path(root)
    if phase not in {"reserved", "planned", "fit"}:
        raise ValueError("unknown recovery layout phase")
    root_mode = 0o700 if phase == "reserved" else 0o555
    expected_children = {
        "fit",
        "development_eval",
        "confirmation_eval",
        RECOVERY_LAYOUT_RECEIPT_NAME,
    }
    if phase != "reserved":
        expected_children.add("recovery_plan.json")
    root_binding = capture_physical_directory_binding(
        root, root_mode, "recovery root", expected_children
    )
    receipt_path = root / RECOVERY_LAYOUT_RECEIPT_NAME
    receipt = BoundFile(
        str(receipt_path),
        receipt_path,
        sha256_file(receipt_path),
        "recovery layout receipt",
        required_mode=0o444,
        required_parent_mode=root_mode,
    )
    try:
        document = load_exact_json(receipt.text(), "recovery layout receipt")
        expected_directory_modes = {
            "fit": 0o555 if phase == "fit" and fit_state == "sealed" else 0o700,
            "development_eval": 0o700,
            "confirmation_eval": 0o700,
        }
        directories = {}
        for name, mode in expected_directory_modes.items():
            if name == "fit" and fit_state == "publisher-residue":
                children = tuple(sorted(item.name for item in (root / name).iterdir()))
            else:
                children = (
                    ("motor.pt",) if name == "fit" and fit_state != "empty" else ()
                )
            directories[name] = capture_physical_directory_binding(
                root / name, mode, f"recovery {name}", children
            )
        root_reserved_identity = document.get("root_identity")
        directory_reserved_identities = {
            name: document.get("directories", {}).get(name, {}).get("identity")
            for name in directories
        }
        regular_file_nlink_delta = document.get("directory_regular_file_nlink_delta")
        if type(
            regular_file_nlink_delta
        ) is not int or regular_file_nlink_delta not in {
            0,
            1,
        }:
            raise ValueError("recovery layout link-count policy is malformed")
        _require_reserved_layout_identity(root_reserved_identity, "recovery root")
        for name, identity in directory_reserved_identities.items():
            _require_reserved_layout_identity(identity, f"recovery {name}")
        expected_document = _expected_layout_receipt_from_identities(
            root,
            source_contract,
            root_reserved_identity,
            directory_reserved_identities,
            regular_file_nlink_delta,
        )
        if not type_strict_equal(document, expected_document) or not (
            _matches_reserved_layout_identity(
                root_binding["descriptor_identity"],
                root_reserved_identity,
                root_mode,
                regular_file_children=0 if phase == "reserved" else 1,
                regular_file_nlink_delta=regular_file_nlink_delta,
            )
        ):
            raise ValueError("recovery layout receipt identity mismatch")
        for name, binding in directories.items():
            frozen = document["directories"][name]
            if frozen.get("path") != str(
                root / name
            ) or not _matches_reserved_layout_identity(
                binding["descriptor_identity"],
                frozen["identity"],
                expected_directory_modes[name],
                regular_file_children=len(binding["children"]),
                regular_file_nlink_delta=regular_file_nlink_delta,
            ):
                raise ValueError(
                    "recovery subdirectory identity differs from reservation"
                )
        receipt.verify()
        return canonical_json_document(
            {
                "schema": RECOVERY_LAYOUT_BINDING_SCHEMA,
                "phase_independent_receipt": {
                    "path": str(receipt.path),
                    "sha256": receipt.sha256,
                    "document": document,
                },
                "root_identity": document["root_identity"],
                "directory_identities": {
                    name: document["directories"][name]["identity"]
                    for name in sorted(directories)
                },
                "ancestor_chain": [
                    {
                        "path": item["path"],
                        "identity": {
                            key: value
                            for key, value in item["identity"].items()
                            if key != "mode"
                        },
                    }
                    for item in root_binding["ancestor_chain"]
                ],
            }
        )
    finally:
        receipt.close()


def _require_directory(path, mode, label, children=None):
    observed = os.lstat(path)
    if (
        not stat.S_ISDIR(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != mode
    ):
        raise ValueError(f"{label} directory identity or mode mismatch")
    if children is not None and {item.name for item in Path(path).iterdir()} != set(
        children
    ):
        raise ValueError(f"{label} directory is not closed-world")


def _stat_identity(observed):
    return {
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "mode": stat.S_IMODE(observed.st_mode),
        "links": observed.st_nlink,
        "uid": observed.st_uid,
        "gid": observed.st_gid,
        "size": observed.st_size,
        "mtime_ns": observed.st_mtime_ns,
        "ctime_ns": observed.st_ctime_ns,
    }


def _capture_directory(path, mode, children, label):
    expected = Path(path)
    before = os.lstat(expected)
    if (
        not stat.S_ISDIR(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or stat.S_IMODE(before.st_mode) != mode
    ):
        raise ValueError(f"{label} directory identity or mode mismatch")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(expected, flags)
    try:
        opened = os.fstat(descriptor)
        observed_children = sorted(os.listdir(descriptor))
        if observed_children != sorted(children):
            raise ValueError(f"{label} directory is not closed-world")
        after = os.stat(expected, follow_symlinks=False)
        if not (
            _stat_identity(before) == _stat_identity(opened) == _stat_identity(after)
        ):
            raise RuntimeError(f"{label} directory changed during capture")
        return {
            "path": str(expected),
            "kind": "directory",
            "identity": _stat_identity(opened),
            "children": observed_children,
        }
    finally:
        os.close(descriptor)


def _capture_custody_file(path, mode, expected_sha256, label):
    expected = Path(path)
    before = os.lstat(expected)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or stat.S_IMODE(before.st_mode) != mode
        or before.st_nlink != 1
    ):
        raise ValueError(f"{label} file identity or mode mismatch")
    descriptor = os.open(expected, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        digest = hashlib.sha256()
        with os.fdopen(os.dup(descriptor), "rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        after = os.stat(expected, follow_symlinks=False)
        if not (
            _stat_identity(before) == _stat_identity(opened) == _stat_identity(after)
        ):
            raise RuntimeError(f"{label} changed during capture")
        if digest.hexdigest() != expected_sha256:
            raise ValueError(f"{label} SHA-256 mismatch")
        return {
            "path": str(expected),
            "kind": "file",
            "identity": _stat_identity(opened),
            "sha256": digest.hexdigest(),
        }
    finally:
        os.close(descriptor)


def _recovery_parent_installer_binding(source_contract):
    if type(source_contract) is not dict:
        raise ValueError("recovery parent installer source contract is not exact")
    commit = source_contract.get("git_commit")
    manifest = source_contract.get("manifest_sha256")
    if (
        source_contract.get("schema") != RECOVERY_EXECUTOR_SOURCE_SCHEMA
        or not re.fullmatch(r"[0-9a-f]{40}", str(commit))
        or commit == UPSTREAM_SOURCE_COMMIT
        or not re.fullmatch(r"[0-9a-f]{64}", str(manifest))
    ):
        raise ValueError("recovery parent installer source contract is invalid")
    return canonical_json_document(
        {
            "schema": RECOVERY_EXECUTOR_SOURCE_SCHEMA,
            "git_commit": commit,
            "manifest_sha256": manifest,
            "source_contract_sha256": stable_json_sha256(source_contract),
        }
    )


def _stable_recovery_parent_identity(observed):
    return {
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "mode": stat.S_IMODE(observed.st_mode),
        "uid": observed.st_uid,
        "gid": observed.st_gid,
    }


def _expected_recovery_parent_receipt(parent_identity, installer_binding):
    return canonical_json_document(
        {
            "audit": RECOVERY_PARENT_RECEIPT_AUDIT,
            "path": str(RECOVERY_PARENT),
            "directory_identity": parent_identity,
            "installer_source_binding": installer_binding,
            "required_umask": "0077",
            "claim_boundary": RECOVERY_PARENT_RECEIPT_CLAIM_BOUNDARY,
        }
    )


def _recovery_parent_receipt_payload(document):
    return canonical_json_receipt_bytes(document)


def _publisher_boot_token():
    linux_boot_id = Path("/proc/sys/kernel/random/boot_id")
    if linux_boot_id.is_file():
        value = _read_kernel_text(linux_boot_id, "kernel boot id").strip().lower()
        if (
            re.fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                value,
            )
            is None
        ):
            raise RuntimeError("kernel boot id is malformed")
        return f"linux-boot-id:{value}"
    root = os.stat("/", follow_symlinks=False)
    material = (
        f"{os.uname().sysname}\0{os.uname().nodename}\0"
        f"{root.st_dev}\0{root.st_ino}\0{root.st_ctime_ns}"
    ).encode("utf-8")
    return "portable-test-token:" + sha256_bytes(material)


def _publisher_process_start_token(pid):
    if type(pid) is not int or pid <= 0:
        raise ValueError("publisher pid is malformed")
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.is_file():
        text = _read_kernel_text(proc_stat, "publisher process stat").strip()
        close = text.rfind(")")
        if close < 0 or not text[:close].startswith(f"{pid} ("):
            raise RuntimeError("publisher process stat is malformed")
        fields = text[close + 2 :].split()
        if len(fields) < 20 or not fields[19].isdigit():
            raise RuntimeError("publisher process start time is malformed")
        return f"linux-start-ticks:{fields[19]}"
    observed = subprocess.check_output(
        ["/bin/ps", "-o", "lstart=", "-p", str(pid)],
        text=True,
        env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    ).strip()
    if not observed:
        raise ProcessLookupError(pid)
    return "portable-test-start:" + sha256_bytes(observed.encode("ascii"))


def _publisher_process_identity():
    return canonical_json_document(
        {
            "pid": os.getpid(),
            "process_start": _publisher_process_start_token(os.getpid()),
            "boot": _publisher_boot_token(),
            "node": os.uname().nodename,
            "uid": os.getuid(),
            "gid": os.getgid(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID") or None,
        }
    )


def _publisher_process_is_live(identity):
    required = {
        "pid",
        "process_start",
        "boot",
        "node",
        "uid",
        "gid",
        "slurm_job_id",
    }
    if type(identity) is not dict or set(identity) != required:
        raise ValueError("publisher process identity is malformed")
    if (
        type(identity["pid"]) is not int
        or identity["pid"] <= 0
        or type(identity["process_start"]) is not str
        or type(identity["boot"]) is not str
        or type(identity["node"]) is not str
        or type(identity["uid"]) is not int
        or type(identity["gid"]) is not int
        or (
            identity["slurm_job_id"] is not None
            and type(identity["slurm_job_id"]) is not str
        )
    ):
        raise ValueError("publisher process identity types are malformed")
    if (
        identity["node"] != os.uname().nodename
        or identity["boot"] != _publisher_boot_token()
    ):
        return False
    try:
        observed = _publisher_process_start_token(identity["pid"])
    except (FileNotFoundError, ProcessLookupError, subprocess.CalledProcessError):
        return False
    return observed == identity["process_start"]


def _publisher_file_identity(observed):
    return canonical_json_document(
        {
            "device": observed.st_dev,
            "inode": observed.st_ino,
            "file_type": stat.S_IFMT(observed.st_mode),
            "uid": observed.st_uid,
            "gid": observed.st_gid,
        }
    )


def _publisher_directory_identity(observed):
    return canonical_json_document(
        {
            "device": observed.st_dev,
            "inode": observed.st_ino,
            "file_type": stat.S_IFMT(observed.st_mode),
            "uid": observed.st_uid,
            "gid": observed.st_gid,
        }
    )


class PublisherOwnership:
    """Exclusive, crash-recoverable ownership of one publication namespace."""

    def __init__(
        self,
        directory_fd,
        directory_path,
        *,
        lock_name,
        purpose,
        target_name,
        stage_name,
    ):
        self.directory_fd = os.dup(directory_fd)
        self.directory_path = Path(directory_path)
        self.lock_name = lock_name
        self.purpose = purpose
        self.target_name = target_name
        self.stage_name = stage_name
        self.lock_fd = -1
        self.created_lock = False
        self.document = None
        self._closed = False
        try:
            self._acquire()
        except Exception:
            if self.created_lock and self.lock_fd >= 0:
                try:
                    linked = os.stat(
                        self.lock_name,
                        dir_fd=self.directory_fd,
                        follow_symlinks=False,
                    )
                    if _stat_identity(linked) == _stat_identity(os.fstat(self.lock_fd)):
                        os.unlink(self.lock_name, dir_fd=self.directory_fd)
                        os.fsync(self.directory_fd)
                except (FileNotFoundError, OSError):
                    pass
            if self.lock_fd >= 0:
                os.close(self.lock_fd)
            os.close(self.directory_fd)
            self.lock_fd = -1
            self.directory_fd = -1
            self._closed = True
            raise

    def _expected_document(self, stage):
        return canonical_json_document(
            {
                "schema": RECOVERY_PUBLISHER_SCHEMA,
                "purpose": self.purpose,
                "lock_name": self.lock_name,
                "target_name": self.target_name,
                "stage_name": self.stage_name,
                "directory": {
                    "path": str(self.directory_path),
                    "identity": _publisher_directory_identity(
                        os.fstat(self.directory_fd)
                    ),
                },
                "owner": _publisher_process_identity(),
                "stage": stage,
            }
        )

    def _read_document(self):
        os.lseek(self.lock_fd, 0, os.SEEK_SET)
        chunks = []
        while True:
            block = os.read(self.lock_fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        payload = b"".join(chunks)
        if not payload:
            return None
        try:
            text = payload.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("publisher ownership journal is not ASCII") from exc
        document = load_exact_json(text, "publisher ownership journal")
        if payload != canonical_json_receipt_bytes(document):
            raise ValueError("publisher ownership journal is not canonical")
        return document

    def _write_document(self, document):
        payload = canonical_json_receipt_bytes(document)
        os.lseek(self.lock_fd, 0, os.SEEK_SET)
        os.ftruncate(self.lock_fd, 0)
        offset = 0
        while offset < len(payload):
            offset += os.write(self.lock_fd, payload[offset:])
        os.fsync(self.lock_fd)
        os.fsync(self.directory_fd)
        self.document = document

    def _validate_document_scope(self, document):
        if type(document) is not dict or set(document) != {
            "schema",
            "purpose",
            "lock_name",
            "target_name",
            "stage_name",
            "directory",
            "owner",
            "stage",
        }:
            raise ValueError("publisher ownership journal schema mismatch")
        expected_directory = {
            "path": str(self.directory_path),
            "identity": _publisher_directory_identity(os.fstat(self.directory_fd)),
        }
        if (
            document["schema"] != RECOVERY_PUBLISHER_SCHEMA
            or document["purpose"] != self.purpose
            or document["lock_name"] != self.lock_name
            or document["target_name"] != self.target_name
            or document["stage_name"] != self.stage_name
            or not type_strict_equal(document["directory"], expected_directory)
        ):
            raise ValueError("publisher ownership journal scope mismatch")
        _publisher_process_is_live(document["owner"])
        stage = document["stage"]
        if stage is not None:
            if type(stage) is not dict or set(stage) != {"name", "identity"}:
                raise ValueError("publisher stage binding is malformed")
            if stage["name"] != self.stage_name or type(stage["identity"]) is not dict:
                raise ValueError("publisher stage binding is out of scope")

    def _recover_stale_stage(self, document):
        self._validate_document_scope(document)
        stage_binding = document["stage"]
        stage_exists = self.stage_name in os.listdir(self.directory_fd)
        if stage_binding is None:
            if stage_exists:
                raise ValueError("unbound publisher stage is foreign content")
            return
        if not stage_exists:
            if self.target_name not in os.listdir(self.directory_fd):
                raise RuntimeError("bound publisher stage disappeared")
            target = os.stat(
                self.target_name, dir_fd=self.directory_fd, follow_symlinks=False
            )
            if (
                not type_strict_equal(
                    _publisher_file_identity(target), stage_binding["identity"]
                )
                or target.st_nlink != 1
                or stat.S_IMODE(target.st_mode) != 0o444
            ):
                raise ValueError("committed publisher target is foreign content")
            return
        stage = os.stat(
            self.stage_name, dir_fd=self.directory_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(stage.st_mode)
            or stat.S_ISLNK(stage.st_mode)
            or not type_strict_equal(
                _publisher_file_identity(stage), stage_binding["identity"]
            )
            or stage.st_uid != os.getuid()
            or stage.st_gid != os.getgid()
            or stat.S_IMODE(stage.st_mode) not in {0o600, 0o444}
        ):
            raise ValueError("publisher stage inode is foreign or substituted")
        target_exists = self.target_name in os.listdir(self.directory_fd)
        if target_exists:
            target = os.stat(
                self.target_name, dir_fd=self.directory_fd, follow_symlinks=False
            )
            if (
                _publisher_file_identity(target) != _publisher_file_identity(stage)
                or target.st_nlink != 2
                or stage.st_nlink != 2
                or stat.S_IMODE(target.st_mode) != 0o444
            ):
                raise ValueError("publisher target is foreign or not an atomic commit")
        elif stage.st_nlink != 1:
            raise ValueError("uncommitted publisher stage has a foreign hard link")
        os.unlink(self.stage_name, dir_fd=self.directory_fd)
        os.fsync(self.directory_fd)
        if target_exists:
            target = os.stat(
                self.target_name, dir_fd=self.directory_fd, follow_symlinks=False
            )
            if target.st_nlink != 1 or stat.S_IMODE(target.st_mode) != 0o444:
                raise RuntimeError("publisher target did not settle after recovery")

    def _acquire(self):
        flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        try:
            self.lock_fd = os.open(
                self.lock_name,
                flags | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=self.directory_fd,
            )
            self.created_lock = True
        except FileExistsError:
            self.lock_fd = os.open(
                self.lock_name, flags, 0o600, dir_fd=self.directory_fd
            )
        linked = os.stat(
            self.lock_name, dir_fd=self.directory_fd, follow_symlinks=False
        )
        opened = os.fstat(self.lock_fd)
        if (
            _stat_identity(linked) != _stat_identity(opened)
            or not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_nlink != 1
            or opened.st_uid != os.getuid()
            or opened.st_gid != os.getgid()
        ):
            raise ValueError("publisher ownership file identity mismatch")
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another live publisher owns this namespace") from exc
        prior = self._read_document()
        if prior is not None:
            self._validate_document_scope(prior)
            if _publisher_process_is_live(prior["owner"]):
                raise RuntimeError("publisher journal names a live process")
            self._recover_stale_stage(prior)
        elif self.stage_name in os.listdir(self.directory_fd):
            raise ValueError("unowned publisher stage is foreign content")
        self._write_document(self._expected_document(None))

    def bind_stage(self, stage_fd):
        if self._closed or self.document["stage"] is not None:
            raise RuntimeError("publisher ownership cannot bind another stage")
        linked = os.stat(
            self.stage_name, dir_fd=self.directory_fd, follow_symlinks=False
        )
        opened = os.fstat(stage_fd)
        if (
            _stat_identity(linked) != _stat_identity(opened)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_uid != os.getuid()
            or opened.st_gid != os.getgid()
        ):
            raise RuntimeError("publisher stage creation was not exclusive")
        stage = {
            "name": self.stage_name,
            "identity": _publisher_file_identity(opened),
        }
        self._write_document(self._expected_document(stage))

    def clear_stage_binding(self):
        if self._closed or self.stage_name in os.listdir(self.directory_fd):
            raise RuntimeError("publisher stage still exists")
        self._write_document(self._expected_document(None))

    def close(self, *, preserve=False):
        if self._closed:
            return
        try:
            if not preserve:
                if self.stage_name in os.listdir(self.directory_fd):
                    raise RuntimeError("cannot release ownership with a live stage")
                linked = os.stat(
                    self.lock_name, dir_fd=self.directory_fd, follow_symlinks=False
                )
                if _stat_identity(linked) != _stat_identity(os.fstat(self.lock_fd)):
                    raise RuntimeError("publisher ownership file was retargeted")
                os.unlink(self.lock_name, dir_fd=self.directory_fd)
                os.fsync(self.directory_fd)
            fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
        finally:
            if self.lock_fd >= 0:
                os.close(self.lock_fd)
            os.close(self.directory_fd)
            self.lock_fd = -1
            self.directory_fd = -1
            self._closed = True


def _capture_recovery_parent_receipt(expected_document):
    path = RECOVERY_PARENT / RECOVERY_PARENT_RECEIPT_NAME
    payload = _recovery_parent_receipt_payload(expected_document)
    before = os.lstat(path)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o444
        or before.st_nlink != 1
        or before.st_uid != os.getuid()
        or before.st_gid != os.getgid()
    ):
        raise ValueError("recovery parent receipt file identity or mode mismatch")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if _stat_identity(opened) != _stat_identity(before):
            raise RuntimeError("recovery parent receipt changed during binding")
        with os.fdopen(os.dup(descriptor), "rb") as source:
            observed_payload = source.read()
        linked = os.stat(path, follow_symlinks=False)
        if _stat_identity(linked) != _stat_identity(opened):
            raise RuntimeError("recovery parent receipt changed while reading")
        if observed_payload != payload:
            raise ValueError("recovery parent receipt bytes mismatch")
        observed_document = load_exact_json(
            observed_payload.decode("ascii"), "recovery parent receipt"
        )
        if not type_strict_equal(observed_document, expected_document):
            raise ValueError("recovery parent receipt document mismatch")
        return {
            "path": str(path),
            "kind": "file",
            "identity": _stat_identity(opened),
            "sha256": sha256_bytes(observed_payload),
            "document": observed_document,
        }
    finally:
        os.close(descriptor)


def _publish_recovery_parent_receipt(
    parent_fd, expected_document, *, allow_install, ownership=None
):
    children = sorted(
        name
        for name in os.listdir(parent_fd)
        if name not in {RECOVERY_PARENT_OWNER_NAME, RECOVERY_PARENT_STAGE_NAME}
    )
    receipt_name = RECOVERY_PARENT_RECEIPT_NAME
    if receipt_name in children:
        return _capture_recovery_parent_receipt(expected_document)
    if children:
        raise ValueError("unreceipted recovery parent is not empty")
    if not allow_install:
        raise ValueError("recovery parent receipt is missing")
    if ownership is None:
        raise RuntimeError("recovery parent receipt installation lacks ownership")

    payload = _recovery_parent_receipt_payload(expected_document)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    receipt_fd = os.open(RECOVERY_PARENT_STAGE_NAME, flags, 0o600, dir_fd=parent_fd)
    try:
        created = os.fstat(receipt_fd)
        if (
            not stat.S_ISREG(created.st_mode)
            or created.st_nlink != 1
            or stat.S_IMODE(created.st_mode) != 0o600
            or created.st_uid != os.getuid()
            or created.st_gid != os.getgid()
        ):
            raise RuntimeError("secure recovery parent receipt creation failed")
        ownership.bind_stage(receipt_fd)
        with os.fdopen(os.dup(receipt_fd), "wb") as sink:
            sink.write(payload)
            sink.flush()
            os.fsync(sink.fileno())
        linked = os.stat(
            RECOVERY_PARENT_STAGE_NAME, dir_fd=parent_fd, follow_symlinks=False
        )
        if _stat_identity(linked) != _stat_identity(os.fstat(receipt_fd)):
            raise RuntimeError("recovery parent receipt changed before publication")
        os.fchmod(receipt_fd, 0o444)
        os.fsync(receipt_fd)
        os.link(
            RECOVERY_PARENT_STAGE_NAME,
            receipt_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        committed = os.stat(receipt_name, dir_fd=parent_fd, follow_symlinks=False)
        staged = os.stat(
            RECOVERY_PARENT_STAGE_NAME, dir_fd=parent_fd, follow_symlinks=False
        )
        if (
            _publisher_file_identity(committed) != _publisher_file_identity(staged)
            or committed.st_nlink != 2
            or stat.S_IMODE(committed.st_mode) != 0o444
        ):
            raise RuntimeError("recovery parent receipt atomic commit failed")
        os.fsync(parent_fd)
        os.unlink(RECOVERY_PARENT_STAGE_NAME, dir_fd=parent_fd)
        os.fsync(parent_fd)
        ownership.clear_stage_binding()
    finally:
        os.close(receipt_fd)
    return _capture_recovery_parent_receipt(expected_document)


def _validate_recovery_parent_children(parent_fd):
    root_pattern = re.compile(r"upstream_[0-9a-f]{64}_executor_[0-9a-f]{40}")
    for name in os.listdir(parent_fd):
        if name == RECOVERY_PARENT_RECEIPT_NAME:
            continue
        if name == RECOVERY_PARENT_OWNER_NAME:
            observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(observed.st_mode)
                or stat.S_ISLNK(observed.st_mode)
                or stat.S_IMODE(observed.st_mode) != 0o600
                or observed.st_nlink != 1
                or observed.st_uid != os.getuid()
                or observed.st_gid != os.getgid()
            ):
                raise ValueError("recovery parent ownership identity mismatch")
            continue
        if name == RECOVERY_PARENT_STAGE_NAME:
            raise ValueError("recovery parent contains an unowned receipt stage")
        if not root_pattern.fullmatch(name):
            raise ValueError("recovery parent contains an unrecognized child")
        observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        mode = stat.S_IMODE(observed.st_mode)
        if (
            not stat.S_ISDIR(observed.st_mode)
            or stat.S_ISLNK(observed.st_mode)
            or mode not in {0o700, 0o555}
            or observed.st_uid != os.getuid()
            or observed.st_gid != os.getgid()
        ):
            raise ValueError("recovery parent child identity or mode mismatch")
        child_fd = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            children = set(os.listdir(child_fd))
        finally:
            os.close(child_fd)
        expected = {
            RECOVERY_LAYOUT_RECEIPT_NAME,
            "fit",
            "development_eval",
            "confirmation_eval",
        }
        if mode == 0o555:
            expected.add("recovery_plan.json")
        if children != expected:
            raise ValueError(
                "recovery parent child is not a recognized recovery layout"
            )


def ensure_recovery_parent(source_contract, *, allow_install):
    """Require or durably install the exact receipted recovery parent."""
    enforce_secure_creation_umask()
    parent = RECOVERY_PARENT
    container = parent.parent
    if not parent.is_absolute() or container.resolve(strict=True) != container:
        raise ValueError("recovery parent container is aliased or not physical")
    container_before = os.lstat(container)
    if not stat.S_ISDIR(container_before.st_mode) or stat.S_ISLNK(
        container_before.st_mode
    ):
        raise ValueError("recovery parent container is not a regular directory")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    container_fd = os.open(container, directory_flags)
    try:
        container_opened = os.fstat(container_fd)
        if (container_opened.st_dev, container_opened.st_ino) != (
            container_before.st_dev,
            container_before.st_ino,
        ):
            raise RuntimeError("recovery parent container changed during binding")
        try:
            linked_parent = os.stat(
                parent.name, dir_fd=container_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            if not allow_install:
                raise ValueError("recovery parent is missing") from None
            os.mkdir(parent.name, mode=0o700, dir_fd=container_fd)
            os.fsync(container_fd)
            linked_parent = os.stat(
                parent.name, dir_fd=container_fd, follow_symlinks=False
            )
        if (
            not stat.S_ISDIR(linked_parent.st_mode)
            or stat.S_ISLNK(linked_parent.st_mode)
            or stat.S_IMODE(linked_parent.st_mode) != 0o700
            or linked_parent.st_uid != os.getuid()
            or linked_parent.st_gid != os.getgid()
        ):
            raise ValueError("recovery parent identity, owner, or mode mismatch")
        parent_fd = os.open(parent.name, directory_flags, dir_fd=container_fd)
        try:
            parent_opened = os.fstat(parent_fd)
            if _stable_recovery_parent_identity(parent_opened) != (
                _stable_recovery_parent_identity(linked_parent)
            ):
                raise RuntimeError("recovery parent changed during binding")
            if parent.resolve(strict=True) != parent:
                raise ValueError("recovery parent is an aliased physical path")
            parent_identity = _stable_recovery_parent_identity(parent_opened)
            installer_binding = _recovery_parent_installer_binding(source_contract)
            expected_document = _expected_recovery_parent_receipt(
                parent_identity, installer_binding
            )
            publisher_names = {
                RECOVERY_PARENT_OWNER_NAME,
                RECOVERY_PARENT_STAGE_NAME,
            }.intersection(os.listdir(parent_fd))
            if (
                RECOVERY_PARENT_RECEIPT_NAME in os.listdir(parent_fd)
                and not publisher_names
            ):
                receipt = _publish_recovery_parent_receipt(
                    parent_fd,
                    expected_document,
                    allow_install=False,
                )
            else:
                if not allow_install:
                    raise ValueError(
                        "recovery parent receipt publication is interrupted"
                    )
                ownership = PublisherOwnership(
                    parent_fd,
                    parent,
                    lock_name=RECOVERY_PARENT_OWNER_NAME,
                    purpose="recovery-parent-receipt",
                    target_name=RECOVERY_PARENT_RECEIPT_NAME,
                    stage_name=RECOVERY_PARENT_STAGE_NAME,
                )
                try:
                    receipt = _publish_recovery_parent_receipt(
                        parent_fd,
                        expected_document,
                        allow_install=True,
                        ownership=ownership,
                    )
                    ownership.close()
                except Exception:
                    preserve = bool(
                        ownership.document
                        and ownership.document.get("stage") is not None
                    )
                    ownership.close(preserve=preserve)
                    raise
            _validate_recovery_parent_children(parent_fd)
            parent_after = os.stat(
                parent.name, dir_fd=container_fd, follow_symlinks=False
            )
            if _stable_recovery_parent_identity(parent_after) != parent_identity:
                raise RuntimeError("recovery parent changed after receipt validation")
            os.fsync(parent_fd)
            os.fsync(container_fd)
            ancestor_binding = capture_physical_directory_binding(
                parent, 0o700, "recovery parent"
            )
            return canonical_json_document(
                {
                    "schema": RECOVERY_PARENT_BINDING_SCHEMA,
                    "path": str(parent),
                    "identity": parent_identity,
                    "ancestor_chain": ancestor_binding["ancestor_chain"],
                    "receipt": receipt,
                }
            )
        finally:
            os.close(parent_fd)
    finally:
        os.close(container_fd)


def verify_recovery_parent_binding(expected, source_contract):
    observed = ensure_recovery_parent(source_contract, allow_install=False)
    if not type_strict_equal(observed, expected):
        raise RuntimeError("recovery parent or durable receipt changed after binding")
    return observed


def capture_upstream_custody_snapshot():
    root_children = {
        "plan.json",
        "fit",
        "development_eval",
        "confirmation_eval",
        *(f"shard_{index:02d}" for index in range(upstream.CANONICAL_FEATURE_SHARDS)),
    }
    entries = [
        _capture_directory(
            UPSTREAM_ROOT, 0o555, root_children, "upstream canonical root"
        ),
        _capture_custody_file(
            UPSTREAM_PLAN_PATH,
            0o444,
            UPSTREAM_PLAN_SHA256,
            "upstream canonical plan",
        ),
    ]
    for name in ("fit", "development_eval", "confirmation_eval"):
        entries.append(
            _capture_directory(UPSTREAM_ROOT / name, 0o700, (), f"upstream {name}")
        )
    for index, expected_sha256 in enumerate(UPSTREAM_SHARD_SHA256):
        directory = UPSTREAM_ROOT / f"shard_{index:02d}"
        entries.append(
            _capture_directory(
                directory, 0o555, ("features.pt",), f"upstream shard {index}"
            )
        )
        entries.append(
            _capture_custody_file(
                directory / "features.pt",
                0o444,
                expected_sha256,
                f"upstream shard {index} artifact",
            )
        )
    entries.extend(
        (
            _capture_directory(
                UPSTREAM_CONFIRMATION_PATH.parent,
                0o555,
                ("commitment.json",),
                "upstream confirmation commitment",
            ),
            _capture_custody_file(
                UPSTREAM_CONFIRMATION_PATH,
                0o444,
                UPSTREAM_CONFIRMATION_COMMITMENT_SHA256,
                "upstream confirmation commitment",
            ),
        )
    )
    return canonical_json_document(
        {"schema": UPSTREAM_CUSTODY_SCHEMA, "entries": entries}
    )


def assert_upstream_custody_unchanged(expected, phase):
    observed = capture_upstream_custody_snapshot()
    if not type_strict_equal(observed, expected):
        raise RuntimeError(f"upstream custody changed {phase}")
    return observed


def validate_upstream_layout():
    return capture_upstream_custody_snapshot()


def load_upstream_plan(*, repo_root=None):
    upstream_custody_snapshot = validate_upstream_layout()
    plan_bound = BoundFile(
        str(UPSTREAM_PLAN_PATH),
        UPSTREAM_PLAN_PATH,
        UPSTREAM_PLAN_SHA256,
        "upstream canonical plan",
        required_mode=0o444,
        required_parent_mode=0o555,
    )
    _require_directory(
        UPSTREAM_CONFIRMATION_PATH.parent,
        0o555,
        "upstream confirmation commitment",
        ("commitment.json",),
    )
    confirmation_bound = BoundFile(
        str(UPSTREAM_CONFIRMATION_PATH),
        UPSTREAM_CONFIRMATION_PATH,
        UPSTREAM_CONFIRMATION_COMMITMENT_SHA256,
        "upstream confirmation commitment",
        required_mode=0o444,
        required_parent_mode=0o555,
    )
    try:
        plan = load_exact_json(plan_bound.text(), "upstream canonical plan")
        if (
            plan.get("audit") != upstream.CANONICAL_PLAN_AUDIT
            or plan.get("canonical") is not True
            or plan.get("plan_path") != str(UPSTREAM_PLAN_PATH)
        ):
            raise ValueError("upstream canonical plan header mismatch")
        source_contract, source_hashes = verify_upstream_source_snapshot(
            plan, repo_root=repo_root
        )
        if plan.get("board_rows_sha256") != UPSTREAM_BOARD_ROWS_SHA256:
            raise ValueError("upstream board-row identity mismatch")
        if stable_json_sha256(plan.get("board")) != UPSTREAM_CANONICAL_BOARD_SHA256:
            raise ValueError("upstream canonical board hash mismatch")
        commitment = plan.get("confirmation_commitment")
        if (
            type(commitment) is not dict
            or commitment.get("path") != str(UPSTREAM_CONFIRMATION_PATH)
            or commitment.get("sha256") != UPSTREAM_CONFIRMATION_COMMITMENT_SHA256
        ):
            raise ValueError("upstream confirmation commitment binding mismatch")
        commitment_document = load_exact_json(
            confirmation_bound.text(), "upstream confirmation commitment"
        )
        if not type_strict_equal(commitment.get("document"), commitment_document):
            raise ValueError("upstream confirmation commitment bytes differ from plan")
        validate_confirmation_generator_contract(plan, source_contract, source_hashes)
        expected_fit = {
            "seed": upstream.FIT_SEED,
            "rank": upstream.RANK,
            "quota": upstream.FIT_QUOTA,
            "updates": upstream.CANONICAL_UPDATES,
            "batch_size": upstream.CANONICAL_BATCH,
            "lr": upstream.CANONICAL_LR,
            "weight_decay": upstream.CANONICAL_WEIGHT_DECAY,
        }
        for name, value in expected_fit.items():
            if not type_strict_equal(plan.get("fit_budget", {}).get(name), value):
                raise ValueError(f"upstream fit budget changed: {name}")
        if (
            plan.get("shard_count") != upstream.CANONICAL_FEATURE_SHARDS
            or len(plan.get("shards", ())) != upstream.CANONICAL_FEATURE_SHARDS
            or plan.get("fit_artifact") != str(UPSTREAM_ROOT / "fit" / "motor.pt")
        ):
            raise ValueError("upstream plan shard or output contract mismatch")
        plan_bound.verify()
        confirmation_bound.verify()
        return (
            plan_bound,
            confirmation_bound,
            plan,
            source_contract,
            source_hashes,
            upstream_custody_snapshot,
        )
    except Exception:
        plan_bound.close()
        confirmation_bound.close()
        raise


_ED25519_Q = 2**255 - 19
_ED25519_L = 2**252 + 27742317777372353535851937790883648493
_ED25519_D = (-121665 * pow(121666, _ED25519_Q - 2, _ED25519_Q)) % _ED25519_Q
_ED25519_I = pow(2, (_ED25519_Q - 1) // 4, _ED25519_Q)
_ED25519_IDENTITY = (0, 1, 1, 0)


def _ed25519_recover_x(y, sign):
    xx = (
        (y * y - 1) * pow(_ED25519_D * y * y + 1, _ED25519_Q - 2, _ED25519_Q)
    ) % _ED25519_Q
    x = pow(xx, (_ED25519_Q + 3) // 8, _ED25519_Q)
    if (x * x - xx) % _ED25519_Q:
        x = (x * _ED25519_I) % _ED25519_Q
    if (x * x - xx) % _ED25519_Q:
        raise ValueError("Ed25519 point is not on the curve")
    if (x & 1) != sign:
        x = _ED25519_Q - x
    if x == 0 and sign:
        raise ValueError("Ed25519 point has non-canonical sign")
    return x


def _ed25519_decode(encoded):
    if type(encoded) is not bytes or len(encoded) != 32:
        raise ValueError("Ed25519 point must be exactly 32 bytes")
    raw = int.from_bytes(encoded, "little")
    sign = raw >> 255
    y = raw & ((1 << 255) - 1)
    if y >= _ED25519_Q:
        raise ValueError("Ed25519 point encoding is non-canonical")
    x = _ed25519_recover_x(y, sign)
    point = (x, y, 1, (x * y) % _ED25519_Q)
    if _ed25519_encode(point) != encoded:
        raise ValueError("Ed25519 point failed canonical round trip")
    return point


def _ed25519_add(left, right):
    x1, y1, z1, t1 = left
    x2, y2, z2, t2 = right
    a = ((y1 - x1) * (y2 - x2)) % _ED25519_Q
    b = ((y1 + x1) * (y2 + x2)) % _ED25519_Q
    c = (2 * _ED25519_D * t1 * t2) % _ED25519_Q
    d = (2 * z1 * z2) % _ED25519_Q
    e = (b - a) % _ED25519_Q
    f = (d - c) % _ED25519_Q
    g = (d + c) % _ED25519_Q
    h = (b + a) % _ED25519_Q
    return (
        e * f % _ED25519_Q,
        g * h % _ED25519_Q,
        f * g % _ED25519_Q,
        e * h % _ED25519_Q,
    )


def _ed25519_scalar_multiply(point, scalar):
    result = _ED25519_IDENTITY
    addend = point
    value = int(scalar)
    while value:
        if value & 1:
            result = _ed25519_add(result, addend)
        addend = _ed25519_add(addend, addend)
        value >>= 1
    return result


def _ed25519_encode(point):
    x, y, z, _t = point
    inverse = pow(z, _ED25519_Q - 2, _ED25519_Q)
    affine_x = x * inverse % _ED25519_Q
    affine_y = y * inverse % _ED25519_Q
    return int(affine_y | ((affine_x & 1) << 255)).to_bytes(32, "little")


_ED25519_BASE_Y = 4 * pow(5, _ED25519_Q - 2, _ED25519_Q) % _ED25519_Q
_ED25519_BASE_X = _ed25519_recover_x(_ED25519_BASE_Y, 0)
_ED25519_BASE = (
    _ED25519_BASE_X,
    _ED25519_BASE_Y,
    1,
    _ED25519_BASE_X * _ED25519_BASE_Y % _ED25519_Q,
)


def verify_ed25519_signature(public_key, message, signature):
    """Strict RFC 8032 verification without an in-repository signing dependency."""
    if type(public_key) is not bytes or len(public_key) != 32:
        raise ValueError("review public key must be exactly 32 bytes")
    if (
        type(message) is not bytes
        or type(signature) is not bytes
        or len(signature) != 64
    ):
        raise ValueError("review signature encoding is invalid")
    public_point = _ed25519_decode(public_key)
    r_encoded, s_encoded = signature[:32], signature[32:]
    r_point = _ed25519_decode(r_encoded)
    scalar = int.from_bytes(s_encoded, "little")
    if scalar >= _ED25519_L:
        raise ValueError("review signature scalar is non-canonical")
    if (
        _ed25519_encode(_ed25519_scalar_multiply(public_point, _ED25519_L))
        != _ed25519_encode(_ED25519_IDENTITY)
        or _ed25519_encode(_ed25519_scalar_multiply(r_point, _ED25519_L))
        != _ed25519_encode(_ED25519_IDENTITY)
        or _ed25519_encode(public_point) == _ed25519_encode(_ED25519_IDENTITY)
    ):
        raise ValueError("review signature uses a non-prime-order point")
    challenge = (
        int.from_bytes(
            hashlib.sha512(r_encoded + public_key + message).digest(), "little"
        )
        % _ED25519_L
    )
    left = _ed25519_scalar_multiply(_ED25519_BASE, scalar)
    right = _ed25519_add(r_point, _ed25519_scalar_multiply(public_point, challenge))
    if _ed25519_encode(left) != _ed25519_encode(right):
        raise ValueError("hostile review Ed25519 signature is invalid")
    return True


def expected_review_statement(
    recovery_source_commit,
    recovery_source_contract,
    executor_runtime_contract,
    recovery_layout_binding,
):
    root = recovery_root(recovery_source_commit)
    return canonical_json_document(
        {
            "audit": RECOVERY_REVIEW_STATEMENT_AUDIT,
            "decision": "GO",
            "signer": {
                "algorithm": "Ed25519",
                "key_id": PRODUCTION_REVIEW_KEY_ID,
                "sequence": REVIEW_SIGNER_SEQUENCE,
            },
            "recovery_executor_source_contract": recovery_source_contract,
            "executor_runtime_contract": executor_runtime_contract,
            "upstream_plan_sha256": UPSTREAM_PLAN_SHA256,
            "normalization_contract": normalization_contract(),
            "allowed_transformation": ALLOWED_TRANSFORMATION,
            "slurm_h100_contract": EXPECTED_SLURM_REQUEST,
            "output_contract": {
                "root": str(root),
                "fit_artifact": str(root / "fit" / "motor.pt"),
                "layout_binding": recovery_layout_binding,
                "upstream_root_must_remain_untouched": str(UPSTREAM_ROOT),
            },
            "fit_claim_boundary": RECOVERY_FIT_CLAIM_BOUNDARY,
            "review_claim_boundary": REVIEW_CLAIM_BOUNDARY,
            "review_trust_boundary": REVIEW_TRUST_BOUNDARY,
        }
    )


def load_hostile_review(
    recovery_source_commit,
    recovery_source_contract,
    executor_runtime_contract,
    recovery_layout_binding,
    review_path,
    review_sha256,
):
    expected_path = recovery_review_path(recovery_source_commit)
    _require_directory(
        expected_path.parent,
        0o555,
        "hostile review receipt",
        ("hostile_review.json",),
    )
    bound = BoundFile(
        review_path,
        expected_path,
        review_sha256,
        "hostile review receipt",
        required_mode=0o444,
        required_parent_mode=0o555,
    )
    try:
        review_bytes = bound.bytes()
        try:
            review_text = review_bytes.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("hostile review receipt is not canonical ASCII") from exc
        review = load_exact_json(review_text, "hostile review receipt")
        if type(review) is not dict or set(review) != {
            "audit",
            "algorithm",
            "key_id",
            "signed_payload",
            "signature_base64",
        }:
            raise ValueError("hostile review receipt schema mismatch")
        expected_statement = expected_review_statement(
            recovery_source_commit,
            recovery_source_contract,
            executor_runtime_contract,
            recovery_layout_binding,
        )
        if (
            review["audit"] != RECOVERY_REVIEW_AUDIT
            or review["algorithm"] != "Ed25519"
            or review["key_id"] != PRODUCTION_REVIEW_KEY_ID
            or not type_strict_equal(review["signed_payload"], expected_statement)
        ):
            raise ValueError("hostile review receipt does not authorize exact recovery")
        signed_payload_bytes = canonical_json_payload(expected_statement).encode(
            "ascii"
        )
        if (
            canonical_json_payload(review["signed_payload"]).encode("ascii")
            != signed_payload_bytes
        ):
            raise ValueError("hostile review signed envelope is not canonical")
        if review_bytes != canonical_json_receipt_bytes(review):
            raise ValueError("hostile review outer receipt encoding is not canonical")
        signature = decode_canonical_base64(
            review["signature_base64"], 64, "hostile review signature"
        )
        verify_ed25519_signature(
            bytes.fromhex(PRODUCTION_REVIEW_PUBLIC_KEY_HEX),
            signed_payload_bytes,
            signature,
        )
        bound.verify()
        return bound, review
    except Exception:
        bound.close()
        raise


def _bind_frozen_inputs(plan):
    bounds = {}
    try:
        for name in ("checkpoint", "tokenizer", "episodes", "cycle"):
            descriptor = plan["frozen_inputs"][name]
            bounds[name] = BoundFile(
                descriptor["path"],
                Path(descriptor["path"]),
                descriptor["sha256"],
                f"upstream frozen {name}",
            )
        return bounds
    except Exception:
        for bound in bounds.values():
            bound.close()
        raise


def reconstruct_board(plan, frozen_bounds):
    assert_reviewed_callable_exports()
    tokenizer = REVIEWED_TOKENIZER_FROM_STR(frozen_bounds["tokenizer"].text())
    rows, generated_board = REVIEWED_GENERATE_FIT_ROWS(
        tokenizer,
        frozen_bounds["episodes"].text(),
        upstream.FIT_SEED,
        upstream.FIT_QUOTA,
    )
    proof, normalized_board = build_normalization_proof(
        generated_board, plan["board"], rows
    )
    if normalized_board != plan["board"]:
        raise ValueError("normalized board differs from upstream plan")
    if REVIEWED_STABLE_JSON_SHA256([row["prefix_sha256"] for row in rows]) != plan.get(
        "extraction_order_sha256"
    ):
        raise ValueError("upstream row order changed")
    control_labels, control = REVIEWED_PERMUTED_CONTROL_LABELS(rows)
    if not type_strict_equal(control, plan["fit_budget"]["control"]):
        raise ValueError("upstream shuffled control changed")
    _schedule, schedule_sha256 = REVIEWED_BATCH_SCHEDULE(
        len(rows),
        plan["fit_budget"]["batch_size"],
        plan["fit_budget"]["updates"],
        plan["fit_budget"]["seed"],
    )
    if schedule_sha256 != plan["fit_budget"]["schedule_sha256"]:
        raise ValueError("upstream fit schedule changed")
    initial_state, initial_sha256 = REVIEWED_INITIAL_MOTOR_STATE(plan["d_model"])
    if initial_sha256 != plan["fit_budget"]["initial_state_sha256"]:
        raise ValueError("upstream initial motor state changed")
    for bound in frozen_bounds.values():
        bound.verify()
    return {
        "tokenizer": tokenizer,
        "rows": rows,
        "normalization_proof": proof,
        "normalized_board": normalized_board,
        "control_labels": control_labels,
        "control": control,
        "initial_state": initial_state,
        "initial_state_sha256": initial_sha256,
    }


def bind_and_merge_upstream_shards(plan, rows, source_contract, source_hashes):
    bounds = []
    payloads = []
    try:
        for index, (descriptor, expected_sha256) in enumerate(
            zip(plan["shards"], UPSTREAM_SHARD_SHA256)
        ):
            expected_path = UPSTREAM_ROOT / f"shard_{index:02d}" / "features.pt"
            if descriptor.get("artifact") != str(expected_path):
                raise ValueError(f"upstream shard {index} path substitution")
            bound = BoundFile(
                descriptor["artifact"],
                expected_path,
                expected_sha256,
                f"upstream shard {index}",
                required_mode=0o444,
                required_parent_mode=0o555,
            )
            bounds.append(bound)
        for bound in bounds:
            payloads.append((bound.sha256, str(bound.path), safe_torch_load(bound)))
        expected_bindings = {
            "base_checkpoint_sha256": plan["frozen_inputs"]["checkpoint"]["sha256"],
            "tokenizer_sha256": plan["frozen_inputs"]["tokenizer"]["sha256"],
            "episodes_sha256": plan["frozen_inputs"]["episodes"]["sha256"],
            "cycle_sha256": plan["frozen_inputs"]["cycle"]["sha256"],
            "confirmation_commitment_sha256": UPSTREAM_CONFIRMATION_COMMITMENT_SHA256,
            "scientific_source_sha256": source_hashes,
        }
        assert_reviewed_callable_exports()
        features, merge = REVIEWED_MERGE_FEATURE_SHARDS(
            payloads,
            rows,
            expected_bindings,
            source_contract,
            plan,
            UPSTREAM_PLAN_SHA256,
        )
        for bound in bounds:
            bound.verify()
        receipts = []
        for index, (bound, descriptor, payload) in enumerate(
            zip(bounds, plan["shards"], (item[2] for item in payloads))
        ):
            receipts.append(
                {
                    "shard_index": index,
                    "path": str(bound.path),
                    "sha256": bound.sha256,
                    "bytes": bound.identity["size"],
                    "descriptor": descriptor,
                    "feature_payload_sha256": payload["feature_payload_sha256"],
                    "sentinel_payload_sha256": payload["sentinel_payload_sha256"],
                }
            )
        return bounds, features, merge, receipts, expected_bindings
    except Exception:
        for bound in bounds:
            bound.close()
        raise


def _review_binding(bound, review):
    return {"path": str(bound.path), "sha256": bound.sha256, "document": review}


def build_recovery_plan_document(
    recovery_source_commit,
    recovery_source_contract,
    executor_runtime_contract,
    recovery_parent_binding,
    recovery_layout_binding,
    review_bound,
    review,
    upstream_plan,
    upstream_source_contract,
    upstream_source_hashes,
    upstream_custody_snapshot,
    context,
    feature_merge,
    shard_receipts,
):
    root = recovery_root(recovery_source_commit)
    plan_path = root / "recovery_plan.json"
    generator_contract = upstream_plan["confirmation_commitment"]["document"][
        "generator_source_contract"
    ]
    deployed_vocabulary = deployment_vocabulary_binding(
        context["tokenizer"],
        upstream_plan,
        feature_merge["deployment_logit_dtype"],
    )
    parameter_ledger = deployment_parameter_ledger(
        dict(EXPECTED_BASE_PARAMETER_CONFIG), upstream_plan["fit_budget"]["rank"]
    )
    document = {
        "audit": RECOVERY_PLAN_AUDIT,
        "recovery": True,
        "recovery_plan_path": str(plan_path),
        "recovery_executor_source_contract": recovery_source_contract,
        "executor_runtime_contract": executor_runtime_contract,
        "recovery_parent_binding": recovery_parent_binding,
        "hostile_review_binding": _review_binding(review_bound, review),
        "upstream_protocol": {
            "source_contract": upstream_source_contract,
            "scientific_source_sha256": upstream_source_hashes,
            "custody_snapshot": upstream_custody_snapshot,
            "plan_binding": {
                "path": str(UPSTREAM_PLAN_PATH),
                "sha256": UPSTREAM_PLAN_SHA256,
                "audit": upstream_plan["audit"],
            },
            "confirmation_commitment_binding": {
                "path": upstream_plan["confirmation_commitment"]["path"],
                "sha256": upstream_plan["confirmation_commitment"]["sha256"],
                "generator_source_contract": generator_contract,
            },
            "frozen_inputs": upstream_plan["frozen_inputs"],
            "shard_receipts": shard_receipts,
            "feature_merge": feature_merge,
        },
        "normalization_proof": context["normalization_proof"],
        "allowed_transformation": ALLOWED_TRANSFORMATION,
        "fit_contract": {
            "checkpoint_step": upstream_plan["checkpoint_step"],
            "d_model": upstream_plan["d_model"],
            "vocab_size": upstream_plan["vocab_size"],
            "zero_id": upstream_plan["zero_id"],
            "one_id": upstream_plan["one_id"],
            "board": upstream_plan["board"],
            "board_rows_sha256": upstream_plan["board_rows_sha256"],
            "extraction_order_sha256": upstream_plan["extraction_order_sha256"],
            "fit_budget": upstream_plan["fit_budget"],
            "runtime_contract": upstream_plan["runtime_contract"],
            "deployment_vocabulary": deployed_vocabulary,
            "parameter_ledger": parameter_ledger,
            "constant_bias_control": DOWNSTREAM_EVALUATION_CONTRACT[
                "constant_bias_control"
            ],
            "nuisance_only_control": DOWNSTREAM_EVALUATION_CONTRACT[
                "nuisance_only_control"
            ],
        },
        "downstream_evaluation_contract": DOWNSTREAM_EVALUATION_CONTRACT,
        "output_contract": {
            "root": str(root),
            "fit_artifact": str(root / "fit" / "motor.pt"),
            "development_eval_artifact": str(
                root / "development_eval" / "evaluation.json"
            ),
            "confirmation_eval_artifact": str(
                root / "confirmation_eval" / "evaluation.json"
            ),
            "upstream_root_must_remain_untouched": str(UPSTREAM_ROOT),
            "layout_binding": recovery_layout_binding,
            "slurm_h100_contract": EXPECTED_SLURM_REQUEST,
        },
        "deserialization_contract": DESERIALIZATION_CONTRACT,
        "claim_boundary": RECOVERY_PLAN_CLAIM_BOUNDARY,
    }
    return canonical_json_document(document)


def validate_recovery_plan_document(
    observed,
    expected,
    recovery_source_commit,
):
    if type(observed) is not dict or set(observed) != RECOVERY_PLAN_KEYS:
        raise ValueError("recovery plan schema mismatch")
    if (
        observed.get("audit") != RECOVERY_PLAN_AUDIT
        or observed.get("recovery") is not True
        or not type_strict_equal(observed, expected)
    ):
        raise ValueError(
            "recovery plan differs from independently reconstructed contract"
        )
    root = recovery_root(recovery_source_commit)
    if observed["output_contract"]["root"] != str(root):
        raise ValueError("recovery output root mismatch")
    for name in (
        "fit_artifact",
        "development_eval_artifact",
        "confirmation_eval_artifact",
    ):
        path = Path(observed["output_contract"][name])
        if path == UPSTREAM_ROOT or UPSTREAM_ROOT in path.parents:
            raise ValueError("recovery output aliases the old canonical root")
        if root not in path.parents:
            raise ValueError("recovery output escapes immutable recovery root")
    if not type_strict_equal(
        observed["allowed_transformation"], ALLOWED_TRANSFORMATION
    ):
        raise ValueError("recovery plan admits extra transformations")
    if not type_strict_equal(
        observed["downstream_evaluation_contract"], DOWNSTREAM_EVALUATION_CONTRACT
    ):
        raise ValueError("recovery plan changes the frozen downstream decision")


def _publish_recovery_plan(root, document):
    enforce_secure_creation_umask()
    if root.parent != RECOVERY_PARENT:
        raise ValueError("recovery root parent differs from receipted parent")
    verify_recovery_parent_binding(
        document.get("recovery_parent_binding"),
        document.get("recovery_executor_source_contract"),
    )
    if not root.exists() or root.is_symlink() or not root.is_absolute():
        raise FileNotFoundError("recovery root must be an exact reserved path")
    before_layout = capture_recovery_layout_binding(
        root,
        document.get("recovery_executor_source_contract"),
        phase="reserved",
    )
    if not type_strict_equal(
        before_layout, document.get("output_contract", {}).get("layout_binding")
    ):
        raise ValueError("recovery plan layout differs from signed reservation")
    parent_stat = os.lstat(root.parent)
    if (
        not stat.S_ISDIR(parent_stat.st_mode)
        or stat.S_ISLNK(parent_stat.st_mode)
        or stat.S_IMODE(parent_stat.st_mode) != 0o700
    ):
        raise ValueError("recovery parent is not the secure receipted directory")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(root.parent, directory_flags)
    try:
        opened_parent = os.fstat(parent_fd)
        if (opened_parent.st_dev, opened_parent.st_ino) != (
            parent_stat.st_dev,
            parent_stat.st_ino,
        ):
            raise ValueError("recovery parent changed during exclusive publication")
        root_fd = os.open(root.name, directory_flags, dir_fd=parent_fd)
        opened_root = os.fstat(root_fd)
        linked_root = os.stat(root.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(linked_root.st_mode)
            or stat.S_ISLNK(linked_root.st_mode)
            or (opened_root.st_dev, opened_root.st_ino)
            != (linked_root.st_dev, linked_root.st_ino)
            or stat.S_IMODE(opened_root.st_mode) != 0o700
            or opened_root.st_uid != os.getuid()
            or opened_root.st_gid != os.getgid()
        ):
            raise ValueError("exclusive recovery root identity mismatch")
    except Exception:
        os.close(parent_fd)
        raise
    try:
        if set(os.listdir(root_fd)) != {
            RECOVERY_LAYOUT_RECEIPT_NAME,
            "fit",
            "development_eval",
            "confirmation_eval",
        }:
            raise ValueError("reserved recovery root changed before plan publication")
        payload = canonical_json_receipt_bytes(document)
        plan_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        plan_flags |= getattr(os, "O_NOFOLLOW", 0)
        plan_fd = os.open("recovery_plan.json", plan_flags, 0o600, dir_fd=root_fd)
        with os.fdopen(plan_fd, "wb") as sink:
            sink.write(payload)
            sink.flush()
            os.fsync(sink.fileno())
            os.fchmod(sink.fileno(), 0o444)
            os.fsync(sink.fileno())
            sealed_plan = os.fstat(sink.fileno())
            if stat.S_IMODE(sealed_plan.st_mode) != 0o444 or sealed_plan.st_nlink != 1:
                raise RuntimeError("secure recovery plan publication failed")
        if set(os.listdir(root_fd)) != {
            "recovery_plan.json",
            "fit",
            "development_eval",
            "confirmation_eval",
            RECOVERY_LAYOUT_RECEIPT_NAME,
        }:
            raise ValueError("exclusive recovery root gained unexpected children")
        os.fsync(root_fd)
        os.fchmod(root_fd, 0o555)
        os.fsync(root_fd)
        current_parent = os.lstat(root.parent)
        current_root = os.stat(root.name, dir_fd=parent_fd, follow_symlinks=False)
        if (current_parent.st_dev, current_parent.st_ino) != (
            opened_parent.st_dev,
            opened_parent.st_ino,
        ) or (current_root.st_dev, current_root.st_ino) != (
            opened_root.st_dev,
            opened_root.st_ino,
        ):
            raise RuntimeError("recovery publication path changed before sealing")
        os.fsync(parent_fd)
        verify_recovery_parent_binding(
            document["recovery_parent_binding"],
            document["recovery_executor_source_contract"],
        )
        after_layout = capture_recovery_layout_binding(
            root,
            document["recovery_executor_source_contract"],
            phase="planned",
        )
        if not type_strict_equal(after_layout, before_layout):
            raise RuntimeError("recovery layout changed during plan publication")
    finally:
        os.close(root_fd)
        os.close(parent_fd)


def recovery_fit_state(root, ownership=None):
    directory = Path(root) / "fit"
    observed = os.lstat(directory)
    if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
        raise ValueError("recovery fit is not a regular directory")
    mode = stat.S_IMODE(observed.st_mode)
    children = {item.name for item in directory.iterdir()}
    if RECOVERY_FIT_OWNER_NAME in children:
        if ownership is None:
            allowed = {
                frozenset({RECOVERY_FIT_OWNER_NAME}),
                frozenset({RECOVERY_FIT_OWNER_NAME, RECOVERY_FIT_STAGE_NAME}),
                frozenset({RECOVERY_FIT_OWNER_NAME, "motor.pt"}),
                frozenset(
                    {RECOVERY_FIT_OWNER_NAME, RECOVERY_FIT_STAGE_NAME, "motor.pt"}
                ),
            }
            if mode != 0o700 or frozenset(children) not in allowed:
                raise ValueError("recovery fit publisher residue is not closed-world")
            for name in children:
                linked = os.lstat(directory / name)
                required_mode = 0o600 if name == RECOVERY_FIT_OWNER_NAME else None
                if (
                    not stat.S_ISREG(linked.st_mode)
                    or stat.S_ISLNK(linked.st_mode)
                    or linked.st_uid != os.getuid()
                    or linked.st_gid != os.getgid()
                    or (
                        required_mode is not None
                        and stat.S_IMODE(linked.st_mode) != required_mode
                    )
                    or (
                        name == RECOVERY_FIT_STAGE_NAME
                        and stat.S_IMODE(linked.st_mode) not in {0o600, 0o444}
                    )
                    or (name == "motor.pt" and stat.S_IMODE(linked.st_mode) != 0o444)
                ):
                    raise ValueError("recovery fit publisher residue identity mismatch")
            return "publisher-residue"
        if ownership.lock_name != RECOVERY_FIT_OWNER_NAME:
            raise RuntimeError("recovery fit ownership scope mismatch")
        linked = os.stat(
            RECOVERY_FIT_OWNER_NAME,
            dir_fd=ownership.directory_fd,
            follow_symlinks=False,
        )
        if _stat_identity(linked) != _stat_identity(os.fstat(ownership.lock_fd)):
            raise RuntimeError("recovery fit publisher ownership was retargeted")
        children.remove(RECOVERY_FIT_OWNER_NAME)
    if RECOVERY_FIT_STAGE_NAME in children:
        raise RuntimeError("recovery fit stage was not recovered under ownership")
    if mode == 0o700 and not children:
        return "empty"
    if children != {"motor.pt"}:
        raise ValueError("recovery fit directory is not closed-world")
    artifact = directory / "motor.pt"
    artifact_stat = os.lstat(artifact)
    if (
        not stat.S_ISREG(artifact_stat.st_mode)
        or stat.S_ISLNK(artifact_stat.st_mode)
        or artifact_stat.st_nlink != 1
    ):
        raise ValueError("recovery fit artifact identity mismatch")
    artifact_mode = stat.S_IMODE(artifact_stat.st_mode)
    if artifact_mode != 0o444:
        raise ValueError("recovery fit artifact mode mismatch")
    if mode == 0o700:
        return "recoverable"
    if mode == 0o555:
        return "sealed"
    raise ValueError("recovery fit directory mode mismatch")


class BoundDir:
    """Open directory bound to an externally signed reserved identity."""

    def __init__(
        self,
        path,
        expected_path,
        signed_reserved_identity,
        regular_file_nlink_delta,
        label,
        *,
        required_mode,
        children=None,
    ):
        raw = os.fspath(path)
        expected = Path(expected_path)
        if raw != str(expected) or not expected.is_absolute():
            raise ValueError(f"{label} path aliases or differs from frozen path")
        _require_reserved_layout_identity(signed_reserved_identity, label)
        self.path = expected
        self.label = label
        self.signed_reserved_identity = canonical_json_document(
            signed_reserved_identity
        )
        if type(
            regular_file_nlink_delta
        ) is not int or regular_file_nlink_delta not in {
            0,
            1,
        }:
            raise ValueError(f"{label} link-count policy is malformed")
        self.regular_file_nlink_delta = regular_file_nlink_delta
        self._ancestor_chain = _open_physical_directory_chain(expected, label)
        self.fd = self._ancestor_chain[-1]["fd"]
        try:
            self.identity = _bound_directory_identity(os.fstat(self.fd))
            self.verify(required_mode=required_mode, children=children)
        except Exception:
            self.close()
            raise

    def verify(self, *, required_mode, children=None):
        _verify_physical_directory_chain(self._ancestor_chain, self.label)
        opened = _bound_directory_identity(os.fstat(self.fd))
        parent = self._ancestor_chain[-2]
        linked = _bound_directory_identity(
            os.stat(self.path.name, dir_fd=parent["fd"], follow_symlinks=False)
        )
        observed_children = sorted(os.listdir(self.fd))
        if opened != linked or not _matches_reserved_layout_identity(
            opened,
            self.signed_reserved_identity,
            required_mode,
            regular_file_children=len(observed_children),
            regular_file_nlink_delta=self.regular_file_nlink_delta,
        ):
            raise RuntimeError(f"{self.label} differs from signed reservation")
        if children is not None and observed_children != sorted(children):
            raise ValueError(f"{self.label} directory is not closed-world")
        self.identity = opened
        return opened

    def chmod(self, current_mode, new_mode, *, children):
        self.verify(required_mode=current_mode, children=children)
        os.fchmod(self.fd, int(new_mode))
        os.fsync(self.fd)
        self._ancestor_chain[-1]["identity"] = _stable_directory_identity(
            os.fstat(self.fd)
        )
        return self.verify(required_mode=new_mode, children=children)

    def close(self):
        for item in reversed(getattr(self, "_ancestor_chain", ())):
            try:
                os.close(item["fd"])
            except OSError:
                pass
        self._ancestor_chain = []
        self.fd = -1


def _signed_fit_directory_identity(layout_binding, directory):
    path = Path(directory)
    if (
        type(layout_binding) is not dict
        or layout_binding.get("schema") != RECOVERY_LAYOUT_BINDING_SCHEMA
    ):
        raise ValueError("signed recovery layout binding is malformed")
    identity = layout_binding.get("directory_identities", {}).get("fit")
    receipt = layout_binding.get("phase_independent_receipt", {}).get("document", {})
    fit_receipt = receipt.get("directories", {}).get("fit", {})
    if fit_receipt.get("path") != str(path) or not type_strict_equal(
        identity, fit_receipt.get("identity")
    ):
        raise ValueError("signed recovery fit directory identity is inconsistent")
    _require_reserved_layout_identity(identity, "signed recovery fit")
    regular_file_nlink_delta = receipt.get("directory_regular_file_nlink_delta")
    if type(regular_file_nlink_delta) is not int or regular_file_nlink_delta not in {
        0,
        1,
    }:
        raise ValueError("signed recovery layout link-count policy is malformed")
    return identity, regular_file_nlink_delta


def _open_recovery_fit_directory(
    directory, required_mode, layout_binding, children=None
):
    path = Path(directory)
    identity, regular_file_nlink_delta = _signed_fit_directory_identity(
        layout_binding, path
    )
    return BoundDir(
        str(path),
        path,
        identity,
        regular_file_nlink_delta,
        "recovery fit",
        required_mode=required_mode,
        children=children,
    )


def acquire_recovery_fit_ownership(out, layout_binding):
    path = Path(out)
    if not path.is_absolute() or path.name != "motor.pt":
        raise ValueError("recovery artifact path is not exact")
    directory = _open_recovery_fit_directory(
        path.parent, 0o700, layout_binding, children=None
    )
    try:
        return PublisherOwnership(
            directory.fd,
            path.parent,
            lock_name=RECOVERY_FIT_OWNER_NAME,
            purpose="recovery-fit-motor",
            target_name=path.name,
            stage_name=RECOVERY_FIT_STAGE_NAME,
        )
    finally:
        directory.close()


def _verify_owned_fit_directory(ownership, layout_binding, children):
    directory = _open_recovery_fit_directory(
        ownership.directory_path, 0o700, layout_binding, children=children
    )
    try:
        if _publisher_directory_identity(os.fstat(directory.fd)) != (
            _publisher_directory_identity(os.fstat(ownership.directory_fd))
        ):
            raise RuntimeError("owned recovery fit directory was retargeted")
    finally:
        directory.close()


def prepare_recovery_fit_publication(out, layout_binding):
    ownership = acquire_recovery_fit_ownership(out, layout_binding)
    try:
        state = recovery_fit_state(Path(out).parent.parent, ownership)
        _verify_owned_fit_directory(
            ownership,
            layout_binding,
            (RECOVERY_FIT_OWNER_NAME,) + (("motor.pt",) if state != "empty" else ()),
        )
        ownership.close()
        return state
    except Exception:
        preserve = bool(
            ownership.document and ownership.document.get("stage") is not None
        )
        ownership.close(preserve=preserve)
        raise


def publish_recovery_torch(out, value, layout_binding):
    enforce_secure_creation_umask()
    assert_reviewed_callable_exports()
    path = Path(out)
    if not path.is_absolute() or path.name != "motor.pt":
        raise ValueError("recovery artifact path is not exact")
    ownership = acquire_recovery_fit_ownership(path, layout_binding)
    artifact_fd = -1
    try:
        if recovery_fit_state(path.parent.parent, ownership) != "empty":
            raise FileExistsError("recovery fit directory is not empty")
        _verify_owned_fit_directory(
            ownership, layout_binding, (RECOVERY_FIT_OWNER_NAME,)
        )
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        artifact_fd = os.open(
            RECOVERY_FIT_STAGE_NAME,
            flags,
            0o600,
            dir_fd=ownership.directory_fd,
        )
        ownership.bind_stage(artifact_fd)
        with os.fdopen(artifact_fd, "w+b", closefd=False) as sink:
            REVIEWED_TORCH_SAVE(value, sink)
            assert_reviewed_callable_exports()
            sink.flush()
            os.fsync(artifact_fd)
            linked = os.stat(
                RECOVERY_FIT_STAGE_NAME,
                dir_fd=ownership.directory_fd,
                follow_symlinks=False,
            )
            if (
                _stat_identity(linked) != _stat_identity(os.fstat(artifact_fd))
                or linked.st_nlink != 1
            ):
                raise RuntimeError("recovery artifact stage changed while writing")
            sink.seek(0)
            digest = hashlib.sha256()
            for block in iter(lambda: sink.read(1024 * 1024), b""):
                digest.update(block)
            _verify_owned_fit_directory(
                ownership,
                layout_binding,
                (RECOVERY_FIT_OWNER_NAME, RECOVERY_FIT_STAGE_NAME),
            )
            os.fchmod(artifact_fd, 0o444)
            os.fsync(artifact_fd)
        os.link(
            RECOVERY_FIT_STAGE_NAME,
            path.name,
            src_dir_fd=ownership.directory_fd,
            dst_dir_fd=ownership.directory_fd,
            follow_symlinks=False,
        )
        committed = os.stat(
            path.name, dir_fd=ownership.directory_fd, follow_symlinks=False
        )
        staged = os.stat(
            RECOVERY_FIT_STAGE_NAME,
            dir_fd=ownership.directory_fd,
            follow_symlinks=False,
        )
        if (
            _publisher_file_identity(committed) != _publisher_file_identity(staged)
            or committed.st_nlink != 2
            or stat.S_IMODE(committed.st_mode) != 0o444
        ):
            raise RuntimeError("recovery artifact atomic no-replace commit failed")
        os.fsync(ownership.directory_fd)
        os.unlink(RECOVERY_FIT_STAGE_NAME, dir_fd=ownership.directory_fd)
        os.fsync(ownership.directory_fd)
        os.close(artifact_fd)
        artifact_fd = -1
        ownership.clear_stage_binding()
        _verify_owned_fit_directory(
            ownership, layout_binding, (RECOVERY_FIT_OWNER_NAME, path.name)
        )
        ownership.close()
        return digest.hexdigest()
    except Exception:
        if artifact_fd >= 0:
            os.close(artifact_fd)
            artifact_fd = -1
        preserve = bool(
            ownership.document and ownership.document.get("stage") is not None
        )
        ownership.close(preserve=preserve)
        raise
    finally:
        if artifact_fd >= 0:
            os.close(artifact_fd)


def seal_recovery_fit(out, layout_binding):
    path = Path(out)
    if not path.is_absolute() or path.name != "motor.pt":
        raise ValueError("recovery artifact path is not exact")
    directory = _open_recovery_fit_directory(
        path.parent, 0o700, layout_binding, children=(path.name,)
    )
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        artifact_fd = os.open(path.name, flags, dir_fd=directory.fd)
        try:
            artifact = os.fstat(artifact_fd)
            linked = os.stat(path.name, dir_fd=directory.fd, follow_symlinks=False)
            if (
                _stat_identity(artifact) != _stat_identity(linked)
                or not stat.S_ISREG(artifact.st_mode)
                or artifact.st_nlink != 1
                or stat.S_IMODE(artifact.st_mode) != 0o444
            ):
                raise ValueError("recovery fit artifact cannot be sealed")
            os.fsync(artifact_fd)
        finally:
            os.close(artifact_fd)
        os.fsync(directory.fd)
        directory.chmod(0o700, 0o555, children=(path.name,))
        current = os.fstat(directory.fd)
        current_artifact = os.stat(
            path.name, dir_fd=directory.fd, follow_symlinks=False
        )
        if (
            not _matches_reserved_layout_identity(
                _bound_directory_identity(current),
                directory.signed_reserved_identity,
                0o555,
                regular_file_children=1,
                regular_file_nlink_delta=directory.regular_file_nlink_delta,
            )
            or sorted(os.listdir(directory.fd)) != [path.name]
            or current_artifact.st_dev != artifact.st_dev
            or current_artifact.st_ino != artifact.st_ino
            or current_artifact.st_nlink != 1
            or stat.S_IMODE(current_artifact.st_mode) != 0o444
        ):
            raise RuntimeError("recovery fit directory changed while sealing")
    finally:
        directory.close()


def validate_recovery_layout(root, *, fit_state="empty"):
    _require_directory(
        root,
        0o555,
        "recovery root",
        (
            "recovery_plan.json",
            RECOVERY_LAYOUT_RECEIPT_NAME,
            "fit",
            "development_eval",
            "confirmation_eval",
        ),
    )
    plan = root / "recovery_plan.json"
    observed = os.lstat(plan)
    if (
        not stat.S_ISREG(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o444
        or observed.st_nlink != 1
    ):
        raise ValueError("recovery plan file identity mismatch")
    observed_fit_state = recovery_fit_state(root)
    if observed_fit_state != fit_state:
        raise ValueError(
            f"recovery fit state mismatch: expected {fit_state}, got {observed_fit_state}"
        )
    if fit_state == "publisher-residue":
        fit_children = tuple(sorted(item.name for item in (root / "fit").iterdir()))
        _require_directory(root / "fit", 0o700, "recovery fit", fit_children)
    _require_directory(root / "development_eval", 0o700, "recovery development", ())
    _require_directory(root / "confirmation_eval", 0o700, "recovery confirmation", ())


def parse_slurm_key_value_record(text, label="scontrol record"):
    if type(text) is not str or not text.strip():
        raise ValueError(f"{label} is empty")
    result = {}
    for token in shlex.split(text.strip()):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if not key or key in result:
            raise ValueError(f"{label} has a duplicate or empty key")
        result[key] = value
    if not result:
        raise ValueError(f"{label} has no key-value fields")
    return result


def parse_slurm_tres(value, label):
    if type(value) is not str or not value:
        raise ValueError(f"{label} is empty")
    result = {}
    for item in value.split(","):
        if "=" not in item:
            raise ValueError(f"{label} contains an untyped TRES item")
        key, raw = item.rsplit("=", 1)
        if (
            not key
            or key in result
            or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?[KMGTP]?", raw)
        ):
            raise ValueError(f"{label} contains a malformed or duplicate TRES")
        result[key] = raw
    return result


def _memory_to_mib(value):
    match = re.fullmatch(r"([0-9]+)([KMGTP]?)([cn]?)", str(value))
    if match is None:
        raise ValueError("Slurm memory field is malformed")
    amount, unit, _scope = match.groups()
    factors = {
        "": 1 / (1024 * 1024),
        "K": 1 / 1024,
        "M": 1,
        "G": 1024,
        "T": 1024**2,
        "P": 1024**3,
    }
    result = int(int(amount) * factors[unit])
    if result <= 0:
        raise ValueError("Slurm memory field is non-positive")
    return result


def _time_to_seconds(value):
    text = str(value)
    days = 0
    if "-" in text:
        day_text, text = text.split("-", 1)
        if not day_text.isdigit():
            raise ValueError("Slurm time field is malformed")
        days = int(day_text)
    parts = text.split(":")
    if len(parts) not in {2, 3} or not all(part.isdigit() for part in parts):
        raise ValueError("Slurm time field is malformed")
    if len(parts) == 2:
        hours, minutes, seconds = 0, int(parts[0]), int(parts[1])
    else:
        hours, minutes, seconds = map(int, parts)
    if minutes >= 60 or seconds >= 60:
        raise ValueError("Slurm time field is out of range")
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def parse_sacct_pipe_records(text, fields):
    if type(text) is not str or type(fields) is not tuple or not fields:
        raise ValueError("sacct parser contract is invalid")
    records = []
    for line in text.splitlines():
        if not line.strip():
            continue
        values = line.rstrip("|").split("|")
        if len(values) != len(fields):
            raise ValueError("sacct record field count mismatch")
        record = dict(zip(fields, values))
        if any(value == "" for value in record.values()):
            raise ValueError("sacct record contains an empty field")
        records.append(record)
    if not records:
        raise ValueError("sacct returned no records")
    return records


def parse_slurm_hostnames(text, label):
    if type(text) is not str:
        raise ValueError(f"{label} host list is malformed")
    hosts = [line.strip() for line in text.splitlines() if line.strip()]
    if (
        not hosts
        or len(hosts) != len(set(hosts))
        or any(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", host) is None for host in hosts)
    ):
        raise ValueError(f"{label} host list is malformed or duplicated")
    return hosts


def validate_slurm_records(job, accounting):
    if (
        type(job) is not dict
        or type(accounting) is not dict
        or any(
            type(key) is not str or type(value) is not str for key, value in job.items()
        )
        or any(
            type(key) is not str or type(value) is not str
            for key, value in accounting.items()
        )
    ):
        raise ValueError("Slurm allocation records must be exact string dictionaries")
    required_job = {
        "JobId",
        "Account",
        "Partition",
        "NumNodes",
        "NumTasks",
        "CPUs/Task",
        "MinMemoryNode",
        "TimeLimit",
        "Requeue",
        "ReqTRES",
        "AllocTRES",
        "TresPerNode",
        "JobState",
        "NodeList",
        "ExcNodeList",
    }
    if not required_job.issubset(job):
        raise ValueError("scontrol record is missing required allocation fields")
    expected = EXPECTED_SLURM_REQUEST
    exact = {
        "Account": expected["account"],
        "Partition": expected["partition"],
        "NumNodes": str(expected["nodes"]),
        "NumTasks": str(expected["tasks"]),
        "CPUs/Task": str(expected["cpus_per_task"]),
        "Requeue": "0",
        "JobState": "RUNNING",
    }
    for key, value in exact.items():
        if job[key] != value:
            raise ValueError(f"Slurm allocation mismatch: {key}")
    if (
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", job["NodeList"]) is None
        or job["NodeList"] in EXPECTED_EXCLUDED_NODES
        or not job["ExcNodeList"]
    ):
        raise ValueError("Slurm node allocation or exclusion is malformed")
    if _memory_to_mib(job["MinMemoryNode"]) != expected["memory_mib"]:
        raise ValueError("Slurm allocation memory mismatch")
    if _time_to_seconds(job["TimeLimit"]) != expected["time_limit_seconds"]:
        raise ValueError("Slurm allocation time limit mismatch")
    typed_gpu = f"gres/gpu:{expected['gpu_tres_name']}"
    for label, raw in (("requested", job["ReqTRES"]), ("allocated", job["AllocTRES"])):
        tres = parse_slurm_tres(raw, f"{label} TRES")
        if set(tres) != {"billing", "cpu", "mem", "node", "gres/gpu", typed_gpu}:
            raise ValueError(f"Slurm {label} TRES schema mismatch")
        if (
            not re.fullmatch(r"[1-9][0-9]*", tres["billing"])
            or tres.get("cpu") != str(expected["cpus_per_task"])
            or _memory_to_mib(tres.get("mem")) != expected["memory_mib"]
            or tres.get("node") != str(expected["nodes"])
            or tres.get("gres/gpu") != str(expected["gpu_count"])
            or tres.get(typed_gpu) != str(expected["gpu_count"])
        ):
            raise ValueError(f"Slurm {label} TRES mismatch")
    if (
        job["TresPerNode"]
        != f"gres/gpu:{expected['gpu_tres_name']}:{expected['gpu_count']}"
    ):
        raise ValueError("Slurm typed GPU GRES mismatch")
    accounting_exact = {
        "JobIDRaw": job["JobId"],
        "Account": expected["account"],
        "Partition": expected["partition"],
        "AllocCPUS": str(expected["cpus_per_task"]),
        "NNodes": str(expected["nodes"]),
        "NTasks": str(expected["tasks"]),
        "State": "RUNNING",
        "TimelimitRaw": str(expected["time_limit_seconds"]),
    }
    for key, value in accounting_exact.items():
        if accounting.get(key) != value:
            raise ValueError(f"sacct allocation mismatch: {key}")
    if _memory_to_mib(accounting.get("ReqMem")) != expected["memory_mib"]:
        raise ValueError("sacct requested memory mismatch")
    for key in ("ReqTRES", "AllocTRES"):
        if parse_slurm_tres(accounting.get(key), f"sacct {key}") != parse_slurm_tres(
            job[key], f"scontrol {key}"
        ):
            raise ValueError(f"sacct/scontrol {key} mismatch")
    return True


def _normalize_pci_bus_id(value):
    match = re.fullmatch(
        r"(?:(?P<long>[0-9A-Fa-f]{8})|(?P<short>[0-9A-Fa-f]{4})):"
        r"(?P<bus>[0-9A-Fa-f]{2}):(?P<device>[0-9A-Fa-f]{2})\."
        r"(?P<function>[0-7])",
        str(value),
    )
    if match is None:
        raise ValueError("GPU PCI bus id is malformed")
    domain = match.group("long") or match.group("short")
    if len(domain) == 8:
        if domain[:4] != "0000":
            raise ValueError("GPU PCI domain is not canonical")
        domain = domain[4:]
    return (
        f"{domain}:{match.group('bus')}:{match.group('device')}."
        f"{match.group('function')}"
    ).lower()


def _read_kernel_text(path, label):
    payload, _identity = _preimport_read_regular(path, label, one_link=False)
    try:
        return payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"{label} is not ASCII") from exc


def _capture_effective_gpu_cgroup_authorization(job_id, gpu_index, pci_bus_id):
    if not re.fullmatch(r"[0-9]+", str(gpu_index)):
        raise RuntimeError("attested GPU index is malformed")
    selected_path = Path(f"/dev/nvidia{gpu_index}")
    selected = os.stat(selected_path, follow_symlinks=False)
    if not stat.S_ISCHR(selected.st_mode) or stat.S_ISLNK(selected.st_mode):
        raise RuntimeError("attested NVIDIA device node is not a character device")

    authorized = []
    for entry in sorted(os.scandir("/dev"), key=lambda item: item.name):
        if re.fullmatch(r"nvidia[0-9]+", entry.name) is None:
            continue
        observed = os.stat(entry.path, follow_symlinks=False)
        if not stat.S_ISCHR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
            raise RuntimeError(
                "NVIDIA GPU device node is not an exact character device"
            )
        try:
            descriptor = os.open(
                entry.path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
        except PermissionError:
            continue
        try:
            opened = os.fstat(descriptor)
            if _preimport_stat_identity(opened) != _preimport_stat_identity(observed):
                raise RuntimeError("NVIDIA device node changed during authorization")
            authorized.append(entry.path)
        finally:
            os.close(descriptor)
    if authorized != [str(selected_path)]:
        raise RuntimeError("cgroup does not authorize exactly one NVIDIA GPU device")

    major = os.major(selected.st_rdev)
    minor = os.minor(selected.st_rdev)
    sysfs_link = Path(f"/sys/dev/char/{major}:{minor}/device")
    sysfs_device = sysfs_link.resolve(strict=True)
    pci_components = [
        item
        for item in sysfs_device.parts
        if re.fullmatch(r"[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]", item)
    ]
    normalized_pci = _normalize_pci_bus_id(pci_bus_id)
    if len(pci_components) != 1 or pci_components[0].lower() != normalized_pci:
        raise RuntimeError("NVIDIA device node does not map to the attested PCI bus")

    cgroup_text = _read_kernel_text("/proc/self/cgroup", "process cgroup membership")
    memberships = []
    for line in cgroup_text.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3 or not fields[0].isdigit() or not fields[2].startswith("/"):
            raise RuntimeError("process cgroup membership is malformed")
        controllers = [item for item in fields[1].split(",") if item]
        if "devices" in controllers or (fields[0] == "0" and not controllers):
            memberships.append(
                {
                    "hierarchy": fields[0],
                    "controllers": controllers,
                    "path": fields[2],
                }
            )
    if len(memberships) != 1:
        raise RuntimeError("process has ambiguous device cgroup membership")
    membership = memberships[0]
    if (
        re.search(
            rf"(?:^|[/_.-])job[_-]?{re.escape(str(job_id))}(?:$|[/_.-])",
            membership["path"],
        )
        is None
    ):
        raise RuntimeError("device cgroup is not bound to the Slurm job id")

    return canonical_json_document(
        {
            "membership": membership,
            "authorization_method": "effective_character_device_open",
            "authorized_gpu_device_nodes": authorized,
            "selected_device": {
                "path": str(selected_path),
                "major": major,
                "minor": minor,
                "mode": stat.S_IMODE(selected.st_mode),
                "device": selected.st_dev,
                "inode": selected.st_ino,
                "links": selected.st_nlink,
            },
            "sysfs_device_path": str(sysfs_device),
            "pci_bus_id": normalized_pci,
            "effective_authorization": True,
        }
    )


def _parse_nvidia_smi_list(text, gpu):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    pattern = re.compile(
        r"GPU ([0-9]+): (.+) \(UUID: "
        r"(GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\)"
    )
    if len(lines) != 1:
        raise RuntimeError("nvidia-smi -L exposed a GPU or MIG identity ambiguity")
    match = pattern.fullmatch(lines[0])
    if match is None or [match.group(1), match.group(2), match.group(3)] != [
        gpu["index"],
        gpu["name"],
        gpu["uuid"],
    ]:
        raise RuntimeError("nvidia-smi list identity differs from queried GPU")
    return canonical_json_document(
        {
            "mode": gpu["mig.mode.current"],
            "gpu_uuid": gpu["uuid"],
            "gpu_instance_id": None,
            "compute_instance_id": None,
            "mig_device_uuid": None,
            "nvidia_smi_list": lines,
        }
    )


def capture_slurm_h100_attestation():
    for binary, label in (
        (PINNED_SCONTROL, "scontrol"),
        (PINNED_SACCT, "sacct"),
        (PINNED_NVIDIA_SMI, "nvidia-smi"),
    ):
        if not binary.is_file() or binary.is_symlink():
            raise RuntimeError(f"pinned {label} executable is unavailable")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[1-9][0-9]*", job_id):
        raise RuntimeError("recovery fit requires an exact Slurm job id")
    if os.environ.get("SLURM_RESTART_COUNT", "0") != "0":
        raise RuntimeError("recovery fit refuses restarted Slurm allocations")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"[^,\s]+", cuda_visible):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must name exactly one GPU")
    job_text = subprocess.check_output(
        [str(PINNED_SCONTROL), "show", "job", "-o", job_id], text=True
    )
    job = parse_slurm_key_value_record(job_text)
    fields = (
        "JobIDRaw",
        "Account",
        "Partition",
        "AllocCPUS",
        "NNodes",
        "NTasks",
        "ReqMem",
        "ReqTRES",
        "AllocTRES",
        "State",
        "TimelimitRaw",
    )
    accounting_text = subprocess.check_output(
        [
            str(PINNED_SACCT),
            "-X",
            "-n",
            "-P",
            "-j",
            job_id,
            "--format=" + ",".join(fields),
        ],
        text=True,
    )
    records = parse_sacct_pipe_records(accounting_text, fields)
    matches = [record for record in records if record["JobIDRaw"] == job_id]
    if len(matches) != 1:
        raise RuntimeError("sacct did not return exactly one allocation record")
    accounting = matches[0]
    validate_slurm_records(job, accounting)
    allocated_nodes = parse_slurm_hostnames(
        subprocess.check_output(
            [str(PINNED_SCONTROL), "show", "hostnames", job["NodeList"]],
            text=True,
        ),
        "allocated",
    )
    excluded_nodes = parse_slurm_hostnames(
        subprocess.check_output(
            [str(PINNED_SCONTROL), "show", "hostnames", job["ExcNodeList"]],
            text=True,
        ),
        "excluded",
    )
    if (
        allocated_nodes != [job["NodeList"]]
        or excluded_nodes != list(EXPECTED_EXCLUDED_NODES)
        or set(allocated_nodes) & set(excluded_nodes)
    ):
        raise RuntimeError("Slurm NodeList/ExcNodeList violates the sealed exclusion")
    node_binding = canonical_json_document(
        {
            "node_list_expression": job["NodeList"],
            "allocated_nodes": allocated_nodes,
            "allocated_nodes_sha256": stable_json_sha256(allocated_nodes),
            "excluded_node_list_expression": job["ExcNodeList"],
            "excluded_nodes": excluded_nodes,
            "excluded_nodes_sha256": stable_json_sha256(excluded_nodes),
        }
    )
    query_fields = (
        "index",
        "uuid",
        "pci.bus_id",
        "name",
        "memory.total",
        "compute_cap",
        "mig.mode.current",
    )
    gpu_text = subprocess.check_output(
        [
            str(PINNED_NVIDIA_SMI),
            "--query-gpu=" + ",".join(query_fields),
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = [
        [value.strip() for value in line.split(",")]
        for line in gpu_text.splitlines()
        if line.strip()
    ]
    if len(rows) != 1 or len(rows[0]) != len(query_fields):
        raise RuntimeError("nvidia-smi did not expose exactly one complete GPU")
    gpu = dict(zip(query_fields, rows[0]))
    expected = EXPECTED_SLURM_REQUEST
    try:
        memory_mib = int(gpu["memory.total"])
    except ValueError as exc:
        raise RuntimeError("GPU memory total is not an integer MiB value") from exc
    if (
        gpu["name"] != expected["gpu_name"]
        or gpu["compute_cap"] != expected["compute_capability"]
        or gpu["mig.mode.current"] != expected["mig_mode"]
        or not expected["memory_total_mib_min"]
        <= memory_mib
        <= expected["memory_total_mib_max"]
        or REVIEWED_CUDA_DEVICE_COUNT() != 1
    ):
        raise RuntimeError(
            "visible GPU type, capability, memory, or MIG state mismatch"
        )
    if cuda_visible not in {
        gpu["index"],
        gpu["uuid"],
        gpu["uuid"].removeprefix("GPU-"),
        gpu["pci.bus_id"],
        _normalize_pci_bus_id(gpu["pci.bus_id"]),
    }:
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not bind the attested GPU")
    slurm_job_gpus = os.environ.get("SLURM_JOB_GPUS", "")
    slurm_step_gpus = os.environ.get("SLURM_STEP_GPUS", slurm_job_gpus)
    gpu_tokens = {
        gpu["index"],
        gpu["uuid"],
        gpu["uuid"].removeprefix("GPU-"),
        gpu["pci.bus_id"],
        _normalize_pci_bus_id(gpu["pci.bus_id"]),
    }
    if slurm_job_gpus not in gpu_tokens or slurm_step_gpus not in gpu_tokens:
        raise RuntimeError("Slurm GRES GPU identity differs from the attested GPU")
    cgroup = _capture_effective_gpu_cgroup_authorization(
        job_id, gpu["index"], gpu["pci.bus_id"]
    )
    list_text = subprocess.check_output([str(PINNED_NVIDIA_SMI), "-L"], text=True)
    mig_identity = _parse_nvidia_smi_list(list_text, gpu)
    return canonical_json_document(
        {
            "schema": RECOVERY_SLURM_CONTRACT_SCHEMA,
            "job_id": int(job_id),
            "restart_count": 0,
            "cuda_visible_devices": cuda_visible,
            "slurm_gres": {
                "account": job["Account"],
                "job_id": job["JobId"],
                "alloc_tres": job["AllocTRES"],
                "tres_per_node": job["TresPerNode"],
                "slurm_job_gpus": slurm_job_gpus,
                "slurm_step_gpus": slurm_step_gpus,
            },
            "slurm_node_binding": node_binding,
            "scontrol": job,
            "sacct": accounting,
            "gpu": {**gpu, "memory.total": memory_mib},
            "cgroup_device_authorization": cgroup,
            "mig_identity": mig_identity,
            "contract": EXPECTED_SLURM_REQUEST,
        }
    )


def validate_slurm_h100_attestation(attestation):
    expected_keys = {
        "schema",
        "job_id",
        "restart_count",
        "cuda_visible_devices",
        "slurm_gres",
        "slurm_node_binding",
        "scontrol",
        "sacct",
        "gpu",
        "cgroup_device_authorization",
        "mig_identity",
        "contract",
    }
    if (
        type(attestation) is not dict
        or set(attestation) != expected_keys
        or attestation.get("schema") != RECOVERY_SLURM_CONTRACT_SCHEMA
        or not type_strict_equal(attestation.get("contract"), EXPECTED_SLURM_REQUEST)
        or type(attestation.get("job_id")) is not int
        or type(attestation.get("restart_count")) is not int
        or attestation.get("restart_count") != 0
    ):
        raise ValueError("recovery Slurm/H100 attestation schema mismatch")
    validate_slurm_records(attestation.get("scontrol"), attestation.get("sacct"))
    job_id = str(attestation["job_id"])
    if (
        attestation["scontrol"].get("JobId") != job_id
        or attestation["sacct"].get("JobIDRaw") != job_id
    ):
        raise ValueError("recovery Slurm JobID attestation mismatch")
    node_binding = attestation.get("slurm_node_binding")
    node_keys = {
        "node_list_expression",
        "allocated_nodes",
        "allocated_nodes_sha256",
        "excluded_node_list_expression",
        "excluded_nodes",
        "excluded_nodes_sha256",
    }
    if (
        type(node_binding) is not dict
        or set(node_binding) != node_keys
        or node_binding["node_list_expression"]
        != attestation["scontrol"]["NodeList"]
        or node_binding["excluded_node_list_expression"]
        != attestation["scontrol"]["ExcNodeList"]
        or node_binding["allocated_nodes"]
        != [attestation["scontrol"]["NodeList"]]
        or node_binding["excluded_nodes"] != list(EXPECTED_EXCLUDED_NODES)
        or node_binding["allocated_nodes_sha256"]
        != stable_json_sha256(node_binding["allocated_nodes"])
        or node_binding["excluded_nodes_sha256"]
        != stable_json_sha256(node_binding["excluded_nodes"])
        or set(node_binding["allocated_nodes"]) & set(node_binding["excluded_nodes"])
        or "evc43" not in node_binding["excluded_nodes"]
    ):
        raise ValueError("recovery Slurm node allocation violates exclusions")
    gpu = attestation.get("gpu")
    if type(gpu) is not dict or set(gpu) != {
        "index",
        "uuid",
        "pci.bus_id",
        "name",
        "memory.total",
        "compute_cap",
        "mig.mode.current",
    }:
        raise ValueError("recovery GPU attestation schema mismatch")
    expected = EXPECTED_SLURM_REQUEST
    if (
        type(gpu["memory.total"]) is not int
        or gpu["name"] != expected["gpu_name"]
        or gpu["compute_cap"] != expected["compute_capability"]
        or gpu["mig.mode.current"] != expected["mig_mode"]
        or not expected["memory_total_mib_min"]
        <= gpu["memory.total"]
        <= expected["memory_total_mib_max"]
        or not re.fullmatch(
            r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
            gpu["uuid"],
        )
    ):
        raise ValueError("recovery GPU hardware attestation mismatch")
    normalized_pci = _normalize_pci_bus_id(gpu["pci.bus_id"])
    if attestation["cuda_visible_devices"] not in {
        gpu["index"],
        gpu["uuid"],
        gpu["uuid"].removeprefix("GPU-"),
        gpu["pci.bus_id"],
        normalized_pci,
    }:
        raise ValueError("recovery CUDA_VISIBLE_DEVICES attestation mismatch")

    gres = attestation.get("slurm_gres")
    if type(gres) is not dict or set(gres) != {
        "account",
        "job_id",
        "alloc_tres",
        "tres_per_node",
        "slurm_job_gpus",
        "slurm_step_gpus",
    }:
        raise ValueError("recovery Slurm GRES attestation schema mismatch")
    gpu_tokens = {
        gpu["index"],
        gpu["uuid"],
        gpu["uuid"].removeprefix("GPU-"),
        gpu["pci.bus_id"],
        normalized_pci,
    }
    if (
        gres["account"] != expected["account"]
        or gres["job_id"] != job_id
        or gres["alloc_tres"] != attestation["scontrol"]["AllocTRES"]
        or gres["tres_per_node"] != attestation["scontrol"]["TresPerNode"]
        or gres["slurm_job_gpus"] not in gpu_tokens
        or gres["slurm_step_gpus"] not in gpu_tokens
    ):
        raise ValueError("recovery Slurm GRES does not bind the attested GPU")

    cgroup = attestation.get("cgroup_device_authorization")
    cgroup_keys = {
        "membership",
        "authorization_method",
        "authorized_gpu_device_nodes",
        "selected_device",
        "sysfs_device_path",
        "pci_bus_id",
        "effective_authorization",
    }
    if type(cgroup) is not dict or set(cgroup) != cgroup_keys:
        raise ValueError("recovery cgroup device attestation schema mismatch")
    membership = cgroup["membership"]
    selected = cgroup["selected_device"]
    selected_path = f"/dev/nvidia{gpu['index']}"
    if (
        type(membership) is not dict
        or set(membership) != {"hierarchy", "controllers", "path"}
        or type(membership["hierarchy"]) is not str
        or type(membership["controllers"]) is not list
        or any(type(item) is not str for item in membership["controllers"])
        or type(membership["path"]) is not str
        or re.search(
            rf"(?:^|[/_.-])job[_-]?{re.escape(job_id)}(?:$|[/_.-])",
            membership["path"],
        )
        is None
        or type(selected) is not dict
        or set(selected)
        != {"path", "major", "minor", "mode", "device", "inode", "links"}
        or selected["path"] != selected_path
        or any(
            type(selected[name]) is not int
            for name in ("major", "minor", "mode", "device", "inode", "links")
        )
        or cgroup["authorization_method"] != "effective_character_device_open"
        or cgroup["authorized_gpu_device_nodes"] != [selected_path]
        or cgroup["pci_bus_id"] != normalized_pci
        or cgroup["effective_authorization"] is not True
        or normalized_pci not in Path(cgroup["sysfs_device_path"]).parts
    ):
        raise ValueError("recovery cgroup does not authorize the attested PCI GPU")

    mig_identity = attestation.get("mig_identity")
    expected_mig_keys = {
        "mode",
        "gpu_uuid",
        "gpu_instance_id",
        "compute_instance_id",
        "mig_device_uuid",
        "nvidia_smi_list",
    }
    if (
        type(mig_identity) is not dict
        or set(mig_identity) != expected_mig_keys
        or mig_identity["mode"] != expected["mig_mode"]
        or mig_identity["gpu_uuid"] != gpu["uuid"]
        or any(
            mig_identity[name] is not None
            for name in (
                "gpu_instance_id",
                "compute_instance_id",
                "mig_device_uuid",
            )
        )
        or type(mig_identity["nvidia_smi_list"]) is not list
        or any(type(item) is not str for item in mig_identity["nvidia_smi_list"])
    ):
        raise ValueError("recovery MIG identity is not one-to-one and disabled")
    try:
        parsed_mig_identity = _parse_nvidia_smi_list(
            "\n".join(mig_identity["nvidia_smi_list"]), gpu
        )
    except RuntimeError as exc:
        raise ValueError(
            "recovery MIG identity is not one-to-one and disabled"
        ) from exc
    if not type_strict_equal(parsed_mig_identity, mig_identity):
        raise ValueError("recovery MIG identity is not one-to-one and disabled")
    return True


def require_recovery_cuda_runtime(upstream_plan):
    assert_reviewed_callable_exports()
    slurm_attestation = capture_slurm_h100_attestation()
    device = REVIEWED_REQUIRE_CANONICAL_CUDA()
    observed = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device": REVIEWED_CUDA_GET_DEVICE_NAME(0),
    }
    expected = upstream_plan["runtime_contract"]["artifact_runtime"]
    if not type_strict_equal(observed, expected):
        raise RuntimeError("recovery H100 runtime differs from frozen upstream runtime")
    return device, slurm_attestation


def prepare_context(args):
    executor_runtime_contract = capture_executor_runtime_contract()
    recovery_source_contract = build_recovery_executor_source_contract(
        args.recovery_source_commit,
        args.recovery_source_manifest_sha256,
    )
    recovery_parent_binding = ensure_recovery_parent(
        recovery_source_contract,
        allow_install=False,
    )
    root = recovery_root(args.recovery_source_commit)
    if args.command == "plan":
        recovery_layout_binding = capture_recovery_layout_binding(
            root, recovery_source_contract, phase="reserved"
        )
    else:
        fit_state = recovery_fit_state(root)
        recovery_layout_binding = capture_recovery_layout_binding(
            root,
            recovery_source_contract,
            phase="fit",
            fit_state=fit_state,
        )
    review_bound, review = load_hostile_review(
        args.recovery_source_commit,
        recovery_source_contract,
        executor_runtime_contract,
        recovery_layout_binding,
        args.hostile_review,
        args.hostile_review_sha256,
    )
    upstream_plan_bound = None
    confirmation_bound = None
    frozen_bounds = {}
    shard_bounds = []
    try:
        (
            upstream_plan_bound,
            confirmation_bound,
            upstream_plan,
            source_contract,
            source_hashes,
            upstream_custody_snapshot,
        ) = load_upstream_plan()
        frozen_bounds = _bind_frozen_inputs(upstream_plan)
        board_context = reconstruct_board(upstream_plan, frozen_bounds)
        shard_bounds, features, feature_merge, receipts, expected_bindings = (
            bind_and_merge_upstream_shards(
                upstream_plan,
                board_context["rows"],
                source_contract,
                source_hashes,
            )
        )
        expected_plan = build_recovery_plan_document(
            args.recovery_source_commit,
            recovery_source_contract,
            executor_runtime_contract,
            recovery_parent_binding,
            recovery_layout_binding,
            review_bound,
            review,
            upstream_plan,
            source_contract,
            source_hashes,
            upstream_custody_snapshot,
            board_context,
            feature_merge,
            receipts,
        )
        for bound in (
            review_bound,
            upstream_plan_bound,
            confirmation_bound,
            *frozen_bounds.values(),
            *shard_bounds,
        ):
            bound.verify()
        return {
            "recovery_source_contract": recovery_source_contract,
            "executor_runtime_contract": executor_runtime_contract,
            "recovery_parent_binding": recovery_parent_binding,
            "recovery_layout_binding": recovery_layout_binding,
            "command": args.command,
            "review_bound": review_bound,
            "review": review,
            "upstream_plan_bound": upstream_plan_bound,
            "confirmation_bound": confirmation_bound,
            "upstream_plan": upstream_plan,
            "upstream_source_contract": source_contract,
            "upstream_source_hashes": source_hashes,
            "upstream_custody_snapshot": upstream_custody_snapshot,
            "frozen_bounds": frozen_bounds,
            "board_context": board_context,
            "shard_bounds": shard_bounds,
            "features": features,
            "feature_merge": feature_merge,
            "shard_receipts": receipts,
            "expected_bindings": expected_bindings,
            "expected_recovery_plan": expected_plan,
        }
    except Exception:
        if upstream_plan_bound is not None:
            upstream_plan_bound.close()
        if confirmation_bound is not None:
            confirmation_bound.close()
        review_bound.close()
        for bound in frozen_bounds.values():
            bound.close()
        for bound in shard_bounds:
            bound.close()
        raise


def close_context(context):
    context["review_bound"].close()
    context["upstream_plan_bound"].close()
    context["confirmation_bound"].close()
    for bound in context["frozen_bounds"].values():
        bound.close()
    for bound in context["shard_bounds"]:
        bound.close()


def verify_context_bindings(context, recovery_plan_bound=None):
    bounds = [
        context["review_bound"],
        context["upstream_plan_bound"],
        context["confirmation_bound"],
        *context["frozen_bounds"].values(),
        *context["shard_bounds"],
    ]
    if recovery_plan_bound is not None:
        bounds.insert(0, recovery_plan_bound)
    for bound in bounds:
        bound.verify()
    observed_runtime = capture_executor_runtime_contract()
    if not type_strict_equal(observed_runtime, context["executor_runtime_contract"]):
        raise RuntimeError("recovery executor runtime changed after binding")
    verify_recovery_parent_binding(
        context["recovery_parent_binding"],
        context["recovery_source_contract"],
    )
    root = Path(context["expected_recovery_plan"]["output_contract"]["root"])
    fit_state = recovery_fit_state(root)
    observed_layout = capture_recovery_layout_binding(
        root,
        context["recovery_source_contract"],
        phase="fit" if context["command"] == "fit" else "planned",
        fit_state=fit_state,
    )
    if not type_strict_equal(observed_layout, context["recovery_layout_binding"]):
        raise RuntimeError("recovery output layout changed after review binding")


def _reserve(args):
    executor_runtime_contract = capture_executor_runtime_contract()
    source_contract = build_recovery_executor_source_contract(
        args.recovery_source_commit,
        args.recovery_source_manifest_sha256,
    )
    root = recovery_root(args.recovery_source_commit)
    if os.fspath(args.recovery_root) != str(root):
        raise ValueError("reservation root aliases immutable derived path")
    parent_binding = ensure_recovery_parent(source_contract, allow_install=True)
    layout_binding = reserve_recovery_layout(root, source_contract, parent_binding)
    statement = expected_review_statement(
        args.recovery_source_commit,
        source_contract,
        executor_runtime_contract,
        layout_binding,
    )
    print(
        json.dumps(
            {
                "audit": RECOVERY_LAYOUT_RECEIPT_AUDIT,
                "recovery_root": str(root),
                "layout_binding": layout_binding,
                "unsigned_review_statement": statement,
                "gpu_status": "NO-GO-pending-external-signature-and-fresh-exact-review",
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _plan(args):
    context = prepare_context(args)
    try:
        root = recovery_root(args.recovery_source_commit)
        raw_root = os.fspath(args.recovery_root)
        raw_plan = os.fspath(args.recovery_plan)
        if raw_root != str(root) or raw_plan != str(root / "recovery_plan.json"):
            raise ValueError(
                "caller recovery root or plan aliases immutable derived path"
            )
        validate_recovery_plan_document(
            context["expected_recovery_plan"],
            context["expected_recovery_plan"],
            args.recovery_source_commit,
        )
        _publish_recovery_plan(root, context["expected_recovery_plan"])
        validate_recovery_layout(root, fit_state="empty")
        print(
            json.dumps(
                {
                    "audit": RECOVERY_PLAN_AUDIT,
                    "plan": str(root / "recovery_plan.json"),
                    "sha256": sha256_file(root / "recovery_plan.json"),
                    "gpu_status": "NO-GO-until-this-exact-plan-and-source-remain-reviewed",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        close_context(context)


def _legacy_candidate_from_fit_payload(fit_payload):
    if type(fit_payload) is not dict or set(fit_payload) != LEGACY_PAYLOAD_KEYS:
        raise ValueError("recovery fit payload schema mismatch")
    return {
        "audit": upstream.CANONICAL_FIT_AUDIT,
        "canonical": True,
        **fit_payload,
    }


def _require_type_strict(value, expected, label):
    if not type_strict_equal(value, expected):
        raise ValueError(f"legacy payload type-strict mismatch: {label}")


def _validate_legacy_state_types(state, label):
    expected_keys = {"down.weight", "down.bias", "up.weight", "up.bias"}
    if type(state) is not collections.OrderedDict or set(state) != expected_keys:
        raise ValueError(f"legacy payload state type mismatch: {label}")
    for name, tensor in state.items():
        if type(tensor) is not torch.Tensor:
            raise ValueError(f"legacy payload tensor type mismatch: {label}.{name}")


def _validate_legacy_fit_report_types(report, fit_budget, label):
    expected_keys = {
        "updates",
        "batch_size",
        "lr",
        "weight_decay",
        "schedule_sha256",
        "first_loss",
        "final_loss",
        "min_loss",
    }
    if type(report) is not dict or set(report) != expected_keys:
        raise ValueError(f"legacy payload fit report type mismatch: {label}")
    for name in ("updates", "batch_size", "lr", "weight_decay"):
        _require_type_strict(report[name], fit_budget[name], f"{label}.{name}")
    _require_type_strict(
        report["schedule_sha256"], fit_budget["schedule_sha256"], f"{label}.schedule"
    )
    for name in ("first_loss", "final_loss", "min_loss"):
        if type(report[name]) is not float or not math.isfinite(report[name]):
            raise ValueError(f"legacy payload loss type mismatch: {label}.{name}")


def _validate_legacy_linear_types(report):
    expected_keys = {
        "train_rows",
        "test_rows",
        "test_correct",
        "test_accuracy",
        "schedule_sha256",
        "claim_boundary",
    }
    if type(report) is not dict or set(report) != expected_keys:
        raise ValueError("legacy payload linear diagnostic type mismatch")
    for name in ("train_rows", "test_rows", "test_correct"):
        if type(report[name]) is not int:
            raise ValueError(f"legacy payload linear count type mismatch: {name}")
    if type(report["test_accuracy"]) is not float or not math.isfinite(
        report["test_accuracy"]
    ):
        raise ValueError("legacy payload linear accuracy type mismatch")
    for name in ("schedule_sha256", "claim_boundary"):
        if type(report[name]) is not str:
            raise ValueError(f"legacy payload linear string type mismatch: {name}")


def _expected_teacher_evidence(fit_payload, context, device):
    plan = context["upstream_plan"]
    assert_reviewed_callable_exports()
    treatment = REVIEWED_MODULE_TO(
        REVIEWED_CARRY_MOTOR(plan["d_model"], plan["fit_budget"]["rank"]),
        device,
    )
    shuffled = REVIEWED_MODULE_TO(
        REVIEWED_CARRY_MOTOR(plan["d_model"], plan["fit_budget"]["rank"]),
        device,
    )
    REVIEWED_MODULE_LOAD_STATE_DICT(treatment, fit_payload["treatment"], strict=True)
    REVIEWED_MODULE_LOAD_STATE_DICT(shuffled, fit_payload["shuffled"], strict=True)
    REVIEWED_MODULE_EVAL(treatment)
    REVIEWED_MODULE_EVAL(shuffled)
    return REVIEWED_CANONICAL_FIT_EVIDENCE(
        context["features"],
        context["board_context"]["rows"],
        context["board_context"]["control_labels"],
        treatment,
        shuffled,
        context["feature_merge"]["teacher_metric_feature_payload_sha256"],
        device,
    )


def validate_legacy_payload_type_strict(fit_payload, context, device):
    if type(fit_payload) is not dict or set(fit_payload) != LEGACY_PAYLOAD_KEYS:
        raise ValueError("legacy recovery fit payload schema mismatch")
    plan = context["upstream_plan"]
    fit_budget = plan["fit_budget"]
    expected_bindings = context["expected_bindings"]
    parameter_count = (
        plan["d_model"] * fit_budget["rank"]
        + fit_budget["rank"]
        + 2 * fit_budget["rank"]
        + 2
    )
    expected_exact = {
        "plan_sha256": UPSTREAM_PLAN_SHA256,
        "base_checkpoint_sha256": expected_bindings["base_checkpoint_sha256"],
        "tokenizer_sha256": expected_bindings["tokenizer_sha256"],
        "episodes_sha256": expected_bindings["episodes_sha256"],
        "cycle_sha256": expected_bindings["cycle_sha256"],
        "confirmation_commitment_sha256": expected_bindings[
            "confirmation_commitment_sha256"
        ],
        "scientific_source_sha256": context["upstream_source_hashes"],
        "source_contract": context["upstream_source_contract"],
        "checkpoint_step": plan["checkpoint_step"],
        "d_model": plan["d_model"],
        "rank": fit_budget["rank"],
        "parameter_count": parameter_count,
        "extract_batch": plan["runtime_contract"]["extract_batch"],
        "feature_shard_merge": context["feature_merge"],
        "deployment_logit_dtype": context["features"]["deployment_logit_dtype"],
        "zero_id": plan["zero_id"],
        "one_id": plan["one_id"],
        "initial_state_sha256": fit_budget["initial_state_sha256"],
        "board": context["board_context"]["normalized_board"],
        "control": context["board_context"]["control"],
        "claim_boundary": upstream.CANONICAL_FIT_CLAIM_BOUNDARY,
    }
    for name, expected in expected_exact.items():
        _require_type_strict(fit_payload[name], expected, name)
    for arm in ("treatment", "shuffled"):
        _validate_legacy_state_types(fit_payload[arm], arm)
        state_sha256 = fit_payload[f"{arm}_state_sha256"]
        if type(state_sha256) is not str or not re.fullmatch(
            r"[0-9a-f]{64}", state_sha256
        ):
            raise ValueError(f"legacy payload state receipt type mismatch: {arm}")
        _validate_legacy_fit_report_types(fit_payload[f"{arm}_fit"], fit_budget, arm)
    _validate_legacy_linear_types(fit_payload["linear_diagnostic"])
    expected_evidence = _expected_teacher_evidence(fit_payload, context, device)
    _require_type_strict(
        fit_payload["fit_feature_metrics"],
        expected_evidence,
        "fit_feature_metrics",
    )
    covered = set(expected_exact) | {
        "treatment",
        "shuffled",
        "treatment_state_sha256",
        "shuffled_state_sha256",
        "treatment_fit",
        "shuffled_fit",
        "linear_diagnostic",
        "fit_feature_metrics",
    }
    if covered != LEGACY_PAYLOAD_KEYS:
        raise AssertionError("legacy payload type validator coverage drift")


def validate_constant_bias_payload(payload, fit_payload, context):
    if type(payload) is not dict or set(payload) != CONSTANT_BIAS_PAYLOAD_KEYS:
        raise ValueError("constant-bias payload schema mismatch")
    fit_budget = context["upstream_plan"]["fit_budget"]
    expected_initial, expected_initial_sha256 = initial_constant_bias_state()
    expected_exact = {
        "schema": CONSTANT_BIAS_PAYLOAD_SCHEMA,
        "control_id": CONSTANT_BIAS_CONTROL_ID,
        "parameterization": constant_bias_parameterization(),
        "training_rows": len(context["features"]["labels"]),
        "training_feature_payload_sha256": context["feature_merge"][
            "teacher_metric_feature_payload_sha256"
        ],
        "label_source": "features.labels_true",
        "initial_state_sha256": expected_initial_sha256,
        "claim_boundary": CONSTANT_BIAS_CLAIM_BOUNDARY,
    }
    for name, expected in expected_exact.items():
        _require_type_strict(payload[name], expected, f"constant_bias.{name}")
    state = payload["state"]
    if type(state) is not collections.OrderedDict or tuple(state) != ("delta",):
        raise ValueError("constant-bias state must contain only delta")
    delta = state["delta"]
    if (
        type(delta) is not torch.Tensor
        or delta.dtype != torch.float32
        or tuple(delta.shape) != ()
        or not bool(REVIEWED_ISFINITE(delta))
    ):
        raise ValueError("constant-bias delta state is malformed")
    if not re.fullmatch(r"[0-9a-f]{64}", payload["state_sha256"]):
        raise ValueError("constant-bias state receipt is malformed")
    if payload["state_sha256"] != REVIEWED_TENSOR_STATE_SHA256(state):
        raise ValueError("constant-bias state receipt mismatch")
    if expected_initial["delta"].item() != 0.0:
        raise AssertionError("constant-bias initial state is not zero")
    expected_motor, expected_fit = _recovery_fit_constant_bias(
        context["features"],
        context["features"]["labels"],
        expected_initial,
        "cpu",
        fit_budget["updates"],
        fit_budget["batch_size"],
        fit_budget["lr"],
        fit_budget["weight_decay"],
        fit_budget["seed"],
        fit_rows=canonical_fit_selection_rows(context),
        fit_model_binding=_context_fit_model_binding(context),
    )
    if not type_strict_equal(payload["fit"], expected_fit):
        raise ValueError("constant_bias full-board selection evidence mismatch")
    if not type_strict_equal(state, REVIEWED_MODULE_STATE_DICT(expected_motor)):
        raise ValueError("constant_bias selected state is not the converged checkpoint")
    if (
        payload["fit"].get("converged") is not True
        or payload["fit"].get("selected_is_final_iterate") is not False
        or payload["fit"].get("optimizer_final_iterate_used") is not False
    ):
        raise ValueError("constant_bias selected checkpoint is not converged")
    expected_diagnostic = raw_carry_margin_diagnostic(
        context["features"]["base01"],
        context["features"]["labels"],
        float(delta.item()),
    )
    _require_type_strict(
        payload["raw_margin_diagnostic"],
        expected_diagnostic,
        "constant_bias.raw_margin_diagnostic",
    )


def validate_nuisance_only_payload(payload, fit_payload, context):
    if type(payload) is not dict or set(payload) != NUISANCE_ONLY_PAYLOAD_KEYS:
        raise ValueError("nuisance-only payload schema mismatch")
    fit_budget = context["upstream_plan"]["fit_budget"]
    rows = context["board_context"]["rows"]
    if len(rows) != len(context["features"]["labels"]):
        raise ValueError("nuisance-only training row count mismatch")
    nuisance_metadata_from_rows(rows)
    widths = sorted({row["width"] for row in rows})
    if widths != list(NUISANCE_TRAIN_WIDTHS):
        raise ValueError("nuisance-only fit accessed nontraining widths")
    expected_initial, expected_initial_sha256 = initial_nuisance_only_state()
    expected_exact = {
        "schema": NUISANCE_ONLY_PAYLOAD_SCHEMA,
        "control_id": NUISANCE_ONLY_CONTROL_ID,
        "parameterization": nuisance_only_parameterization(),
        "training_rows": len(context["features"]["labels"]),
        "training_feature_payload_sha256": context["feature_merge"][
            "teacher_metric_feature_payload_sha256"
        ],
        "training_metadata_receipt": nuisance_metadata_receipt(rows),
        "capacity_ledger": nuisance_capacity_ledger(),
        "label_source": "features.labels_true",
        "initial_state_sha256": expected_initial_sha256,
        "width_ood_policy": nuisance_width_ood_policy(),
        "claim_boundary": NUISANCE_ONLY_CLAIM_BOUNDARY,
    }
    for name, expected in expected_exact.items():
        _require_type_strict(payload[name], expected, f"nuisance_only.{name}")
    state = payload["state"]
    if type(state) is not collections.OrderedDict or tuple(state) != ("cell_delta",):
        raise ValueError("nuisance-only state may contain only saturated cell deltas")
    expected_shapes = {"cell_delta": (NUISANCE_PARAMETER_COUNT,)}
    for name, shape in expected_shapes.items():
        tensor = state[name]
        if (
            type(tensor) is not torch.Tensor
            or tensor.dtype != torch.float32
            or tuple(tensor.shape) != shape
            or not bool(REVIEWED_ISFINITE(tensor).all())
        ):
            raise ValueError(f"nuisance-only state is malformed: {name}")
    if any(bool(value.ne(0).any()) for value in expected_initial.values()):
        raise AssertionError("nuisance-only initial state is not zero")
    if not re.fullmatch(r"[0-9a-f]{64}", payload["state_sha256"]):
        raise ValueError("nuisance-only state receipt is malformed")
    if payload["state_sha256"] != REVIEWED_TENSOR_STATE_SHA256(state):
        raise ValueError("nuisance-only state receipt mismatch")
    expected_motor, expected_fit = _recovery_fit_nuisance_only(
        context["features"],
        canonical_fit_selection_rows(context),
        context["features"]["labels"],
        expected_initial,
        "cpu",
        fit_budget["updates"],
        fit_budget["batch_size"],
        fit_budget["lr"],
        fit_budget["weight_decay"],
        fit_budget["seed"],
        fit_model_binding=_context_fit_model_binding(context),
    )
    if not type_strict_equal(payload["fit"], expected_fit):
        raise ValueError("nuisance_only full-board selection evidence mismatch")
    if not type_strict_equal(state, REVIEWED_MODULE_STATE_DICT(expected_motor)):
        raise ValueError("nuisance_only selected state is not the converged checkpoint")
    if (
        payload["fit"].get("converged") is not True
        or payload["fit"].get("selected_is_final_iterate") is not False
        or payload["fit"].get("optimizer_final_iterate_used") is not False
    ):
        raise ValueError("nuisance_only selected checkpoint is not converged")


def validate_recovery_fit_bundle(
    bundle,
    recovery_plan,
    recovery_plan_sha256,
    context,
    checkpoint,
    device,
):
    if type(bundle) is not dict or set(bundle) != RECOVERY_FIT_KEYS:
        raise ValueError("v11 recovery fit schema mismatch")
    if bundle.get("audit") != RECOVERY_FIT_AUDIT or bundle.get("recovery") is not True:
        raise ValueError("fit is not a v11 recovery artifact")
    if "canonical" in bundle or bundle.get("audit") == upstream.CANONICAL_FIT_AUDIT:
        raise ValueError("recovery fit may never publish as v8 canonical")
    expected = {
        "recovery_plan_sha256": recovery_plan_sha256,
        "recovery_executor_source_contract": recovery_plan[
            "recovery_executor_source_contract"
        ],
        "executor_runtime_contract": recovery_plan["executor_runtime_contract"],
        "recovery_parent_binding": recovery_plan["recovery_parent_binding"],
        "recovery_layout_binding": recovery_plan["output_contract"]["layout_binding"],
        "upstream_protocol_source_contract": recovery_plan["upstream_protocol"][
            "source_contract"
        ],
        "upstream_plan_binding": recovery_plan["upstream_protocol"]["plan_binding"],
        "upstream_shard_receipts": recovery_plan["upstream_protocol"]["shard_receipts"],
        "normalization_proof": recovery_plan["normalization_proof"],
        "allowed_transformation": ALLOWED_TRANSFORMATION,
        "deserialization_contract": DESERIALIZATION_CONTRACT,
        "parameter_ledger": validate_deployment_parameter_ledger(
            bundle["parameter_ledger"],
            checkpoint["cfg"],
            context["upstream_plan"]["fit_budget"]["rank"],
        ),
        "claim_boundary": RECOVERY_FIT_CLAIM_BOUNDARY,
    }
    for name, value in expected.items():
        if not type_strict_equal(bundle.get(name), value):
            raise ValueError(f"recovery fit provenance mismatch: {name}")
    validate_slurm_h100_attestation(bundle.get("slurm_h100_attestation"))
    if not type_strict_equal(
        bundle["parameter_ledger"],
        recovery_plan["fit_contract"]["parameter_ledger"],
    ):
        raise ValueError("recovery fit parameter ledger differs from signed plan")
    validate_legacy_payload_type_strict(bundle["fit_payload"], context, device)
    validate_constant_bias_payload(
        bundle["constant_bias_payload"], bundle["fit_payload"], context
    )
    validate_nuisance_only_payload(
        bundle["nuisance_only_payload"], bundle["fit_payload"], context
    )
    expected_trajectory_proof = build_trajectory_replay_proof(
        bundle["fit_payload"],
        bundle["constant_bias_payload"],
        bundle["nuisance_only_payload"],
    )
    if not type_strict_equal(
        bundle.get("trajectory_replay_proof"), expected_trajectory_proof
    ):
        raise ValueError("recovery trajectory replay proof schema mismatch")
    legacy = _legacy_candidate_from_fit_payload(bundle["fit_payload"])
    assert_reviewed_callable_exports()
    REVIEWED_VALIDATE_MOTOR_BUNDLE(
        legacy,
        context["expected_bindings"],
        context["upstream_source_hashes"],
        context["upstream_source_contract"],
        UPSTREAM_PLAN_SHA256,
        context["upstream_plan"],
        context["features"],
        context["feature_merge"],
        device,
    )


def _recovery_fit_motor(
    features,
    labels,
    initial_state,
    device,
    updates,
    batch_size,
    lr,
    weight_decay,
    seed,
):
    """Recovery-owned replay of the frozen a0c258e optimizer trajectory."""
    assert_reviewed_callable_exports()
    motor = REVIEWED_MODULE_TO(
        REVIEWED_CARRY_MOTOR(features["hidden"].shape[1], upstream.RANK), device
    )
    REVIEWED_MODULE_LOAD_STATE_DICT(motor, initial_state)
    optimizer = REVIEWED_ADAMW(
        REVIEWED_MODULE_PARAMETERS(motor), lr=lr, weight_decay=weight_decay
    )
    schedule, schedule_sha256 = REVIEWED_BATCH_SCHEDULE(
        len(labels), batch_size, updates, seed
    )
    x = features["hidden"].to(device)
    base01 = features["base01"].to(device)
    other_lse = features["other_lse"].to(device)
    targets = REVIEWED_AS_TENSOR(labels, dtype=torch.long, device=device)
    losses = []
    for update, batch_cpu in enumerate(schedule, 1):
        batch = batch_cpu.to(device)
        REVIEWED_OPTIMIZER_ZERO_GRAD(optimizer, set_to_none=True)
        loss = REVIEWED_FULL_VOCAB_MOTOR_LOSS(
            x[batch], base01[batch], other_lse[batch], targets[batch], motor
        )
        if not REVIEWED_ISFINITE(loss):
            raise FloatingPointError("non-finite recovery motor loss")
        REVIEWED_TENSOR_BACKWARD(loss)
        REVIEWED_ADAMW_STEP(optimizer)
        losses.append(float(loss.detach().cpu()))
        if update == 1 or update % 200 == 0 or update == updates:
            print(
                f"[carry-recovery-replay] update={update}/{updates} "
                f"loss={losses[-1]:.6f}",
                flush=True,
            )
    assert_reviewed_callable_exports()
    return REVIEWED_MODULE_CPU(motor), {
        "updates": updates,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "schedule_sha256": schedule_sha256,
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "min_loss": min(losses),
    }


def _full_board_scalar_statistics(base01, other_lse, labels, indices, delta):
    if type(indices) is not list or not indices:
        raise ValueError("full-board null optimization has a missing metadata cell")
    selected = REVIEWED_TENSOR(indices, dtype=torch.long)
    pair = base01.detach().cpu()[selected]
    other = other_lse.detach().cpu()[selected]
    targets = labels.detach().cpu()[selected]
    if (
        pair.ndim != 2
        or pair.shape[1] != 2
        or pair.dtype != torch.bfloat16
        or other.ndim != 1
        or targets.ndim != 1
        or len(pair) != len(other)
        or len(pair) != len(targets)
        or not bool(REVIEWED_ISFINITE(pair.float()).all())
        or not bool(REVIEWED_ISFINITE(other.float()).all())
        or not bool(((targets == 0) | (targets == 1)).all())
        or type(delta) is not float
        or not math.isfinite(delta)
    ):
        raise ValueError("full-board null optimization inputs are malformed")
    scalar = REVIEWED_TENSOR(delta, dtype=torch.float32, requires_grad=True)
    pair_delta = REVIEWED_STACK((-scalar / 2.0, scalar / 2.0)).expand(len(pair), 2)
    adjusted = (pair + pair_delta.to(pair.dtype)).float()
    carry_lse = REVIEWED_LOGSUMEXP(adjusted, dim=-1)
    total_lse = REVIEWED_LOGADDEXP(other.float(), carry_lse)
    target_logits = adjusted.gather(1, targets.long().unsqueeze(1)).squeeze(1)
    loss = (total_lse - target_logits).mean()
    if not bool(REVIEWED_ISFINITE(loss)):
        raise FloatingPointError("non-finite full-board null objective")
    REVIEWED_TENSOR_BACKWARD(loss)
    gradient = scalar.grad
    if gradient is None or not bool(REVIEWED_ISFINITE(gradient)):
        raise FloatingPointError("non-finite full-board null gradient")
    return float(loss.detach()), float(gradient.detach())


def _full_board_selected_state_cross_entropy(
    base01, other_lse, labels, groups, selected_values
):
    if len(groups) != len(selected_values):
        raise ValueError("full-board selected state and metadata groups differ")
    row_delta = [None] * len(labels)
    for (_stratum_id, indices), delta in zip(groups, selected_values):
        for index in indices:
            if row_delta[index] is not None:
                raise ValueError("full-board selected state has a duplicate row")
            row_delta[index] = delta
    if any(value is None for value in row_delta):
        raise ValueError("full-board selected state has a missing row")
    delta = REVIEWED_TENSOR(row_delta, dtype=torch.float32)
    pair_delta = REVIEWED_STACK((-delta / 2.0, delta / 2.0), dim=-1)
    pair = base01.detach().cpu()
    other = other_lse.detach().cpu()
    targets = labels.detach().cpu()
    adjusted = (pair + pair_delta.to(pair.dtype)).float()
    carry_lse = REVIEWED_LOGSUMEXP(adjusted, dim=-1)
    total_lse = REVIEWED_LOGADDEXP(other.float(), carry_lse)
    target_logits = adjusted.gather(1, targets.long().unsqueeze(1)).squeeze(1)
    loss = (total_lse - target_logits).mean()
    if not bool(REVIEWED_ISFINITE(loss)):
        raise FloatingPointError("non-finite selected full-board null objective")
    return float(loss.detach())


def _deployment_scalar_candidates(checkpoints, deployment_dtype):
    if deployment_dtype != torch.bfloat16:
        raise ValueError("unsupported null deployment logit dtype")
    candidates = set()
    for value, _loss, _gradient in checkpoints:
        candidates.add(float(REVIEWED_TENSOR(value, dtype=torch.float32).item()))
        for sign in (1.0, -1.0):
            half = REVIEWED_TENSOR(sign * value / 2.0, dtype=torch.float32).to(
                deployment_dtype
            )
            lower = REVIEWED_NEXTAFTER(
                half, REVIEWED_TENSOR(-math.inf, dtype=deployment_dtype)
            )
            upper = REVIEWED_NEXTAFTER(
                half, REVIEWED_TENSOR(math.inf, dtype=deployment_dtype)
            )
            for deployed_half in (lower, half, upper):
                candidate = sign * 2.0 * float(deployed_half.float().item())
                if math.isfinite(candidate) and (
                    -NULL_OPTIMIZATION_BOUND <= candidate <= NULL_OPTIMIZATION_BOUND
                ):
                    candidates.add(
                        float(REVIEWED_TENSOR(candidate, dtype=torch.float32).item())
                    )
    return sorted(candidates)


def _solve_full_board_scalar_groups(
    features,
    labels,
    groups,
    control_id,
    fit_rows,
    fit_model_binding,
):
    if (
        type(features) is not dict
        or type(features.get("base01")) is not torch.Tensor
        or type(features.get("other_lse")) is not torch.Tensor
        or type(labels) is not torch.Tensor
        or type(groups) is not list
        or not groups
        or type(control_id) is not str
        or not control_id
        or type(fit_rows) is not list
        or len(fit_rows) != len(labels)
        or type(fit_model_binding) is not dict
        or not fit_model_binding
    ):
        raise ValueError("full-board null optimization contract is malformed")
    if len(features["base01"]) != len(labels) or len(features["other_lse"]) != len(
        labels
    ):
        raise ValueError("full-board null optimization row counts differ")
    if features["base01"].dtype != torch.bfloat16:
        raise ValueError("full-board null deployment logits must be bfloat16")
    covered = []
    selected_values = []
    strata = []
    for group_index, item in enumerate(groups):
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or not item[0]
            or type(item[1]) is not list
            or any(type(index) is not int for index in item[1])
            or len(set(item[1])) != len(item[1])
            or any(not 0 <= index < len(labels) for index in item[1])
        ):
            raise ValueError("full-board null optimization metadata cell is malformed")
        stratum_id, indices = item
        if not indices:
            raise ValueError("full-board null optimization has a missing metadata cell")
        covered.extend(indices)
        target_values = [int(labels[index]) for index in indices]
        target_0_rows = target_values.count(0)
        target_1_rows = target_values.count(1)
        if target_0_rows == 0 or target_1_rows == 0:
            raise ValueError(
                "full-board null optimization requires both targets in every cell"
            )
        low = -NULL_OPTIMIZATION_BOUND
        high = NULL_OPTIMIZATION_BOUND
        low_loss, low_gradient = _full_board_scalar_statistics(
            features["base01"], features["other_lse"], labels, indices, low
        )
        high_loss, high_gradient = _full_board_scalar_statistics(
            features["base01"], features["other_lse"], labels, indices, high
        )
        boundary = None
        checkpoints = []
        if low_gradient >= 0.0:
            boundary = "lower"
            checkpoints = [(low, low_loss, low_gradient)]
        elif high_gradient <= 0.0:
            boundary = "upper"
            checkpoints = [(high, high_loss, high_gradient)]
        else:
            for _step in range(NULL_OPTIMIZATION_BISECTION_STEPS):
                midpoint = low + (high - low) / 2.0
                midpoint_loss, midpoint_gradient = _full_board_scalar_statistics(
                    features["base01"],
                    features["other_lse"],
                    labels,
                    indices,
                    midpoint,
                )
                if midpoint_gradient <= 0.0:
                    low, low_loss, low_gradient = (
                        midpoint,
                        midpoint_loss,
                        midpoint_gradient,
                    )
                else:
                    high, high_loss, high_gradient = (
                        midpoint,
                        midpoint_loss,
                        midpoint_gradient,
                    )
            midpoint = low + (high - low) / 2.0
            midpoint_loss, midpoint_gradient = _full_board_scalar_statistics(
                features["base01"],
                features["other_lse"],
                labels,
                indices,
                midpoint,
            )
            checkpoints = [
                (low, low_loss, low_gradient),
                (midpoint, midpoint_loss, midpoint_gradient),
                (high, high_loss, high_gradient),
            ]
        float32_candidates = _deployment_scalar_candidates(
            checkpoints, features["base01"].dtype
        )
        rescored = []
        for value in float32_candidates:
            objective, gradient = _full_board_scalar_statistics(
                features["base01"],
                features["other_lse"],
                labels,
                indices,
                value,
            )
            rescored.append(
                {
                    "delta": value,
                    "full_vocab_cross_entropy": objective,
                    "gradient": gradient,
                }
            )
        selected = min(
            rescored,
            key=lambda candidate: (
                candidate["full_vocab_cross_entropy"],
                abs(candidate["delta"]),
                candidate["delta"],
            ),
        )
        if boundary is None:
            converged = (
                low_gradient <= 0.0 <= high_gradient
                and high - low <= NULL_OPTIMIZATION_BRACKET_TOLERANCE
            )
        elif boundary == "lower":
            converged = low_gradient >= 0.0
        else:
            converged = high_gradient <= 0.0
        if not converged:
            raise RuntimeError(
                "full-board null optimization did not meet frozen convergence rule"
            )
        selected_values.append(selected["delta"])
        strata.append(
            {
                "stratum_id": stratum_id,
                "parameter_index": group_index,
                "rows": len(indices),
                "target_0_rows": target_0_rows,
                "target_1_rows": target_1_rows,
                "selected_delta": selected["delta"],
                "full_vocab_cross_entropy": selected["full_vocab_cross_entropy"],
                "selected_gradient": selected["gradient"],
                "boundary_solution": boundary,
                "root_bracket": {
                    "lower": low,
                    "lower_gradient": low_gradient,
                    "upper": high,
                    "upper_gradient": high_gradient,
                },
                "checkpoint_candidates": rescored,
                "checkpoint_candidates_sha256": stable_json_sha256(rescored),
                "converged": True,
            }
        )
    if sorted(covered) != list(range(len(labels))):
        raise ValueError(
            "full-board null optimization has duplicate, missing, or extra rows"
        )
    fit_case_payload = {
        "control_id": control_id,
        "base01": features["base01"].detach().cpu(),
        "other_lse": features["other_lse"].detach().cpu(),
        "labels": labels.detach().cpu(),
        "groups": groups,
        "fit_rows": fit_rows,
        "fit_model_binding": fit_model_binding,
    }
    full_board_cross_entropy = _full_board_selected_state_cross_entropy(
        features["base01"],
        features["other_lse"],
        labels,
        groups,
        selected_values,
    )
    evidence = {
        "schema": NULL_OPTIMIZATION_SCHEMA,
        "control_id": control_id,
        "objective": "mean exact full-board full-vocabulary cross entropy",
        "solver": "convex scalar derivative bracketing",
        "deployment_logit_dtype": str(features["base01"].dtype),
        "candidate_neighborhood": (
            "root bracket plus adjacent deployed-dtype levels for both signed "
            "half-deltas"
        ),
        "state_bound": [-NULL_OPTIMIZATION_BOUND, NULL_OPTIMIZATION_BOUND],
        "bisection_steps": NULL_OPTIMIZATION_BISECTION_STEPS,
        "gradient_tolerance": NULL_OPTIMIZATION_GRADIENT_TOLERANCE,
        "bracket_tolerance": NULL_OPTIMIZATION_BRACKET_TOLERANCE,
        "rows": len(labels),
        "parameters": len(groups),
        "fit_case_payload_sha256": scientific_tree_sha256(fit_case_payload),
        "fit_rows_sha256": stable_json_sha256(fit_rows),
        "fit_model_binding_sha256": stable_json_sha256(fit_model_binding),
        "full_board_cross_entropy": full_board_cross_entropy,
        "full_board_cross_entropy_sha256": stable_json_sha256(
            {"value": full_board_cross_entropy}
        ),
        "metadata_strata": strata,
        "metadata_strata_sha256": stable_json_sha256(strata),
        "selection_rule": (
            "minimum recomputed full-board CE, then minimum absolute delta, "
            "then lowest float32 delta"
        ),
        "selected_is_final_iterate": False,
        "optimizer_final_iterate_used": False,
        "width8_selection_access": False,
        "confirmation_selection_access": False,
        "converged": True,
    }
    return selected_values, evidence


def _finalize_null_optimization_evidence(evidence, state):
    finalized = copy.deepcopy(evidence)
    finalized["selected_state_sha256"] = REVIEWED_TENSOR_STATE_SHA256(state)
    finalized["selected_state_scientific_sha256"] = scientific_tree_sha256(state)
    finalized["selected_checkpoint_evidence_sha256"] = stable_json_sha256(evidence)
    return canonical_json_document(finalized)


def _unpublished_fit_model_binding(features, fit_rows):
    return {
        "schema": "carry_motor_unpublished_direct_tensor_binding_v1",
        "fit_rows_sha256": stable_json_sha256(fit_rows),
        "raw_tensor_sha256": scientific_tree_sha256(
            {
                "base01": features["base01"],
                "other_lse": features["other_lse"],
            }
        ),
        "publication_eligible": False,
    }


def _recovery_fit_constant_bias(
    features,
    labels,
    initial_state,
    device,
    updates,
    batch_size,
    lr,
    weight_decay,
    seed,
    *,
    fit_rows=None,
    fit_model_binding=None,
):
    """Solve the favorable scalar null on the complete full-vocabulary board."""
    assert_reviewed_callable_exports()
    del device, updates, batch_size, lr, weight_decay, seed
    if tuple(initial_state) != ("delta",) or initial_state["delta"].item() != 0.0:
        raise ValueError("constant-bias solver requires the frozen zero state")
    if fit_rows is None:
        fit_rows = [{"row_index": index} for index in range(len(labels))]
    if fit_model_binding is None:
        fit_model_binding = _unpublished_fit_model_binding(features, fit_rows)
    selected, evidence = _solve_full_board_scalar_groups(
        features,
        labels,
        [("all_fit_rows", list(range(len(labels))))],
        "constant_bias",
        fit_rows,
        fit_model_binding,
    )
    motor = ConstantBiasMotor()
    with REVIEWED_TORCH_NO_GRAD():
        motor.delta.copy_(REVIEWED_TENSOR(selected[0], dtype=torch.float32))
    state = REVIEWED_MODULE_STATE_DICT(motor)
    assert_reviewed_callable_exports()
    return motor, _finalize_null_optimization_evidence(evidence, state)


def _recovery_fit_nuisance_only(
    features,
    rows,
    labels,
    initial_state,
    device,
    updates,
    batch_size,
    lr,
    weight_decay,
    seed,
    *,
    fit_model_binding=None,
):
    """Solve every saturated fit cell on the complete full-vocabulary board."""
    assert_reviewed_callable_exports()
    del device, updates, batch_size, lr, weight_decay, seed
    metadata = nuisance_metadata_from_rows(rows)
    if sorted({row["width"] for row in rows}) != list(NUISANCE_TRAIN_WIDTHS):
        raise ValueError("nuisance-only fit may use only frozen training widths")
    if len(metadata) != len(labels):
        raise ValueError("nuisance-only metadata and labels differ in length")
    if tuple(initial_state) != ("cell_delta",) or bool(
        initial_state["cell_delta"].ne(0).any()
    ):
        raise ValueError("nuisance-only solver requires the frozen zero state")
    if fit_model_binding is None:
        fit_model_binding = _unpublished_fit_model_binding(features, rows)
    groups = []
    for operation, width, position in NUISANCE_FIT_CELLS:
        indices = [
            index
            for index, row in enumerate(rows)
            if (
                row["operation"],
                row["width"],
                row["position"],
            )
            == (operation, width, position)
        ]
        groups.append((f"{operation}-w{width}-p{position}", indices))
    selected, evidence = _solve_full_board_scalar_groups(
        features,
        labels,
        groups,
        "nuisance_only",
        rows,
        fit_model_binding,
    )
    motor = NuisanceOnlyMotor()
    with REVIEWED_TORCH_NO_GRAD():
        motor.cell_delta.copy_(REVIEWED_TENSOR(selected, dtype=torch.float32))
    state = REVIEWED_MODULE_STATE_DICT(motor)
    assert_reviewed_callable_exports()
    return motor, _finalize_null_optimization_evidence(evidence, state)


def _recovery_fit_linear_diagnostic(features, labels, device, seed):
    """Recovery-owned replay of the frozen non-deployment linear diagnostic."""
    assert_reviewed_callable_exports()
    size = len(labels)
    generator = REVIEWED_TORCH_GENERATOR().manual_seed(seed)
    order = REVIEWED_RANDPERM(size, generator=generator)
    split = int(size * 0.8)
    train_indices, test_indices = order[:split], order[split:]
    classifier = REVIEWED_MODULE_TO(
        REVIEWED_LINEAR(features["hidden"].shape[1], 2), device
    )
    optimizer = REVIEWED_ADAMW(
        REVIEWED_MODULE_PARAMETERS(classifier), lr=1e-2, weight_decay=1e-4
    )
    x = features["hidden"].to(device)
    y = REVIEWED_AS_TENSOR(labels, dtype=torch.long, device=device)
    schedule, schedule_sha256 = REVIEWED_BATCH_SCHEDULE(
        split, min(512, split), 300, seed
    )
    train_indices_device = train_indices.to(device)
    for batch in schedule:
        selected = train_indices_device[batch.to(device)]
        REVIEWED_OPTIMIZER_ZERO_GRAD(optimizer, set_to_none=True)
        loss = REVIEWED_CROSS_ENTROPY(classifier(x[selected]), y[selected])
        REVIEWED_TENSOR_BACKWARD(loss)
        REVIEWED_ADAMW_STEP(optimizer)
    with REVIEWED_TORCH_NO_GRAD():
        prediction = classifier(x[test_indices.to(device)]).argmax(dim=-1)
        correct = int((prediction == y[test_indices.to(device)]).sum().cpu())
    assert_reviewed_callable_exports()
    return {
        "train_rows": split,
        "test_rows": size - split,
        "test_correct": correct,
        "test_accuracy": correct / (size - split),
        "schedule_sha256": schedule_sha256,
        "claim_boundary": upstream.LINEAR_DIAGNOSTIC_CLAIM_BOUNDARY,
    }


def _build_fit_payload(context, checkpoint, device):
    assert_reviewed_callable_exports()
    plan = context["upstream_plan"]
    board_context = context["board_context"]
    features = context["features"]
    fit_budget = plan["fit_budget"]
    cfg = REVIEWED_GPT_CONFIG(**checkpoint["cfg"])
    parameter_ledger = deployment_parameter_ledger(
        checkpoint["cfg"], fit_budget["rank"]
    )
    if (
        checkpoint.get("step") != plan["checkpoint_step"]
        or int(cfg.n_loop) != 1
        or int(cfg.d_model) != plan["d_model"]
        or int(cfg.vocab_size) != plan["vocab_size"]
        or not type_strict_equal(
            parameter_ledger,
            context["expected_recovery_plan"]["fit_contract"]["parameter_ledger"],
        )
    ):
        raise ValueError("checkpoint differs from frozen upstream fit contract")
    REVIEWED_MANUAL_SEED(fit_budget["seed"])
    REVIEWED_CUDA_MANUAL_SEED_ALL(fit_budget["seed"])
    initial_state, initial_state_sha256 = REVIEWED_INITIAL_MOTOR_STATE(plan["d_model"])
    if (
        initial_state_sha256 != board_context["initial_state_sha256"]
        or initial_state_sha256 != fit_budget["initial_state_sha256"]
    ):
        raise ValueError("fit-time initial motor state differs from frozen plan")
    treatment, treatment_fit = _recovery_fit_motor(
        features,
        features["labels"],
        initial_state,
        device,
        fit_budget["updates"],
        fit_budget["batch_size"],
        fit_budget["lr"],
        fit_budget["weight_decay"],
        fit_budget["seed"],
    )
    shuffled, shuffled_fit = _recovery_fit_motor(
        features,
        board_context["control_labels"],
        initial_state,
        device,
        fit_budget["updates"],
        fit_budget["batch_size"],
        fit_budget["lr"],
        fit_budget["weight_decay"],
        fit_budget["seed"],
    )
    if treatment_fit["schedule_sha256"] != shuffled_fit["schedule_sha256"]:
        raise RuntimeError("recovery arms used different fit schedules")
    linear = _recovery_fit_linear_diagnostic(
        features, features["labels"], device, upstream.FIT_SEED + 2
    )
    treatment = REVIEWED_MODULE_EVAL(REVIEWED_MODULE_TO(treatment, device))
    shuffled = REVIEWED_MODULE_EVAL(REVIEWED_MODULE_TO(shuffled, device))
    evidence = REVIEWED_CANONICAL_FIT_EVIDENCE(
        features,
        board_context["rows"],
        board_context["control_labels"],
        treatment,
        shuffled,
        context["feature_merge"]["teacher_metric_feature_payload_sha256"],
        device,
    )
    treatment = REVIEWED_MODULE_CPU(treatment)
    shuffled = REVIEWED_MODULE_CPU(shuffled)
    treatment_state = REVIEWED_MODULE_STATE_DICT(treatment)
    shuffled_state = REVIEWED_MODULE_STATE_DICT(shuffled)
    expected_bindings = {
        key: value
        for key, value in context["expected_bindings"].items()
        if key != "scientific_source_sha256"
    }
    return {
        "plan_sha256": UPSTREAM_PLAN_SHA256,
        **expected_bindings,
        "scientific_source_sha256": context["upstream_source_hashes"],
        "source_contract": context["upstream_source_contract"],
        "checkpoint_step": checkpoint.get("step"),
        "d_model": int(cfg.d_model),
        "rank": fit_budget["rank"],
        "parameter_count": treatment.parameter_count(),
        "extract_batch": plan["runtime_contract"]["extract_batch"],
        "feature_shard_merge": context["feature_merge"],
        "deployment_logit_dtype": features["deployment_logit_dtype"],
        "zero_id": plan["zero_id"],
        "one_id": plan["one_id"],
        "initial_state_sha256": board_context["initial_state_sha256"],
        "treatment": treatment_state,
        "shuffled": shuffled_state,
        "treatment_state_sha256": REVIEWED_TENSOR_STATE_SHA256(treatment_state),
        "shuffled_state_sha256": REVIEWED_TENSOR_STATE_SHA256(shuffled_state),
        "board": board_context["normalized_board"],
        "control": board_context["control"],
        "treatment_fit": treatment_fit,
        "shuffled_fit": shuffled_fit,
        "linear_diagnostic": linear,
        "fit_feature_metrics": evidence,
        "claim_boundary": upstream.CANONICAL_FIT_CLAIM_BOUNDARY,
    }


def _build_constant_bias_payload(context, device):
    features = context["features"]
    fit_budget = context["upstream_plan"]["fit_budget"]
    initial_state, initial_state_sha256 = initial_constant_bias_state()
    fit_model_binding = _context_fit_model_binding(context)
    motor, fit_report = _recovery_fit_constant_bias(
        features,
        features["labels"],
        initial_state,
        device,
        fit_budget["updates"],
        fit_budget["batch_size"],
        fit_budget["lr"],
        fit_budget["weight_decay"],
        fit_budget["seed"],
        fit_rows=canonical_fit_selection_rows(context),
        fit_model_binding=fit_model_binding,
    )
    state = REVIEWED_MODULE_STATE_DICT(motor)
    delta = float(state["delta"].item())
    return {
        "schema": CONSTANT_BIAS_PAYLOAD_SCHEMA,
        "control_id": CONSTANT_BIAS_CONTROL_ID,
        "parameterization": constant_bias_parameterization(),
        "training_rows": len(features["labels"]),
        "training_feature_payload_sha256": context["feature_merge"][
            "teacher_metric_feature_payload_sha256"
        ],
        "label_source": "features.labels_true",
        "initial_state_sha256": initial_state_sha256,
        "state": state,
        "state_sha256": REVIEWED_TENSOR_STATE_SHA256(state),
        "fit": fit_report,
        "raw_margin_diagnostic": raw_carry_margin_diagnostic(
            features["base01"], features["labels"], delta
        ),
        "claim_boundary": CONSTANT_BIAS_CLAIM_BOUNDARY,
    }


def _build_nuisance_only_payload(context, device):
    features = context["features"]
    rows = context["board_context"]["rows"]
    fit_budget = context["upstream_plan"]["fit_budget"]
    initial_state, initial_state_sha256 = initial_nuisance_only_state()
    fit_model_binding = _context_fit_model_binding(context)
    motor, fit_report = _recovery_fit_nuisance_only(
        features,
        canonical_fit_selection_rows(context),
        features["labels"],
        initial_state,
        device,
        fit_budget["updates"],
        fit_budget["batch_size"],
        fit_budget["lr"],
        fit_budget["weight_decay"],
        fit_budget["seed"],
        fit_model_binding=fit_model_binding,
    )
    state = REVIEWED_MODULE_STATE_DICT(motor)
    return {
        "schema": NUISANCE_ONLY_PAYLOAD_SCHEMA,
        "control_id": NUISANCE_ONLY_CONTROL_ID,
        "parameterization": nuisance_only_parameterization(),
        "training_rows": len(features["labels"]),
        "training_feature_payload_sha256": context["feature_merge"][
            "teacher_metric_feature_payload_sha256"
        ],
        "training_metadata_receipt": nuisance_metadata_receipt(rows),
        "capacity_ledger": nuisance_capacity_ledger(),
        "label_source": "features.labels_true",
        "initial_state_sha256": initial_state_sha256,
        "state": state,
        "state_sha256": REVIEWED_TENSOR_STATE_SHA256(state),
        "fit": fit_report,
        "width_ood_policy": nuisance_width_ood_policy(),
        "claim_boundary": NUISANCE_ONLY_CLAIM_BOUNDARY,
    }


def build_trajectory_replay_proof(
    fit_payload, constant_bias_payload, nuisance_only_payload
):
    assert_reviewed_callable_exports()
    if type(fit_payload) is not dict or set(fit_payload) != LEGACY_PAYLOAD_KEYS:
        raise ValueError("trajectory proof requires the complete legacy payload")
    if (
        type(constant_bias_payload) is not dict
        or set(constant_bias_payload) != CONSTANT_BIAS_PAYLOAD_KEYS
    ):
        raise ValueError("trajectory proof requires the complete constant-bias payload")
    if (
        type(nuisance_only_payload) is not dict
        or set(nuisance_only_payload) != NUISANCE_ONLY_PAYLOAD_KEYS
    ):
        raise ValueError("trajectory proof requires the complete nuisance-only payload")
    return canonical_json_document(
        {
            "schema": RECOVERY_TRAJECTORY_PROOF_SCHEMA,
            "comparison": "type-strict-complete-tree-and-tensor-equality",
            "payload_sha256": scientific_tree_sha256(fit_payload),
            "constant_bias_payload_sha256": scientific_tree_sha256(
                constant_bias_payload
            ),
            "nuisance_only_payload_sha256": scientific_tree_sha256(
                nuisance_only_payload
            ),
            "treatment_state_sha256": fit_payload["treatment_state_sha256"],
            "shuffled_state_sha256": fit_payload["shuffled_state_sha256"],
            "constant_bias_state_sha256": constant_bias_payload["state_sha256"],
            "nuisance_only_state_sha256": nuisance_only_payload["state_sha256"],
            "schedule_sha256": fit_payload["treatment_fit"]["schedule_sha256"],
            "reviewed_semantic_closure_sha256": (REVIEWED_SEMANTIC_CLOSURE_SHA256),
            "adamw_updates_per_reader_arm": fit_payload["treatment_fit"]["updates"],
            "constant_bias_full_board_cross_entropy": constant_bias_payload["fit"][
                "full_board_cross_entropy"
            ],
            "nuisance_only_full_board_cross_entropy": nuisance_only_payload["fit"][
                "full_board_cross_entropy"
            ],
            "constant_bias_selected_checkpoint_evidence_sha256": (
                constant_bias_payload["fit"]["selected_checkpoint_evidence_sha256"]
            ),
            "nuisance_only_selected_checkpoint_evidence_sha256": (
                nuisance_only_payload["fit"]["selected_checkpoint_evidence_sha256"]
            ),
            "nulls_converged": (
                constant_bias_payload["fit"]["converged"] is True
                and nuisance_only_payload["fit"]["converged"] is True
            ),
            "arms": ["treatment", "constant_bias", "nuisance_only", "shuffled"],
            "seal_policy": (
                "fresh publication requires a second full replay; every recoverable or "
                "presealed candidate requires a current full replay before acceptance"
            ),
        }
    )


def prove_fit_payload_trajectory(
    fit_payload,
    constant_bias_payload,
    nuisance_only_payload,
    context,
    checkpoint,
    device,
):
    """Recompute AdamW trajectories and globally selected null states before sealing."""
    assert_reviewed_callable_exports()
    expected = _build_fit_payload(context, checkpoint, device)
    if not type_strict_equal(expected, fit_payload):
        raise RuntimeError(
            "published recovery fit differs from independent complete trajectory replay"
        )
    expected_constant = _build_constant_bias_payload(context, device)
    if not type_strict_equal(expected_constant, constant_bias_payload):
        raise RuntimeError(
            "published constant-bias fit differs from independent complete trajectory replay"
        )
    expected_nuisance = _build_nuisance_only_payload(context, device)
    if not type_strict_equal(expected_nuisance, nuisance_only_payload):
        raise RuntimeError(
            "published nuisance-only fit differs from independent complete trajectory "
            "replay"
        )
    proof = build_trajectory_replay_proof(
        expected, expected_constant, expected_nuisance
    )
    if proof["adamw_updates_per_reader_arm"] != 2000:
        raise RuntimeError(
            "recovery reader trajectories did not replay exactly 2,000 updates/arm"
        )
    if proof["nulls_converged"] is not True:
        raise RuntimeError("recovery null selection evidence is nonconverged")
    assert_reviewed_callable_exports()
    return proof


def validate_and_replay_recovery_fit(
    bundle,
    recovery_plan,
    recovery_plan_sha256,
    context,
    checkpoint,
    device,
):
    """The sole gate from finite/self-consistent bytes to seal eligibility."""
    validate_recovery_fit_bundle(
        bundle,
        recovery_plan,
        recovery_plan_sha256,
        context,
        checkpoint,
        device,
    )
    replay_proof = prove_fit_payload_trajectory(
        bundle["fit_payload"],
        bundle["constant_bias_payload"],
        bundle["nuisance_only_payload"],
        context,
        checkpoint,
        device,
    )
    if not type_strict_equal(replay_proof, bundle["trajectory_replay_proof"]):
        raise RuntimeError(
            "recovery artifact self-report differs from trajectory replay"
        )
    return replay_proof


def _fit(args):
    if args.constant_bias_control != CONSTANT_BIAS_CONTROL_ID:
        raise ValueError("fit constant-bias control differs from reviewed null")
    if args.nuisance_only_control != NUISANCE_ONLY_CONTROL_ID:
        raise ValueError("fit nuisance-only control differs from reviewed null")
    if args.null_optimization_control != NULL_OPTIMIZATION_SCHEMA:
        raise ValueError("fit null optimization differs from reviewed full-board rule")
    if args.selection_widths != "4,6":
        raise ValueError("fit selection widths differ from reviewed sealed boundary")
    context = prepare_context(args)
    recovery_plan_bound = None
    try:
        root = recovery_root(args.recovery_source_commit)
        if os.fspath(args.recovery_root) != str(root):
            raise ValueError("fit recovery root aliases immutable derived path")
        expected_plan_path = root / "recovery_plan.json"
        recovery_plan_bound = BoundFile(
            args.recovery_plan,
            expected_plan_path,
            args.recovery_plan_sha256,
            "recovery plan",
            required_mode=0o444,
            required_parent_mode=0o555,
        )
        recovery_plan = load_exact_json(recovery_plan_bound.text(), "recovery plan")
        validate_recovery_plan_document(
            recovery_plan,
            context["expected_recovery_plan"],
            args.recovery_source_commit,
        )
        out = Path(recovery_plan["output_contract"]["fit_artifact"])
        if out != root / "fit" / "motor.pt" or UPSTREAM_ROOT in out.parents:
            raise ValueError("recovery fit output aliases forbidden upstream root")
        signed_layout_binding = recovery_plan["output_contract"]["layout_binding"]
        fit_state = prepare_recovery_fit_publication(out, signed_layout_binding)
        validate_recovery_layout(root, fit_state=fit_state)
        device, slurm_attestation = require_recovery_cuda_runtime(
            context["upstream_plan"]
        )
        verify_context_bindings(context, recovery_plan_bound)
        assert_upstream_custody_unchanged(
            context["upstream_custody_snapshot"], "before recovery fit handling"
        )
        checkpoint = safe_torch_load(context["frozen_bounds"]["checkpoint"])
        if fit_state in {"recoverable", "sealed"}:
            published = BoundFile(
                str(out),
                out,
                sha256_file(out),
                "existing v11 recovery fit",
                required_mode=0o444,
                required_parent_mode=0o700 if fit_state == "recoverable" else 0o555,
            )
            try:
                replay = safe_torch_load(published)
                validate_and_replay_recovery_fit(
                    replay,
                    recovery_plan,
                    recovery_plan_bound.sha256,
                    context,
                    checkpoint,
                    device,
                )
                verify_context_bindings(context, recovery_plan_bound)
                published.verify()
            finally:
                published.close()
            if fit_state == "recoverable":
                assert_upstream_custody_unchanged(
                    context["upstream_custody_snapshot"],
                    "immediately before recovered publication sealing",
                )
                verify_context_bindings(context, recovery_plan_bound)
                seal_recovery_fit(out, signed_layout_binding)
                verify_context_bindings(context, recovery_plan_bound)
                assert_upstream_custody_unchanged(
                    context["upstream_custody_snapshot"],
                    "immediately after recovered publication sealing",
                )
            validate_recovery_layout(root, fit_state="sealed")
            print(
                json.dumps(
                    {
                        "audit": RECOVERY_FIT_AUDIT,
                        "artifact": str(out),
                        "sha256": sha256_file(out),
                        "existing_validated": True,
                        "claim_boundary": RECOVERY_FIT_CLAIM_BOUNDARY,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return
        if fit_state != "empty":
            raise ValueError("recovery fit state is not publishable")
        fit_payload = _build_fit_payload(context, checkpoint, device)
        constant_bias_payload = _build_constant_bias_payload(context, device)
        nuisance_only_payload = _build_nuisance_only_payload(context, device)
        for control_name, payload in (
            ("constant-bias", constant_bias_payload),
            ("nuisance-only", nuisance_only_payload),
        ):
            if (
                payload["fit"]["converged"] is not True
                or payload["fit"]["optimizer_final_iterate_used"] is not False
            ):
                raise RuntimeError(
                    f"{control_name} full-board checkpoint did not converge"
                )
        bundle = {
            "audit": RECOVERY_FIT_AUDIT,
            "recovery": True,
            "recovery_plan_sha256": recovery_plan_bound.sha256,
            "recovery_executor_source_contract": context["recovery_source_contract"],
            "executor_runtime_contract": context["executor_runtime_contract"],
            "recovery_parent_binding": context["recovery_parent_binding"],
            "recovery_layout_binding": context["recovery_layout_binding"],
            "upstream_protocol_source_contract": context["upstream_source_contract"],
            "upstream_plan_binding": recovery_plan["upstream_protocol"]["plan_binding"],
            "upstream_shard_receipts": context["shard_receipts"],
            "normalization_proof": context["board_context"]["normalization_proof"],
            "allowed_transformation": ALLOWED_TRANSFORMATION,
            "deserialization_contract": DESERIALIZATION_CONTRACT,
            "slurm_h100_attestation": slurm_attestation,
            "parameter_ledger": deployment_parameter_ledger(
                checkpoint["cfg"], context["upstream_plan"]["fit_budget"]["rank"]
            ),
            "trajectory_replay_proof": build_trajectory_replay_proof(
                fit_payload, constant_bias_payload, nuisance_only_payload
            ),
            "fit_payload": fit_payload,
            "constant_bias_payload": constant_bias_payload,
            "nuisance_only_payload": nuisance_only_payload,
            "claim_boundary": RECOVERY_FIT_CLAIM_BOUNDARY,
        }
        validate_recovery_fit_bundle(
            bundle,
            recovery_plan,
            recovery_plan_bound.sha256,
            context,
            checkpoint,
            device,
        )
        verify_context_bindings(context, recovery_plan_bound)
        assert_upstream_custody_unchanged(
            context["upstream_custody_snapshot"],
            "immediately before recovery artifact publication",
        )
        published_sha256 = publish_recovery_torch(out, bundle, signed_layout_binding)
        verify_context_bindings(context, recovery_plan_bound)
        assert_upstream_custody_unchanged(
            context["upstream_custody_snapshot"],
            "immediately after recovery artifact publication",
        )
        validate_recovery_layout(root, fit_state="recoverable")
        published = BoundFile(
            str(out),
            out,
            published_sha256,
            "published v11 recovery fit",
            required_mode=0o444,
            required_parent_mode=0o700,
        )
        try:
            replay = safe_torch_load(published)
            validate_and_replay_recovery_fit(
                replay,
                recovery_plan,
                recovery_plan_bound.sha256,
                context,
                checkpoint,
                device,
            )
            verify_context_bindings(context, recovery_plan_bound)
            published.verify()
        finally:
            published.close()
        assert_upstream_custody_unchanged(
            context["upstream_custody_snapshot"],
            "immediately before recovery artifact sealing",
        )
        verify_context_bindings(context, recovery_plan_bound)
        seal_recovery_fit(out, signed_layout_binding)
        verify_context_bindings(context, recovery_plan_bound)
        assert_upstream_custody_unchanged(
            context["upstream_custody_snapshot"],
            "immediately after recovery artifact sealing",
        )
        validate_recovery_layout(root, fit_state="sealed")
        final = BoundFile(
            str(out),
            out,
            published_sha256,
            "sealed v11 recovery fit",
            required_mode=0o444,
            required_parent_mode=0o555,
        )
        try:
            final.verify()
        finally:
            final.close()
        print(
            json.dumps(
                {
                    "audit": RECOVERY_FIT_AUDIT,
                    "artifact": str(out),
                    "sha256": published_sha256,
                    "claim_boundary": RECOVERY_FIT_CLAIM_BOUNDARY,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        if recovery_plan_bound is not None:
            recovery_plan_bound.close()
        close_context(context)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = argparse.ArgumentParser(add_help=False)
    source.add_argument("--recovery-source-commit", required=True)
    source.add_argument("--recovery-source-manifest-sha256", required=True)
    source.add_argument("--recovery-root", required=True)
    reviewed = argparse.ArgumentParser(add_help=False, parents=[source])
    reviewed.add_argument("--hostile-review", required=True)
    reviewed.add_argument("--hostile-review-sha256", required=True)
    reviewed.add_argument("--recovery-plan", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    reserve = commands.add_parser("reserve", parents=[source])
    reserve.set_defaults(func=_reserve)
    plan = commands.add_parser("plan", parents=[reviewed])
    plan.set_defaults(func=_plan)
    fit = commands.add_parser("fit", parents=[reviewed])
    fit.add_argument("--recovery-plan-sha256", required=True)
    fit.add_argument("--constant-bias-control", required=True)
    fit.add_argument("--nuisance-only-control", required=True)
    fit.add_argument("--null-optimization-control", required=True)
    fit.add_argument("--selection-widths", required=True)
    fit.set_defaults(func=_fit)
    return parser


def main():
    enforce_secure_creation_umask()
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
