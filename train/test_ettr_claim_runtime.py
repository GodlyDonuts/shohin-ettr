from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

import pytest

import ettr_claim_runtime as runtime


SOURCE_COMMIT = "1" * 40


def _write_file(path: Path, payload: bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o555 if executable else 0o444)


def _runtime_tree(tmp_path: Path) -> Path:
    root = tmp_path / "runtime"
    _write_file(
        root / runtime.PYTHON_RELATIVE_PATH,
        b"fixture-python\n",
        executable=True,
    )
    _write_file(
        root / "miniforge3/lib/python3.13/os.py",
        b"fixture-stdlib\n",
    )
    _write_file(
        root / "miniforge3/lib/python3.13/site-packages/torch/__init__.py",
        b"fixture-torch\n",
    )
    _write_file(
        root / "miniforge3/lib/python3.13/site-packages/torch/lib/libtorch.so",
        b"fixture-native\n",
    )
    _write_file(
        root
        / "miniforge3/lib/python3.13/site-packages/safetensors/__init__.py",
        b"fixture-safetensors\n",
    )
    candidate = root / runtime.CANDIDATE_SOURCE_RELATIVE_ROOT
    for stage, runner in runtime.STAGE_RUNNERS.items():
        source_names = (*runtime.COMMON_CANDIDATE_SOURCE_FILES, runner)
        for name in source_names:
            _write_file(
                candidate / stage / name,
                f"# {name}\n".encode("ascii"),
            )
    tools = root / runtime.TOOLS_RELATIVE_ROOT
    for name in runtime.TOOL_FILES:
        _write_file(tools / name, f"# {name}\n".encode("ascii"))
    link = root / "miniforge3/bin/python3"
    link.symlink_to("python")
    for directory in sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    root.chmod(0o555)
    return root


def _make_archive(tmp_path: Path) -> tuple[Path, Path, runtime.ETTRClaimRuntimeInventory]:
    root = _runtime_tree(tmp_path)
    archive = tmp_path / "runtime.tar"
    inventory = runtime.build_archive(
        root,
        archive,
        source_commit=SOURCE_COMMIT,
    )
    return root, archive, inventory


def test_runtime_archive_is_deterministic_recursive_and_exact(
    tmp_path: Path,
) -> None:
    first_root = _runtime_tree(tmp_path / "first")
    second_root = _runtime_tree(tmp_path / "second")
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"

    first_inventory = runtime.build_archive(
        first_root,
        first,
        source_commit=SOURCE_COMMIT,
    )
    second_inventory = runtime.build_archive(
        second_root,
        second,
        source_commit=SOURCE_COMMIT,
    )

    assert first_inventory == second_inventory
    assert first.read_bytes() == second.read_bytes()
    assert runtime.validate_archive(first) == first_inventory
    native = next(
        member
        for member in first_inventory.members
        if member.path.endswith("libtorch.so")
    )
    assert native.sha256 == hashlib.sha256(b"fixture-native\n").hexdigest()


def test_runtime_tree_rejects_deep_native_mutation(
    tmp_path: Path,
) -> None:
    root, _, inventory = _make_archive(tmp_path)
    native = (
        root
        / "miniforge3/lib/python3.13/site-packages/torch/lib/libtorch.so"
    )
    native.chmod(0o644)
    native.write_bytes(b"attacker-native\n")
    native.chmod(0o444)
    with pytest.raises(runtime.ETTRClaimRuntimeError, match="tree differs"):
        runtime.validate_runtime_tree(root, inventory)


def test_source_bundle_digest_binds_candidate_and_tool_bytes(
    tmp_path: Path,
) -> None:
    root = _runtime_tree(tmp_path)
    initial = runtime.build_inventory(root, source_commit=SOURCE_COMMIT)
    expected = runtime.reviewed_source_bundle_sha256(
        lambda relative: f"# {Path(relative).name}\n".encode("ascii")
    )
    candidate = (
        root
        / runtime.CANDIDATE_SOURCE_RELATIVE_ROOT
        / "world"
        / runtime.STAGE_RUNNERS["world"]
    )
    candidate.chmod(0o644)
    candidate.write_bytes(b"# attacker runner\n")
    candidate.chmod(0o444)
    changed = runtime.build_inventory(root, source_commit=SOURCE_COMMIT)

    assert initial.source_bundle_sha256() == expected
    assert initial.source_bundle_sha256() != changed.source_bundle_sha256()
    assert initial.source_commit == changed.source_commit


def test_runtime_tree_rejects_symlink_and_writable_roots(
    tmp_path: Path,
) -> None:
    root = _runtime_tree(tmp_path / "source")
    inventory = runtime.build_inventory(root, source_commit=SOURCE_COMMIT)
    symlink = tmp_path / "runtime-link"
    symlink.symlink_to(root, target_is_directory=True)
    with pytest.raises(
        runtime.ETTRClaimRuntimeError,
        match="build root differs",
    ):
        runtime.validate_runtime_tree(symlink, inventory)

    root.chmod(0o755)
    with pytest.raises(
        runtime.ETTRClaimRuntimeError,
        match="build root differs",
    ):
        runtime.validate_runtime_tree(root, inventory)


def test_runtime_inventory_rejects_missing_or_symlink_parents(
    tmp_path: Path,
) -> None:
    root = _runtime_tree(tmp_path)
    inventory = runtime.build_inventory(root, source_commit=SOURCE_COMMIT)
    parent = "miniforge3/lib/python3.13/site-packages/torch"

    without_parent = replace(
        inventory,
        members=tuple(
            member for member in inventory.members if member.path != parent
        ),
    )
    with pytest.raises(
        runtime.ETTRClaimRuntimeError,
        match="parent hierarchy differs",
    ):
        without_parent.validate()

    symlink_parent = replace(
        inventory,
        members=tuple(
            replace(
                member,
                kind="symlink",
                mode=0o777,
                size=0,
                sha256=None,
                link_target="../safetensors",
            )
            if member.path == parent
            else member
            for member in inventory.members
        ),
    )
    with pytest.raises(
        runtime.ETTRClaimRuntimeError,
        match="parent hierarchy differs",
    ):
        symlink_parent.validate()


def test_verified_extraction_is_exact_and_immutable(tmp_path: Path) -> None:
    _, archive, inventory = _make_archive(tmp_path)
    destination = tmp_path / "extracted"

    extracted = runtime.extract_verified_archive(
        archive,
        destination,
        expected_archive_sha256=hashlib.sha256(
            archive.read_bytes()
        ).hexdigest(),
        expected_inventory_sha256=inventory.sha256(),
        expected_source_bundle_sha256=inventory.source_bundle_sha256(),
    )

    assert extracted == inventory
    assert (destination / runtime.INVENTORY_NAME).read_bytes() == (
        inventory.canonical_bytes()
    )
    runtime.validate_runtime_tree(
        destination / runtime.RUNTIME_PREFIX,
        inventory,
    )
    assert destination.stat().st_mode & 0o222 == 0


def test_verified_extraction_rejects_wrong_pins_without_output(
    tmp_path: Path,
) -> None:
    _, archive, inventory = _make_archive(tmp_path)
    cases = (
        (
            "archive",
            "0" * 64,
            inventory.sha256(),
            inventory.source_bundle_sha256(),
        ),
        (
            "inventory",
            hashlib.sha256(archive.read_bytes()).hexdigest(),
            "0" * 64,
            inventory.source_bundle_sha256(),
        ),
        (
            "source",
            hashlib.sha256(archive.read_bytes()).hexdigest(),
            inventory.sha256(),
            "0" * 64,
        ),
    )
    for label, archive_sha256, inventory_sha256, source_sha256 in cases:
        destination = tmp_path / f"extracted-{label}"
        with pytest.raises(runtime.ETTRClaimRuntimeError, match="digest differs"):
            runtime.extract_verified_archive(
                archive,
                destination,
                expected_archive_sha256=archive_sha256,
                expected_inventory_sha256=inventory_sha256,
                expected_source_bundle_sha256=source_sha256,
            )
        assert not destination.exists()


def test_verified_extraction_cli_uses_exact_pins(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, archive, inventory = _make_archive(tmp_path)
    destination = tmp_path / "extracted"

    assert runtime._main(
        (
            "extract",
            "--archive",
            str(archive),
            "--destination",
            str(destination),
            "--expected-archive-sha256",
            hashlib.sha256(archive.read_bytes()).hexdigest(),
            "--expected-inventory-sha256",
            inventory.sha256(),
            "--expected-source-bundle-sha256",
            inventory.source_bundle_sha256(),
        )
    ) == 0

    assert capsys.readouterr().out.strip() == inventory.sha256()
    runtime.validate_runtime_tree(
        destination / runtime.RUNTIME_PREFIX,
        inventory,
    )


def test_extract_exec_retains_descriptor_through_launch_and_removes_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, archive, inventory = _make_archive(tmp_path)
    destination = tmp_path / "extracted"
    observed: list[tuple[str, ...]] = []

    def fake_run(
        command: tuple[str, ...],
        *,
        check: bool,
        env: dict[str, str],
        pass_fds: tuple[int, ...],
    ) -> subprocess.CompletedProcess[bytes]:
        assert check is True
        assert env == {"HOME": "/nonexistent", "PATH": "/usr/bin:/bin"}
        assert command[0] == "/usr/bin/bwrap"
        assert len(pass_fds) == 1
        descriptor_root = f"/proc/self/fd/{pass_fds[0]}/runtime"
        assert descriptor_root in command
        assert runtime.stat.S_ISDIR(os.fstat(pass_fds[0]).st_mode)
        observed.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    assert runtime._main(
        (
            "extract-exec",
            "--archive",
            str(archive),
            "--destination",
            str(destination),
            "--expected-archive-sha256",
            hashlib.sha256(archive.read_bytes()).hexdigest(),
            "--expected-inventory-sha256",
            inventory.sha256(),
            "--expected-source-bundle-sha256",
            inventory.source_bundle_sha256(),
            "--",
            "/usr/bin/bwrap",
            "--ro-bind",
            "{ETTR_RUNTIME_ROOT}",
            "/runtime",
            "--",
            "true",
        )
    ) == 0

    assert len(observed) == 1
    assert not destination.exists()
    assert capsys.readouterr().out.strip() == inventory.sha256()


def test_verified_extraction_rejects_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, archive, inventory = _make_archive(tmp_path / "trusted")
    hostile_root = _runtime_tree(tmp_path / "hostile")
    hostile_file = hostile_root / "miniforge3/lib/python3.13/os.py"
    hostile_file.chmod(0o644)
    hostile_file.write_bytes(b"attacker-stdlib\n")
    hostile_file.chmod(0o444)
    hostile = tmp_path / "hostile.tar"
    runtime.build_archive(
        hostile_root,
        hostile,
        source_commit=SOURCE_COMMIT,
    )
    trusted_archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    original_digest = runtime._descriptor_sha256
    swapped = False

    def swap_after_hash(descriptor: int) -> tuple[str, int]:
        nonlocal swapped
        result = original_digest(descriptor)
        if not swapped:
            swapped = True
            archived = archive.with_suffix(".trusted")
            os.replace(archive, archived)
            os.replace(hostile, archive)
        return result

    monkeypatch.setattr(runtime, "_descriptor_sha256", swap_after_hash)
    destination = tmp_path / "extracted"
    try:
        runtime.extract_verified_archive(
            archive,
            destination,
            expected_archive_sha256=trusted_archive_sha256,
            expected_inventory_sha256=inventory.sha256(),
            expected_source_bundle_sha256=inventory.source_bundle_sha256(),
        )
    except runtime.ETTRClaimRuntimeError as exc:
        assert "changed during extraction" in str(exc)
        assert not destination.exists()
    else:
        assert (
            destination
            / runtime.RUNTIME_PREFIX
            / "miniforge3/lib/python3.13/os.py"
        ).read_bytes() == b"fixture-stdlib\n"


def test_verified_extraction_does_not_follow_or_delete_swapped_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, archive, inventory = _make_archive(tmp_path)
    destination = tmp_path / "extracted"
    moved = tmp_path / "moved-extracted"
    original_extract = runtime._extract_file_at
    swapped = False

    def swap_destination(*args: object, **kwargs: object) -> None:
        nonlocal swapped
        if not swapped:
            swapped = True
            destination.rename(moved)
            destination.mkdir()
            (destination / "sentinel").write_bytes(b"do-not-delete\n")
        original_extract(*args, **kwargs)

    monkeypatch.setattr(runtime, "_extract_file_at", swap_destination)
    with pytest.raises(
        runtime.ETTRClaimRuntimeError,
        match="destination changed",
    ):
        runtime.extract_verified_archive(
            archive,
            destination,
            expected_archive_sha256=hashlib.sha256(
                archive.read_bytes()
            ).hexdigest(),
            expected_inventory_sha256=inventory.sha256(),
            expected_source_bundle_sha256=(
                inventory.source_bundle_sha256()
            ),
        )

    assert (destination / "sentinel").read_bytes() == b"do-not-delete\n"


@pytest.mark.parametrize(
    "relative",
    (
        "miniforge3/lib/python3.13/site-packages/attacker.pth",
        "miniforge3/lib/python3.13/site-packages/attacker.pyc",
        "miniforge3/lib/python3.13/site-packages/__pycache__/attacker.py",
        "miniforge3/lib/python3.13/sitecustomize.py",
        "miniforge3/lib/python3.13/usercustomize.py",
    ),
)
def test_runtime_inventory_rejects_executable_import_metadata(
    tmp_path: Path,
    relative: str,
) -> None:
    root = _runtime_tree(tmp_path)
    parent = (root / relative).parent
    created_parent = False
    if not parent.exists():
        parent.parent.chmod(0o755)
        parent.mkdir()
        parent.parent.chmod(0o555)
        created_parent = True
    parent.chmod(0o755)
    _write_file(root / relative, b"raise RuntimeError('attacker')\n")
    parent.chmod(0o555)
    assert parent.exists() and (created_parent or parent.is_dir())
    with pytest.raises(
        runtime.ETTRClaimRuntimeError,
        match="executable import metadata",
    ):
        runtime.build_inventory(root, source_commit=SOURCE_COMMIT)


def test_runtime_inventory_rejects_extra_candidate_or_tool_source(
    tmp_path: Path,
) -> None:
    root = _runtime_tree(tmp_path)
    candidate = root / runtime.CANDIDATE_SOURCE_RELATIVE_ROOT / "world"
    candidate.chmod(0o755)
    _write_file(candidate / "oracle.py", b"raise RuntimeError\n")
    candidate.chmod(0o555)
    with pytest.raises(
        runtime.ETTRClaimRuntimeError,
        match="candidate source bundle inventory differs",
    ):
        runtime.build_inventory(root, source_commit=SOURCE_COMMIT)


def test_runtime_inventory_rejects_writable_file_and_escaping_symlink(
    tmp_path: Path,
) -> None:
    root = _runtime_tree(tmp_path / "writable")
    target = root / runtime.BOOTSTRAP_RELATIVE_PATH
    target.chmod(0o644)
    with pytest.raises(
        runtime.ETTRClaimRuntimeError,
        match="runtime file metadata differs",
    ):
        runtime.build_inventory(root, source_commit=SOURCE_COMMIT)

    other = _runtime_tree(tmp_path / "link")
    bin_directory = other / "miniforge3/bin"
    bin_directory.chmod(0o755)
    escaping = bin_directory / "escape"
    escaping.symlink_to("../../../../outside")
    bin_directory.chmod(0o555)
    with pytest.raises(
        runtime.ETTRClaimRuntimeError,
        match="escapes runtime root",
    ):
        runtime.build_inventory(other, source_commit=SOURCE_COMMIT)


def test_runtime_archive_rejects_duplicate_traversal_hardlink_and_device(
    tmp_path: Path,
) -> None:
    _, archive, inventory = _make_archive(tmp_path)
    cases = (
        ("duplicate", tarfile.REGTYPE, "runtime/duplicate", ""),
        ("traversal", tarfile.REGTYPE, "../escape", ""),
        ("hardlink", tarfile.LNKTYPE, "runtime/hard", "runtime/target"),
        ("device", tarfile.CHRTYPE, "runtime/device", ""),
    )
    for label, kind, name, linkname in cases:
        hostile = tmp_path / f"{label}.tar"
        with tarfile.open(hostile, "w") as target:
            for member in tarfile.open(archive, "r:"):
                source_archive = tarfile.open(archive, "r:")
                source_member = source_archive.getmember(member.name)
                handle = (
                    source_archive.extractfile(source_member)
                    if source_member.isfile()
                    else None
                )
                target.addfile(source_member, handle)
                source_archive.close()
            injected = tarfile.TarInfo(name=name)
            injected.type = kind
            injected.mode = 0o444
            injected.linkname = linkname
            if kind == tarfile.REGTYPE:
                payload = b"attacker"
                injected.size = len(payload)
                target.addfile(injected, io.BytesIO(payload))
            else:
                target.addfile(injected)
        hostile.chmod(0o444)
        with pytest.raises(runtime.ETTRClaimRuntimeError):
            runtime.validate_archive(
                hostile,
                expected_inventory=inventory,
            )


def test_runtime_archive_requires_exact_inventory_and_runtime_roots(
    tmp_path: Path,
) -> None:
    _, archive, inventory = _make_archive(tmp_path)
    cases = ("missing-runtime-root", "inventory-prefix")
    for case in cases:
        hostile = tmp_path / f"{case}.tar"
        with (
            tarfile.open(archive, "r:") as source,
            tarfile.open(hostile, "w") as target,
        ):
            for member in source:
                if case == "missing-runtime-root" and member.name == "runtime":
                    continue
                handle = source.extractfile(member) if member.isfile() else None
                copied = tarfile.TarInfo(member.name)
                copied.mode = member.mode
                copied.type = member.type
                copied.size = member.size
                copied.linkname = member.linkname
                if (
                    case == "inventory-prefix"
                    and member.name.startswith("runtime/")
                ):
                    copied.name = (
                        f"{runtime.INVENTORY_NAME}/"
                        f"{member.name.removeprefix('runtime/')}"
                    )
                target.addfile(copied, handle)
        hostile.chmod(0o444)
        with pytest.raises(runtime.ETTRClaimRuntimeError):
            runtime.validate_archive(
                hostile,
                expected_inventory=inventory,
            )


def test_runtime_archive_rejects_inventory_reassociation(
    tmp_path: Path,
) -> None:
    _, archive, inventory = _make_archive(tmp_path)
    changed = replace(inventory, source_commit="2" * 40)
    with pytest.raises(
        runtime.ETTRClaimRuntimeError,
        match="inventory differs from expected",
    ):
        runtime.validate_archive(
            archive,
            expected_inventory=changed,
        )


def test_verification_receipt_binds_archive_and_extracted_tree(
    tmp_path: Path,
) -> None:
    root, archive, inventory = _make_archive(tmp_path)
    receipt = runtime.verification_receipt(archive, root, inventory)

    assert receipt.archive_sha256 == hashlib.sha256(
        archive.read_bytes()
    ).hexdigest()
    assert receipt.inventory_sha256 == inventory.sha256()
    assert receipt.source_commit == SOURCE_COMMIT
    assert receipt.member_count == len(inventory.members)
    assert (
        json.loads(receipt.canonical_bytes())["schema"]
        == runtime.CLAIM_RUNTIME_RECEIPT_SCHEMA
    )


def test_archive_must_be_immutable_single_link(tmp_path: Path) -> None:
    _, archive, _ = _make_archive(tmp_path)
    archive.chmod(0o644)
    with pytest.raises(
        runtime.ETTRClaimRuntimeError,
        match="immutable single-link",
    ):
        runtime.validate_archive(archive)
    archive.chmod(0o444)
    hardlink = tmp_path / "runtime-hardlink.tar"
    os.link(archive, hardlink)
    with pytest.raises(
        runtime.ETTRClaimRuntimeError,
        match="immutable single-link",
    ):
        runtime.validate_archive(archive)


def test_noncanonical_inventory_is_rejected(tmp_path: Path) -> None:
    root = _runtime_tree(tmp_path)
    inventory = runtime.build_inventory(root, source_commit=SOURCE_COMMIT)
    value = json.loads(inventory.canonical_bytes())
    payload = json.dumps(value, indent=2).encode("ascii")
    with pytest.raises(
        runtime.ETTRClaimRuntimeError,
        match="not canonical",
    ):
        runtime.ETTRClaimRuntimeInventory.from_bytes(payload)
