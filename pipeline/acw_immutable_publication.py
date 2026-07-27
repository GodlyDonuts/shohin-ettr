"""Fail-closed publication primitives for ACW scientific artifacts."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import stat
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 0x00000004
TREE_PUBLICATION_RECEIPT_FD_ENV = "SHOHIN_ACW_TREE_PUBLICATION_RECEIPT_FD"


def _write_all(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError(errno.EIO, "short write during ACW artifact staging")
        view = view[written:]


def _emit_tree_publication_receipt(
    destination: Path,
    snapshot: dict[str, Any],
) -> None:
    descriptor_raw = os.environ.get(TREE_PUBLICATION_RECEIPT_FD_ENV)
    if descriptor_raw is None:
        return
    try:
        descriptor = int(descriptor_raw)
    except ValueError as error:
        raise ValueError(
            "ACW tree publication receipt descriptor is invalid"
        ) from error
    if descriptor < 0:
        raise ValueError("ACW tree publication receipt descriptor is invalid")
    raw = (
        json.dumps(
            {
                "protocol": "ACW-TREE-PUBLICATION-RECEIPT-v1",
                "destination": str(destination),
                "snapshot": snapshot,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )
    _write_all(descriptor, raw)
    os.fsync(descriptor)


def _open_flags(base: int) -> int:
    return base | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _close_descriptor(
    descriptor: int,
    *,
    label: str,
    primary_error: BaseException | None = None,
) -> None:
    try:
        os.close(descriptor)
    except BaseException as close_error:
        if primary_error is None:
            raise
        primary_error.add_note(f"{label} descriptor cleanup failed: {close_error}")


def fsync_directory(path: Path) -> None:
    with retained_evidence_snapshot() as retained:
        retained.retain_directory(path, fsync_metadata=True)


def _raise_rename_error(source: Path, destination: Path) -> None:
    error = ctypes.get_errno()
    if error == 0:
        error = errno.EIO
    raise OSError(
        error,
        f"atomic no-replace publication failed: {source} -> {destination}",
        destination,
    )


def rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish one file or directory without replacing a name."""

    source_raw = os.fsencode(source)
    destination_raw = os.fsencode(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    ctypes.set_errno(0)
    if sys.platform.startswith("linux"):
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise OSError(errno.ENOSYS, "renameat2 is required for ACW publication")
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            _AT_FDCWD,
            source_raw,
            _AT_FDCWD,
            destination_raw,
            _RENAME_NOREPLACE,
        )
    elif sys.platform == "darwin":
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is None:
            raise OSError(errno.ENOSYS, "renamex_np is required for ACW publication")
        renamex_np.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        renamex_np.restype = ctypes.c_int
        result = renamex_np(source_raw, destination_raw, _RENAME_EXCL)
    else:
        raise OSError(
            errno.ENOTSUP,
            f"atomic no-replace publication is unsupported on {sys.platform}",
        )
    if result != 0:
        _raise_rename_error(source, destination)


def _stable_file_metadata(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _stable_directory_metadata(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _require_name_bound_to_descriptor(
    path: Path,
    descriptor: int,
    record: dict[str, Any],
    *,
    directory: bool,
) -> None:
    descriptor_metadata = os.fstat(descriptor)
    try:
        name_metadata = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise OSError(
            errno.EIO,
            f"ACW descriptor name disappeared during readback: {path}",
        ) from error
    expected_type = stat.S_IFDIR if directory else stat.S_IFREG
    if (
        stat.S_IFMT(descriptor_metadata.st_mode) != expected_type
        or stat.S_IFMT(name_metadata.st_mode) != expected_type
        or descriptor_metadata.st_dev != name_metadata.st_dev
        or descriptor_metadata.st_ino != name_metadata.st_ino
    ):
        raise OSError(
            errno.EIO,
            f"ACW descriptor name changed during readback: {path}",
        )
    stable_metadata = _stable_directory_metadata if directory else _stable_file_metadata
    recorded_metadata = tuple(
        record[key]
        for key in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    )
    if stable_metadata(descriptor_metadata) != recorded_metadata:
        raise OSError(
            errno.EIO,
            f"ACW descriptor metadata changed after readback: {path}",
        )
    if stable_metadata(name_metadata) != recorded_metadata:
        raise OSError(
            errno.EIO,
            f"ACW descriptor name metadata changed after readback: {path}",
        )


def _descriptor_record_from_open_file(
    descriptor: int,
    path: Path,
    *,
    fsync_file: bool,
) -> dict[str, Any]:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"ACW artifact is not a regular file: {path}")
    if metadata.st_nlink != 1:
        raise ValueError(f"ACW artifact has unexpected hard links: {path}")
    if fsync_file:
        os.fsync(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    observed = 0
    while block := os.read(descriptor, 1 << 20):
        digest.update(block)
        observed += len(block)
    final_metadata = os.fstat(descriptor)
    if observed != metadata.st_size or _stable_file_metadata(
        final_metadata
    ) != _stable_file_metadata(metadata):
        raise OSError(errno.EIO, f"ACW artifact changed during readback: {path}")
    return {
        "bytes": observed,
        "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "sha256": digest.hexdigest(),
        "st_dev": metadata.st_dev,
        "st_ino": metadata.st_ino,
        "st_mode": metadata.st_mode,
        "st_nlink": metadata.st_nlink,
        "st_size": metadata.st_size,
        "st_mtime_ns": metadata.st_mtime_ns,
        "st_ctime_ns": metadata.st_ctime_ns,
    }


def _descriptor_record(path: Path, *, fsync_file: bool) -> dict[str, Any]:
    descriptor = os.open(path, _open_flags(os.O_RDONLY))
    try:
        record = _descriptor_record_from_open_file(
            descriptor,
            path,
            fsync_file=fsync_file,
        )
        _require_name_bound_to_descriptor(
            path,
            descriptor,
            record,
            directory=False,
        )
        return record
    finally:
        os.close(descriptor)


def _directory_record_from_open_descriptor(
    descriptor: int,
    path: Path,
    *,
    fsync_metadata: bool,
) -> dict[str, Any]:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise NotADirectoryError(path)
    if fsync_metadata:
        os.fsync(descriptor)
    final_metadata = os.fstat(descriptor)
    if _stable_directory_metadata(final_metadata) != _stable_directory_metadata(
        metadata
    ):
        raise OSError(errno.EIO, f"ACW directory changed during readback: {path}")
    return {
        "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "st_dev": metadata.st_dev,
        "st_ino": metadata.st_ino,
        "st_mode": metadata.st_mode,
        "st_nlink": metadata.st_nlink,
        "st_size": metadata.st_size,
        "st_mtime_ns": metadata.st_mtime_ns,
        "st_ctime_ns": metadata.st_ctime_ns,
    }


def _directory_record(path: Path, *, fsync_metadata: bool) -> dict[str, Any]:
    flags = _open_flags(os.O_RDONLY) | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        record = _directory_record_from_open_descriptor(
            descriptor,
            path,
            fsync_metadata=fsync_metadata,
        )
        _require_name_bound_to_descriptor(
            path,
            descriptor,
            record,
            directory=True,
        )
        return record
    finally:
        os.close(descriptor)


def _content_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in ("bytes", "mode", "sha256")}


def _file_identity_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in ("bytes", "mode", "sha256", "st_dev", "st_ino")}


def _directory_identity_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in ("mode", "st_dev", "st_ino")}


class RetainedEvidenceSnapshot:
    """Hold a set of canonical descriptors through one joint final barrier."""

    def __init__(self) -> None:
        self._files: dict[Path, tuple[int, dict[str, Any]]] = {}
        self._directories: dict[Path, tuple[int, dict[str, Any]]] = {}
        self._extra_descriptors: set[int] = set()
        self._closed = False

    @staticmethod
    def _canonical(path: Path) -> Path:
        return path.expanduser().absolute()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("retained ACW evidence set is already closed")

    def retain_file(self, path: Path) -> dict[str, Any]:
        self._require_open()
        path = self._canonical(path)
        if path in self._directories:
            raise ValueError(f"retained ACW path is both file and directory: {path}")
        if path in self._files:
            return _file_identity_record(self._files[path][1])
        descriptor = os.open(path, _open_flags(os.O_RDONLY))
        try:
            record = _descriptor_record_from_open_file(
                descriptor,
                path,
                fsync_file=False,
            )
            _require_name_bound_to_descriptor(
                path,
                descriptor,
                record,
                directory=False,
            )
        except BaseException:
            os.close(descriptor)
            raise
        self._files[path] = (descriptor, record)
        return _file_identity_record(record)

    def refresh_file(
        self,
        path: Path,
        *,
        fsync_file: bool,
    ) -> dict[str, Any]:
        self._require_open()
        path = self._canonical(path)
        if path not in self._files:
            return self.retain_file(path)
        descriptor, _ = self._files[path]
        record = _descriptor_record_from_open_file(
            descriptor,
            path,
            fsync_file=fsync_file,
        )
        _require_name_bound_to_descriptor(
            path,
            descriptor,
            record,
            directory=False,
        )
        self._files[path] = (descriptor, record)
        return _file_identity_record(record)

    def retain_directory(
        self,
        path: Path,
        *,
        fsync_metadata: bool = False,
    ) -> dict[str, Any]:
        self._require_open()
        path = self._canonical(path)
        if path in self._files:
            raise ValueError(f"retained ACW path is both file and directory: {path}")
        if path in self._directories:
            if fsync_metadata:
                return self.refresh_directory(path, fsync_metadata=True)
            return _directory_identity_record(self._directories[path][1])
        flags = _open_flags(os.O_RDONLY) | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(path, flags)
        try:
            record = _directory_record_from_open_descriptor(
                descriptor,
                path,
                fsync_metadata=fsync_metadata,
            )
            _require_name_bound_to_descriptor(
                path,
                descriptor,
                record,
                directory=True,
            )
        except BaseException:
            os.close(descriptor)
            raise
        self._directories[path] = (descriptor, record)
        return _directory_identity_record(record)

    def refresh_directory(
        self,
        path: Path,
        *,
        fsync_metadata: bool,
    ) -> dict[str, Any]:
        self._require_open()
        path = self._canonical(path)
        if path not in self._directories:
            return self.retain_directory(path, fsync_metadata=fsync_metadata)
        descriptor, _ = self._directories[path]
        record = _directory_record_from_open_descriptor(
            descriptor,
            path,
            fsync_metadata=fsync_metadata,
        )
        _require_name_bound_to_descriptor(
            path,
            descriptor,
            record,
            directory=True,
        )
        self._directories[path] = (descriptor, record)
        return _directory_identity_record(record)

    def retain_tree(self, root: Path) -> dict[str, Any]:
        self._require_open()
        return _retain_complete_tree(self, root)

    def refresh_tree(self, root: Path) -> dict[str, Any]:
        self._require_open()
        return _refresh_complete_tree(self, root)

    def _verify_records(self) -> None:
        for path, (descriptor, expected) in self._files.items():
            try:
                observed = _descriptor_record_from_open_file(
                    descriptor,
                    path,
                    fsync_file=False,
                )
            except ValueError as error:
                raise OSError(
                    errno.EIO,
                    f"retained ACW artifact changed during joint barrier: {path}",
                ) from error
            if observed != expected:
                raise OSError(
                    errno.EIO,
                    f"retained ACW artifact changed during joint barrier: {path}",
                )
        for path, (descriptor, expected) in self._directories.items():
            observed = _directory_record_from_open_descriptor(
                descriptor,
                path,
                fsync_metadata=False,
            )
            if observed != expected:
                raise OSError(
                    errno.EIO,
                    f"retained ACW directory changed during joint barrier: {path}",
                )

    def _verify_names(self, *, reverse: bool = False) -> None:
        files = [
            (path, descriptor, record, False)
            for path, (descriptor, record) in self._files.items()
        ]
        directories = [
            (path, descriptor, record, True)
            for path, (descriptor, record) in self._directories.items()
        ]
        if reverse:
            files.reverse()
            directories.sort(key=lambda item: len(item[0].parts), reverse=True)
        entries = [*files, *directories]
        for path, descriptor, record, directory in entries:
            _require_name_bound_to_descriptor(
                path,
                descriptor,
                record,
                directory=directory,
            )

    def verify(self) -> None:
        self._require_open()
        self._verify_records()
        self._verify_names()
        self._verify_records()
        self._verify_names(reverse=True)

    def rotate_descriptors(self) -> None:
        """Close one descriptor generation, then retain equivalent duplicates."""

        self._require_open()
        for records in (self._files, self._directories):
            for path, (descriptor, record) in list(records.items()):
                duplicate = os.dup(descriptor)
                try:
                    os.close(descriptor)
                except BaseException:
                    records[path] = (duplicate, record)
                    self._extra_descriptors.add(descriptor)
                    raise
                records[path] = (duplicate, record)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error: BaseException | None = None
        descriptors = {
            *(descriptor for descriptor, _ in self._files.values()),
            *(descriptor for descriptor, _ in self._directories.values()),
            *self._extra_descriptors,
        }
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


@contextmanager
def retained_evidence_snapshot() -> Iterator[RetainedEvidenceSnapshot]:
    """Retain a dynamically assembled evidence set until one joint barrier."""

    retained = RetainedEvidenceSnapshot()
    try:
        yield retained
    except BaseException as error:
        try:
            retained.close()
        except BaseException as close_error:
            error.add_note(f"retained ACW descriptor cleanup failed: {close_error}")
        raise
    else:
        try:
            retained.verify()
            retained.rotate_descriptors()
            retained.verify()
        except BaseException as error:
            try:
                retained.close()
            except BaseException as close_error:
                error.add_note(f"retained ACW descriptor cleanup failed: {close_error}")
            raise
        else:
            retained.close()


def _retain_complete_tree(
    retained: RetainedEvidenceSnapshot,
    root: Path,
) -> dict[str, Any]:
    root = root.expanduser().absolute()
    tree_files, tree_directories = _tree_entries(root)
    directories: dict[str, dict[str, Any]] = {}
    for path in tree_directories:
        relative = "." if path == root else path.relative_to(root).as_posix()
        directories[relative] = retained.retain_directory(path)
    files = {
        path.relative_to(root).as_posix(): retained.retain_file(path)
        for path in tree_files
    }
    final_files, final_directories = _tree_entries(root)
    if final_files != tree_files or final_directories != tree_directories:
        raise OSError(
            errno.EIO,
            f"ACW retained tree topology changed during descriptor opening: {root}",
        )
    return {
        "directories": {key: directories[key] for key in sorted(directories)},
        "files": files,
    }


def _refresh_complete_tree(
    retained: RetainedEvidenceSnapshot,
    root: Path,
) -> dict[str, Any]:
    root = root.expanduser().absolute()
    tree_files, tree_directories = _tree_entries(root)
    try:
        directories: dict[str, dict[str, Any]] = {}
        for path in tree_directories:
            relative = "." if path == root else path.relative_to(root).as_posix()
            directories[relative] = retained.refresh_directory(
                path,
                fsync_metadata=False,
            )
        files = {
            path.relative_to(root).as_posix(): retained.refresh_file(
                path,
                fsync_file=False,
            )
            for path in tree_files
        }
    except OSError as error:
        raise OSError(
            errno.EIO,
            f"retained ACW tree changed during descriptor refresh: {root}",
        ) from error
    final_files, final_directories = _tree_entries(root)
    if final_files != tree_files or final_directories != tree_directories:
        raise OSError(
            errno.EIO,
            f"retained ACW tree topology changed during descriptor refresh: {root}",
        )
    return {
        "directories": {key: directories[key] for key in sorted(directories)},
        "files": files,
    }


@contextmanager
def retained_tree_snapshot(root: Path) -> Iterator[dict[str, Any]]:
    """Retain every descriptor in one literal tree through caller success."""

    with retained_evidence_snapshot() as retained:
        yield retained.retain_tree(root)


def write_file_exclusive(
    path: Path, raw: bytes, *, mode: int = 0o600
) -> dict[str, Any]:
    """Create and fsync one file inside a private staging directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = _open_flags(os.O_WRONLY) | os.O_CREAT | os.O_EXCL
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(path, flags, mode)
        created = True
        _write_all(descriptor, raw)
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    except BaseException as error:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        if created:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                raise cleanup_error from error
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
    record = _descriptor_record(path, fsync_file=False)
    expected = {
        "bytes": len(raw),
        "mode": f"{mode:04o}",
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if _content_record(record) != expected:
        raise OSError(errno.EIO, f"staged ACW artifact readback differs: {path}")
    return _content_record(record)


def create_staging_directory(destination: Path) -> Path:
    destination = destination.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.stage-",
            dir=destination.parent,
        )
    )


def _tree_entries(root: Path) -> tuple[list[Path], list[Path]]:
    root_metadata = root.lstat()
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError(f"ACW publication root is not a literal directory: {root}")
    files: list[Path] = []
    directories = [root]
    for path in sorted(root.rglob("*")):
        metadata = path.lstat()
        relative = path.relative_to(root).as_posix()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"ACW publication tree contains a symlink: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            directories.append(path)
        elif stat.S_ISREG(metadata.st_mode):
            files.append(path)
        else:
            raise ValueError(
                f"ACW publication tree contains a special file: {relative}"
            )
    return files, directories


def _tree_snapshot(root: Path, *, fsync_files: bool) -> dict[str, Any]:
    tree_files, tree_directories = _tree_entries(root)
    first_directories: dict[str, dict[str, Any]] = {}
    for path in tree_directories:
        relative = "." if path == root else path.relative_to(root).as_posix()
        first_directories[relative] = _directory_record(path, fsync_metadata=False)
    file_records: dict[str, dict[str, Any]] = {}
    for path in tree_files:
        relative = path.relative_to(root).as_posix()
        file_records[relative] = _descriptor_record(path, fsync_file=fsync_files)

    final_files, final_directories = _tree_entries(root)
    initial_file_names = [path.relative_to(root).as_posix() for path in tree_files]
    final_file_names = [path.relative_to(root).as_posix() for path in final_files]
    initial_directory_names = [
        "." if path == root else path.relative_to(root).as_posix()
        for path in tree_directories
    ]
    final_directory_names = [
        "." if path == root else path.relative_to(root).as_posix()
        for path in final_directories
    ]
    if (
        final_file_names != initial_file_names
        or final_directory_names != initial_directory_names
    ):
        raise OSError(errno.EIO, f"ACW tree topology changed during readback: {root}")

    final_directory_records: dict[str, dict[str, Any]] = {}
    for path in final_directories:
        relative = "." if path == root else path.relative_to(root).as_posix()
        final_directory_records[relative] = _directory_record(
            path,
            fsync_metadata=False,
        )
    if final_directory_records != first_directories:
        raise OSError(
            errno.EIO,
            f"ACW tree directory metadata changed during readback: {root}",
        )
    for path in final_files:
        relative = path.relative_to(root).as_posix()
        descriptor = os.open(path, _open_flags(os.O_RDONLY))
        try:
            _require_name_bound_to_descriptor(
                path,
                descriptor,
                file_records[relative],
                directory=False,
            )
        finally:
            os.close(descriptor)

    terminal_files, terminal_directories = _tree_entries(root)
    terminal_file_names = [path.relative_to(root).as_posix() for path in terminal_files]
    terminal_directory_names = [
        "." if path == root else path.relative_to(root).as_posix()
        for path in terminal_directories
    ]
    if (
        terminal_file_names != final_file_names
        or terminal_directory_names != final_directory_names
    ):
        raise OSError(
            errno.EIO,
            f"ACW tree topology changed after file identity readback: {root}",
        )
    terminal_directory_records: dict[str, dict[str, Any]] = {}
    for path in terminal_directories:
        relative = "." if path == root else path.relative_to(root).as_posix()
        terminal_directory_records[relative] = _directory_record(
            path,
            fsync_metadata=False,
        )
    if terminal_directory_records != final_directory_records:
        raise OSError(
            errno.EIO,
            f"ACW tree directory metadata changed after file identity readback: {root}",
        )
    terminal_file_records: dict[str, dict[str, Any]] = {}
    terminal_file_descriptors: dict[str, tuple[Path, int]] = {}
    try:
        for path in terminal_files:
            relative = path.relative_to(root).as_posix()
            descriptor = os.open(path, _open_flags(os.O_RDONLY))
            terminal_file_descriptors[relative] = (path, descriptor)
            terminal_record = _descriptor_record_from_open_file(
                descriptor,
                path,
                fsync_file=False,
            )
            _require_name_bound_to_descriptor(
                path,
                descriptor,
                terminal_record,
                directory=False,
            )
            if _file_identity_record(terminal_record) != _file_identity_record(
                file_records[relative]
            ):
                raise OSError(
                    errno.EIO,
                    f"ACW tree file contents changed during terminal readback: {path}",
                )
            terminal_file_records[relative] = terminal_record

        barrier_files, barrier_directories = _tree_entries(root)
        barrier_file_names = [
            path.relative_to(root).as_posix() for path in barrier_files
        ]
        barrier_directory_names = [
            "." if path == root else path.relative_to(root).as_posix()
            for path in barrier_directories
        ]
        if (
            barrier_file_names != terminal_file_names
            or barrier_directory_names != terminal_directory_names
        ):
            raise OSError(
                errno.EIO,
                f"ACW tree topology changed during terminal readback: {root}",
            )
        barrier_directory_records: dict[str, dict[str, Any]] = {}
        for path in barrier_directories:
            relative = "." if path == root else path.relative_to(root).as_posix()
            barrier_directory_records[relative] = _directory_record(
                path,
                fsync_metadata=False,
            )
        if barrier_directory_records != terminal_directory_records:
            raise OSError(
                errno.EIO,
                f"ACW tree directory metadata changed during terminal readback: {root}",
            )
        for relative, (path, descriptor) in terminal_file_descriptors.items():
            _require_name_bound_to_descriptor(
                path,
                descriptor,
                terminal_file_records[relative],
                directory=False,
            )

        final_barrier_files, final_barrier_directories = _tree_entries(root)
        final_barrier_file_names = [
            path.relative_to(root).as_posix() for path in final_barrier_files
        ]
        final_barrier_directory_names = [
            "." if path == root else path.relative_to(root).as_posix()
            for path in final_barrier_directories
        ]
        if (
            final_barrier_file_names != barrier_file_names
            or final_barrier_directory_names != barrier_directory_names
        ):
            raise OSError(
                errno.EIO,
                f"ACW tree topology changed during terminal barrier: {root}",
            )
        final_barrier_directory_records: dict[str, dict[str, Any]] = {}
        for path in final_barrier_directories:
            relative = "." if path == root else path.relative_to(root).as_posix()
            final_barrier_directory_records[relative] = _directory_record(
                path,
                fsync_metadata=False,
            )
        if final_barrier_directory_records != barrier_directory_records:
            raise OSError(
                errno.EIO,
                f"ACW tree directory metadata changed during terminal barrier: {root}",
            )
        for relative, (path, descriptor) in terminal_file_descriptors.items():
            _require_name_bound_to_descriptor(
                path,
                descriptor,
                terminal_file_records[relative],
                directory=False,
            )
    finally:
        for _, descriptor in terminal_file_descriptors.values():
            os.close(descriptor)
    files = {
        relative: _file_identity_record(record)
        for relative, record in terminal_file_records.items()
    }
    directories = {
        relative: _directory_identity_record(record)
        for relative, record in terminal_directory_records.items()
    }
    return {"directories": directories, "files": files}


def _mode_is_literal(mode: int) -> bool:
    return mode == stat.S_IMODE(mode)


def _finalize_file_mode(path: Path, mode: int | None) -> dict[str, Any]:
    descriptor = os.open(path, _open_flags(os.O_RDONLY))
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"ACW artifact is not a regular file: {path}")
        if metadata.st_nlink != 1:
            raise ValueError(f"ACW artifact has unexpected hard links: {path}")
        if mode is not None:
            os.fchmod(descriptor, mode)
        record = _descriptor_record_from_open_file(
            descriptor,
            path,
            fsync_file=True,
        )
        _require_name_bound_to_descriptor(
            path,
            descriptor,
            record,
            directory=False,
        )
    finally:
        os.close(descriptor)
    if mode is not None and record["mode"] != f"{mode:04o}":
        raise OSError(errno.EIO, f"ACW artifact mode readback differs: {path}")
    return record


def _finalize_directory_mode(path: Path, mode: int | None) -> dict[str, Any]:
    flags = _open_flags(os.O_RDONLY) | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise NotADirectoryError(path)
        if mode is not None:
            os.fchmod(descriptor, mode)
        record = _directory_record_from_open_descriptor(
            descriptor,
            path,
            fsync_metadata=True,
        )
        _require_name_bound_to_descriptor(
            path,
            descriptor,
            record,
            directory=True,
        )
    finally:
        os.close(descriptor)
    if mode is not None and record["mode"] != f"{mode:04o}":
        raise OSError(errno.EIO, f"ACW directory mode readback differs: {path}")
    return record


def _durable_tree_snapshot(
    root: Path,
    *,
    file_mode: int | None,
    directory_mode: int | None,
) -> dict[str, Any]:
    if (file_mode is None) != (directory_mode is None):
        raise ValueError("ACW file and directory modes must both be set or omitted")
    if file_mode is not None and (
        not _mode_is_literal(file_mode)
        or directory_mode is None
        or not _mode_is_literal(directory_mode)
    ):
        raise ValueError("ACW publication modes must contain only permission bits")
    tree_files, tree_directories = _tree_entries(root)
    files = {
        path.relative_to(root).as_posix(): _file_identity_record(
            _finalize_file_mode(path, file_mode)
        )
        for path in tree_files
    }
    directories: dict[str, dict[str, Any]] = {}
    for path in sorted(
        tree_directories, key=lambda item: len(item.parts), reverse=True
    ):
        relative = "." if path == root else path.relative_to(root).as_posix()
        directories[relative] = _directory_identity_record(
            _finalize_directory_mode(path, directory_mode)
        )
    return {
        "directories": {key: directories[key] for key in sorted(directories)},
        "files": files,
    }


def _verify_tree_modes(
    snapshot: dict[str, Any],
    *,
    file_mode: int,
    directory_mode: int,
    root: Path,
) -> None:
    expected_file_mode = f"{file_mode:04o}"
    expected_directory_mode = f"{directory_mode:04o}"
    if any(
        record["mode"] != expected_file_mode for record in snapshot["files"].values()
    ):
        raise OSError(errno.EIO, f"ACW publication file modes differ: {root}")
    if any(
        record["mode"] != expected_directory_mode
        for record in snapshot["directories"].values()
    ):
        raise OSError(errno.EIO, f"ACW publication directory modes differ: {root}")


def _tree_custody_identity(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "directories": {
            relative: {key: record[key] for key in ("st_dev", "st_ino")}
            for relative, record in snapshot["directories"].items()
        },
        "files": {
            relative: {
                key: record[key] for key in ("bytes", "sha256", "st_dev", "st_ino")
            }
            for relative, record in snapshot["files"].items()
        },
    }


def verify_frozen_tree(
    root: Path,
    *,
    file_mode: int = 0o444,
    directory_mode: int = 0o555,
) -> dict[str, Any]:
    """Read back a complete immutable tree and require its final modes."""

    root = root.expanduser().absolute()
    with retained_tree_snapshot(root) as snapshot:
        _verify_tree_modes(
            snapshot,
            file_mode=file_mode,
            directory_mode=directory_mode,
            root=root,
        )
        return snapshot


def snapshot_tree(root: Path) -> dict[str, Any]:
    """Return a descriptor-bound snapshot of one complete literal tree."""

    with retained_tree_snapshot(root) as snapshot:
        return snapshot


def snapshot_file(path: Path) -> dict[str, Any]:
    """Return descriptor-bound content and inode identity for one literal file."""

    return _file_identity_record(
        _descriptor_record(path.expanduser().absolute(), fsync_file=False)
    )


@contextmanager
def retained_file_snapshot(path: Path) -> Iterator[dict[str, Any]]:
    """Retain one file descriptor and verify its name throughout a caller action."""

    path = path.expanduser().absolute()
    descriptor = os.open(path, _open_flags(os.O_RDONLY))
    try:
        record = _descriptor_record_from_open_file(
            descriptor,
            path,
            fsync_file=False,
        )
        _require_name_bound_to_descriptor(
            path,
            descriptor,
            record,
            directory=False,
        )
        try:
            yield _file_identity_record(record)
        finally:
            try:
                final_record = _descriptor_record_from_open_file(
                    descriptor,
                    path,
                    fsync_file=False,
                )
            except ValueError as error:
                raise OSError(
                    errno.EIO,
                    f"retained ACW artifact changed during caller action: {path}",
                ) from error
            if final_record != record:
                raise OSError(
                    errno.EIO,
                    f"retained ACW artifact changed during caller action: {path}",
                )
            _require_name_bound_to_descriptor(
                path,
                descriptor,
                final_record,
                directory=False,
            )
    finally:
        os.close(descriptor)


@contextmanager
def retained_directory_snapshot(path: Path) -> Iterator[dict[str, Any]]:
    """Retain one directory descriptor and verify its name through a caller action."""

    path = path.expanduser().absolute()
    flags = _open_flags(os.O_RDONLY) | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        record = _directory_record_from_open_descriptor(
            descriptor,
            path,
            fsync_metadata=False,
        )
        _require_name_bound_to_descriptor(
            path,
            descriptor,
            record,
            directory=True,
        )
        try:
            yield _directory_identity_record(record)
        finally:
            final_record = _directory_record_from_open_descriptor(
                descriptor,
                path,
                fsync_metadata=False,
            )
            if final_record != record:
                raise OSError(
                    errno.EIO,
                    f"retained ACW directory changed during caller action: {path}",
                )
            _require_name_bound_to_descriptor(
                path,
                descriptor,
                final_record,
                directory=True,
            )
    finally:
        os.close(descriptor)


def _sync_tree(
    root: Path,
    *,
    file_mode: int | None,
    directory_mode: int | None,
) -> dict[str, Any]:
    durable_snapshot = _durable_tree_snapshot(
        root,
        file_mode=file_mode,
        directory_mode=directory_mode,
    )
    readback_snapshot = _tree_snapshot(root, fsync_files=False)
    if file_mode is not None and directory_mode is not None:
        _verify_tree_modes(
            readback_snapshot,
            file_mode=file_mode,
            directory_mode=directory_mode,
            root=root,
        )
    if readback_snapshot != durable_snapshot:
        raise OSError(
            errno.EIO,
            f"durable ACW tree differs on descriptor readback: {root}",
        )
    return readback_snapshot


def freeze_tree(
    root: Path,
    *,
    file_mode: int = 0o444,
    directory_mode: int = 0o555,
    retained: RetainedEvidenceSnapshot | None = None,
) -> dict[str, Any]:
    """Set final tree modes, fsync all metadata, and verify descriptor identity."""

    root = root.expanduser().absolute()
    if retained is None:
        with retained_evidence_snapshot() as owned:
            return freeze_tree(
                root,
                file_mode=file_mode,
                directory_mode=directory_mode,
                retained=owned,
            )

    retained.retain_directory(root.parent)
    initial_snapshot = retained.retain_tree(root)
    snapshot = _sync_tree(
        root,
        file_mode=file_mode,
        directory_mode=directory_mode,
    )
    fsync_directory(root.parent)
    retained.refresh_directory(root.parent, fsync_metadata=True)
    final_snapshot = retained.refresh_tree(root)
    _verify_tree_modes(
        final_snapshot,
        file_mode=file_mode,
        directory_mode=directory_mode,
        root=root,
    )
    if final_snapshot != snapshot or _tree_custody_identity(
        final_snapshot
    ) != _tree_custody_identity(initial_snapshot):
        raise OSError(
            errno.EIO,
            f"frozen ACW tree changed after parent fsync: {root}",
        )
    return final_snapshot


def freeze_file(path: Path, *, mode: int = 0o444) -> dict[str, Any]:
    """Set one evidence file's final mode and durably verify its named inode."""

    if not _mode_is_literal(mode):
        raise ValueError("ACW publication mode must contain only permission bits")
    path = path.expanduser().absolute()
    descriptor = os.open(path, _open_flags(os.O_RDONLY))
    try:
        with retained_evidence_snapshot() as retained:
            retained.retain_directory(path.parent)
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"ACW artifact is not a regular file: {path}")
            if metadata.st_nlink != 1:
                raise ValueError(f"ACW artifact has unexpected hard links: {path}")
            os.fchmod(descriptor, mode)
            durable_record = _descriptor_record_from_open_file(
                descriptor,
                path,
                fsync_file=True,
            )
            _require_name_bound_to_descriptor(
                path,
                descriptor,
                durable_record,
                directory=False,
            )
            if durable_record["mode"] != f"{mode:04o}":
                raise OSError(errno.EIO, f"ACW artifact mode readback differs: {path}")
            fsync_directory(path.parent)
            retained.refresh_directory(path.parent, fsync_metadata=True)
            final_record = retained.retain_file(path)
            if final_record != _file_identity_record(durable_record):
                raise OSError(
                    errno.EIO,
                    f"frozen ACW artifact changed after parent fsync: {path}",
                )
            return _content_record(final_record)
    except OSError as error:
        if "frozen ACW artifact changed after parent fsync" in str(error):
            raise
        raise OSError(
            error.errno or errno.EIO,
            f"frozen ACW artifact changed after parent fsync: {path}",
        ) from error
    finally:
        os.close(descriptor)


def _restore_staging_modes(root: Path) -> None:
    tree_files, tree_directories = _tree_entries(root)
    for path in sorted(tree_directories, key=lambda item: len(item.parts)):
        path.chmod(0o700)
    for path in tree_files:
        path.chmod(0o600)


def publish_tree_no_replace(
    staging: Path,
    destination: Path,
    *,
    file_mode: int | None = None,
    directory_mode: int | None = None,
) -> dict[str, Any]:
    """Durably and atomically publish a complete staged directory tree."""

    staging = staging.expanduser().absolute()
    destination = destination.expanduser().absolute()
    if staging.parent != destination.parent:
        raise ValueError("ACW staging and destination must share one parent directory")
    if (
        file_mode is None
        and directory_mode is None
        and TREE_PUBLICATION_RECEIPT_FD_ENV in os.environ
    ):
        file_mode = 0o444
        directory_mode = 0o555
    published = False
    try:
        with retained_evidence_snapshot() as retained:
            retained.retain_directory(destination.parent)
            try:
                staged_snapshot = _sync_tree(
                    staging,
                    file_mode=file_mode,
                    directory_mode=directory_mode,
                )
                rename_no_replace(staging, destination)
                published = True
                fsync_directory(destination.parent)
                retained.refresh_directory(
                    destination.parent,
                    fsync_metadata=True,
                )
                final_snapshot = _retain_complete_tree(retained, destination)
                if final_snapshot != staged_snapshot:
                    raise OSError(
                        errno.EIO,
                        (
                            "published ACW directory differs on final descriptor readback: "
                            f"{destination}"
                        ),
                    )
                _emit_tree_publication_receipt(destination, final_snapshot)
                return final_snapshot
            except BaseException as error:
                if not published:
                    try:
                        retained.refresh_directory(
                            destination.parent,
                            fsync_metadata=False,
                        )
                    except BaseException as parent_error:
                        error.add_note(
                            f"ACW tree collision parent refresh failed: {parent_error}"
                        )
                raise
    except BaseException as error:
        if not published and os.path.lexists(staging):
            try:
                _restore_staging_modes(staging)
            except BaseException as cleanup_error:
                error.add_note(f"ACW tree staging mode cleanup failed: {cleanup_error}")
        raise


@contextmanager
def retained_bytes_publication(
    destination: Path,
    raw: bytes,
    *,
    mode: int = 0o444,
) -> Iterator[dict[str, Any]]:
    """Publish one immutable file and retain its descriptor for a caller action."""

    destination = destination.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = -1
    temporary: Path | None = None
    published = False
    try:
        with retained_evidence_snapshot() as retained:
            retained.retain_directory(destination.parent)
            try:
                descriptor, temporary_raw = tempfile.mkstemp(
                    prefix=f".{destination.name}.stage-",
                    dir=destination.parent,
                )
                temporary = Path(temporary_raw)
                _write_all(descriptor, raw)
                os.fchmod(descriptor, mode)
                os.fsync(descriptor)
                staged_record = _descriptor_record_from_open_file(
                    descriptor,
                    temporary,
                    fsync_file=False,
                )
                _require_name_bound_to_descriptor(
                    temporary,
                    descriptor,
                    staged_record,
                    directory=False,
                )
                expected = {
                    "bytes": len(raw),
                    "mode": f"{mode:04o}",
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
                if _content_record(staged_record) != expected:
                    raise OSError(
                        errno.EIO,
                        f"staged ACW file readback differs: {temporary}",
                    )
                writable_descriptor = descriptor
                descriptor = -1
                _close_descriptor(
                    writable_descriptor,
                    label="ACW writable staging",
                )
                descriptor = os.open(temporary, _open_flags(os.O_RDONLY))
                readonly_staged_record = _descriptor_record_from_open_file(
                    descriptor,
                    temporary,
                    fsync_file=False,
                )
                _require_name_bound_to_descriptor(
                    temporary,
                    descriptor,
                    readonly_staged_record,
                    directory=False,
                )
                if readonly_staged_record != staged_record:
                    raise OSError(
                        errno.EIO,
                        f"staged ACW file changed while dropping write access: {temporary}",
                    )
                rename_no_replace(temporary, destination)
                published = True
                fsync_directory(destination.parent)
                retained.refresh_directory(
                    destination.parent,
                    fsync_metadata=True,
                )
                record = _descriptor_record(destination, fsync_file=False)
                if _file_identity_record(record) != _file_identity_record(
                    staged_record
                ):
                    raise OSError(
                        errno.EIO,
                        (
                            "published ACW file differs on final descriptor readback: "
                            f"{destination}"
                        ),
                    )
                _require_name_bound_to_descriptor(
                    destination,
                    descriptor,
                    record,
                    directory=False,
                )
                retained_record = retained.retain_file(destination)
                if retained_record != _file_identity_record(record):
                    raise OSError(
                        errno.EIO,
                        (
                            "published ACW file differs on retained readback: "
                            f"{destination}"
                        ),
                    )
                yield retained_record
            except BaseException as error:
                if not published and temporary is not None:
                    try:
                        temporary.unlink()
                    except FileNotFoundError:
                        pass
                    except BaseException as cleanup_error:
                        error.add_note(
                            f"ACW file staging unlink failed: {cleanup_error}"
                        )
                try:
                    retained.refresh_directory(
                        destination.parent,
                        fsync_metadata=False,
                    )
                except BaseException as parent_error:
                    error.add_note(
                        f"ACW file collision parent refresh failed: {parent_error}"
                    )
                raise
    except BaseException as error:
        if descriptor >= 0:
            _close_descriptor(
                descriptor,
                label="ACW staged publication",
                primary_error=error,
            )
            descriptor = -1
        raise
    else:
        if descriptor >= 0:
            _close_descriptor(descriptor, label="ACW staged publication")


def publish_bytes_no_replace(
    destination: Path,
    raw: bytes,
    *,
    mode: int = 0o444,
) -> dict[str, Any]:
    """Durably publish one immutable file from a unique same-parent stage."""

    with retained_bytes_publication(destination, raw, mode=mode) as record:
        return _content_record(record)
