import errno
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from pipeline import build_acw_development_manifest as builder
from pipeline.acw_hidden_basis_training import (
    DEVELOPMENT_PLAN_PATH,
    DEVELOPMENT_PLAN_RAW_SHA256,
    file_sha256,
)
from pipeline.adjudicate_acw_hidden_basis import (
    DEVELOPMENT_SEEDS,
    DEVELOPMENT_MANIFEST_PROTOCOL,
    DIRECT_STATE_ARM,
    DIRECT_STATE_MANIFEST_PROTOCOL,
    SCORED_ARMS,
)


def _attempt_id(index: int, arm: str) -> str:
    return f"{arm}__{DEVELOPMENT_SEEDS[index]}"


class ACWDevelopmentManifestBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "attempt"
        self.root.mkdir()
        repository = Path(__file__).resolve().parents[1]
        plan = self.root / "development_plan.json"
        shutil.copyfile(repository / DEVELOPMENT_PLAN_PATH, plan)
        plan.chmod(0o444)
        attempt_start = self.root / "attempt_start.json"
        attempt_start.write_bytes(b"{}\n")
        attempt_start.chmod(0o444)
        self._immutable_file(self.root / "attempt_claim.json", b"{}\n")
        self._immutable_file(self.root / "direct_refit_verification.json", b"{}\n")
        self._immutable_file(self.root / "final_refit_verification.json", b"{}\n")
        for role in builder.ROLES:
            self._immutable_file(self.root / builder.ROLE_START_FILES[role], b"{}\n")
            self._immutable_file(
                self.root / builder.ROLE_COMPLETION_FILES[role], b"{}\n"
            )
            self._immutable_file(
                self.root / builder.ROLE_ACCOUNTING_FILES[role], b"{}\n"
            )

    def tearDown(self) -> None:
        for path in sorted(
            self.root.rglob("*"), key=lambda item: len(item.parts), reverse=True
        ):
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o700)
            elif path.exists() and not path.is_symlink():
                path.chmod(0o600)
        self.root.chmod(0o700)
        self.temporary.cleanup()

    def _immutable_file(self, path: Path, raw: bytes = b"artifact\n") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        path.chmod(0o444)

    def _immutable_root(self, path: Path) -> None:
        self._immutable_file(path / "manifest.json", b"{}\n")
        path.chmod(0o555)

    def _monitor_binding(self) -> dict:
        return {
            "job_id": "740999",
            "job_name": builder.MONITOR_JOB_NAME,
            "node": builder.MONITOR_NODE,
            "cpus_per_task": "4",
            "dependency": None,
            "script": {"path": builder.MONITOR_SCRIPT, "sha256": "a" * 64},
            "spool_script_sha256": "a" * 64,
            "scontrol_snapshot_sha256": "b" * 64,
            "process_membership": {
                "cpu_list": "0-3",
                "memory_list": "0",
                "task_cgroup": "/test",
            },
            "runtime_identity_sha256": "c" * 64,
        }

    def _add_run(self, index: int, arm: str) -> None:
        dataset = self.root / "inputs" / "datasets" / f"development_{index}"
        family = "uniform" if arm == "uniform_query_acw" else "cgb"
        bundle = self.root / "inputs" / "bundles" / f"development_{index}_{family}"
        task = self.root / "runs" / f"{index:02d}_{arm}"
        if not dataset.exists():
            self._immutable_root(dataset)
        if not bundle.exists():
            self._immutable_root(bundle)
        self._immutable_file(task / "checkpoint.pt")
        self._immutable_file(task / "evaluation.json", b"{}\n")
        self._immutable_file(task / "replay.json", b"{}\n")
        receipt = builder._hash_bound(
            {
                "schema": "r12_acw_development_attempt_receipt_v1",
                "protocol": "R12-ACW-DEVELOPMENT-ATTEMPT-v1",
                "attempt_id": _attempt_id(index, arm),
                "artifact_root": str(self.root.resolve(strict=True)),
                "task_root": task.relative_to(self.root).as_posix(),
                "completed_once": True,
            }
        )
        self._immutable_file(
            task / "attempt.json", builder.canonical_json_bytes(receipt) + b"\n"
        )

    def test_committed_plan_copy_is_exact_and_hash_bound(self) -> None:
        reference = builder._plan_reference(self.root)
        self.assertEqual(reference["sha256"], DEVELOPMENT_PLAN_RAW_SHA256)
        self.assertEqual(
            file_sha256(self.root / "development_plan.json"),
            DEVELOPMENT_PLAN_RAW_SHA256,
        )

    def test_plan_copy_uses_shared_no_replace_publication(self) -> None:
        publication_root = self.root / "plan-publication"
        for relative in (
            ".",
            "custody",
            "custody/stages",
            "inputs",
            "inputs/datasets",
            "inputs/bundles",
            "runs",
        ):
            path = publication_root / relative
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            path.chmod(0o700)
        raw = b'{"ready_for_g_commit":true}\n'
        expected_sha256 = hashlib.sha256(raw).hexdigest()
        with (
            mock.patch.object(
                builder,
                "DEVELOPMENT_PLAN_RAW_SHA256",
                expected_sha256,
            ),
            mock.patch.object(
                builder,
                "_load_committed_plan_with_raw",
                return_value=({}, raw),
            ) as load_plan,
        ):
            observed_sha256 = builder.publish_development_plan_copy(publication_root)
            with self.assertRaises(FileExistsError):
                builder.publish_development_plan_copy(publication_root)
        load_plan.assert_called_with(require_ready=True)
        published = publication_root / builder.PLAN_COPY_NAME
        self.assertEqual(observed_sha256, expected_sha256)
        self.assertEqual(published.read_bytes(), raw)
        self.assertEqual(stat.S_IMODE(published.stat().st_mode), 0o444)

    def test_layout_durability_fsyncs_every_directory_and_the_root_parent(
        self,
    ) -> None:
        layout_root = Path(self.temporary.name) / "durable-layout"
        expected = {
            layout_root / relative
            for relative in (
                ".",
                "custody",
                "custody/stages",
                "inputs",
                "inputs/datasets",
                "inputs/bundles",
                "runs",
            )
        }
        for path in sorted(expected, key=lambda item: len(item.parts)):
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            path.chmod(0o700)
        with mock.patch.object(
            builder.publication,
            "fsync_directory",
            wraps=builder.publication.fsync_directory,
        ) as fsync_directory:
            builder._durably_verify_role_layout(layout_root, builder.ROLE_PHASE1)
        observed = {Path(call.args[0]) for call in fsync_directory.call_args_list}
        self.assertEqual(observed, expected | {layout_root.parent})

    def test_direct_manifest_has_exact_three_seed_matrix(self) -> None:
        for index in range(3):
            self._add_run(index, DIRECT_STATE_ARM)
        manifest = builder.build_direct_manifest(self.root)
        self.assertEqual(manifest["protocol"], DIRECT_STATE_MANIFEST_PROTOCOL)
        self.assertEqual(len(manifest["reports"]), 3)
        self.assertEqual(
            [record["arm"] for record in manifest["reports"]],
            [DIRECT_STATE_ARM] * 3,
        )
        self.assertEqual(
            [record.get("attempt_id") for record in manifest["reports"]],
            [_attempt_id(index, DIRECT_STATE_ARM) for index in range(3)],
        )
        self.assertEqual(
            set(manifest.get("stage_receipts", {})),
            {"phase1_producer", "phase1_verifier"},
        )
        self.assertEqual(manifest["attempt_start"]["path"], "attempt_start.json")

    def test_development_manifest_has_exact_fixed_matrix_and_authorization(
        self,
    ) -> None:
        for arm in (*SCORED_ARMS, DIRECT_STATE_ARM):
            for index in range(3):
                self._add_run(index, arm)
        self._immutable_file(self.root / "phase2_authorization.json", b"{}\n")
        manifest = builder.build_development_manifest(self.root)
        self.assertEqual(manifest["protocol"], DEVELOPMENT_MANIFEST_PROTOCOL)
        self.assertEqual(len(manifest["reports"]), 27)
        self.assertEqual(
            [record["arm"] for record in manifest["reports"]],
            [arm for arm in (DIRECT_STATE_ARM, *SCORED_ARMS) for _ in range(3)],
        )
        expected_attempts = [
            _attempt_id(index, arm)
            for arm in (DIRECT_STATE_ARM, *SCORED_ARMS)
            for index in range(3)
        ]
        self.assertEqual(
            [record.get("attempt_id") for record in manifest["reports"]],
            expected_attempts,
        )
        self.assertEqual(len(set(expected_attempts)), 27)
        self.assertEqual(
            set(manifest.get("stage_receipts", {})),
            {
                "phase1_producer",
                "phase1_verifier",
                "phase2_producer",
                "phase2_verifier",
            },
        )
        self.assertEqual(
            manifest["phase2_authorization"]["path"],
            "phase2_authorization.json",
        )

    def test_hidden_or_extra_attempt_tree_is_rejected(self) -> None:
        for index in range(3):
            self._add_run(index, DIRECT_STATE_ARM)
        self._immutable_file(
            self.root / "runs" / ".discarded_attempt" / "checkpoint.pt"
        )
        with self.assertRaisesRegex(ValueError, "attempt inventory|unexpected"):
            builder.build_direct_manifest(self.root)

    def test_missing_expected_attempt_is_rejected(self) -> None:
        for index in range(2):
            self._add_run(index, DIRECT_STATE_ARM)
        with self.assertRaises((FileNotFoundError, ValueError)):
            builder.build_direct_manifest(self.root)

    def test_off_root_attempt_via_intermediate_symlink_is_rejected(self) -> None:
        for index in range(3):
            self._add_run(index, DIRECT_STATE_ARM)
        task = self.root / "runs" / f"00_{DIRECT_STATE_ARM}"
        external = Path(self.temporary.name) / "external_attempt"
        shutil.copytree(task, external)
        shutil.rmtree(task)
        task.symlink_to(external, target_is_directory=True)
        with self.assertRaises((ValueError, RuntimeError)):
            builder.build_direct_manifest(self.root)

    def test_off_root_attempt_receipt_cannot_be_silently_ignored(self) -> None:
        for index in range(3):
            self._add_run(index, DIRECT_STATE_ARM)
        task = self.root / "runs" / f"00_{DIRECT_STATE_ARM}"
        receipt = builder._hash_bound(
            {
                "schema": "r12_acw_run_attempt_v1",
                "attempt_id": _attempt_id(0, DIRECT_STATE_ARM),
                "artifact_root": str(Path(self.temporary.name) / "outside"),
            }
        )
        receipt_path = task / "attempt.json"
        receipt_path.chmod(0o600)
        receipt_path.write_bytes(builder.canonical_json_bytes(receipt) + b"\n")
        receipt_path.chmod(0o444)
        with self.assertRaisesRegex(ValueError, "off-root|artifact root|attempt"):
            builder.build_direct_manifest(self.root)

    def test_private_tree_comparison_opens_every_registered_file(self) -> None:
        producer = Path(self.temporary.name) / "producer_dataset"
        verifier = Path(self.temporary.name) / "verifier_dataset"
        manifest = builder._hash_bound({"arrays": {"array.bin": {}}})
        raw_manifest = builder.canonical_json_bytes(manifest) + b"\n"
        for tree in (producer, verifier):
            tree.mkdir()
            (tree / "manifest.json").write_bytes(raw_manifest)
            (tree / "array.bin").write_bytes(b"original")
        builder._require_registered_tree_bytes_equal(producer, verifier, kind="dataset")
        (verifier / "array.bin").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "regeneration differs"):
            builder._require_registered_tree_bytes_equal(
                producer, verifier, kind="dataset"
            )

    def test_exclusive_writer_freezes_bytes_and_refuses_reuse(self) -> None:
        output = self.root / "manifest.json"
        payload = builder._hash_bound({"schema": "test"})
        digest = builder.write_exclusive(output, payload)
        self.assertEqual(digest, file_sha256(output))
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o444)
        self.assertEqual(json.loads(output.read_bytes()), payload)
        with self.assertRaises(FileExistsError):
            builder.write_exclusive(output, payload)

    def test_run_attempt_preserves_collision_over_cleanup_freeze_failure(self) -> None:
        role = builder.ROLE_PHASE1
        job_id = "740999"
        task = self.root / "collision-attempt"
        inputs = {
            name: self.root / f"{name}-input"
            for name in ("dataset", "bundle", "curriculum")
        }
        for path in inputs.values():
            path.mkdir()
        outputs = {
            "checkpoint": task / "checkpoint.pt",
            "evaluation": task / "evaluation.json",
            "replay": task / "replay.json",
        }
        side = {
            "job_id": job_id,
            "node": builder.ROLE_NODES[role],
            "paths": {
                **{name: str(path) for name, path in inputs.items()},
                "task_root": str(task),
                **{name: str(path) for name, path in outputs.items()},
            },
            "train_argv": ["train"],
            "evaluation_argv": ["evaluate"],
            "replay_argv": ["replay"],
        }
        attempt = {
            "attempt_id": "collision-attempt",
            "logical_arm": DIRECT_STATE_ARM,
            "trainer_arm": DIRECT_STATE_ARM,
            "seed": DEVELOPMENT_SEEDS[0],
        }
        stage = {"held_slurm_job_id": job_id}
        collision = FileExistsError(
            errno.EEXIST,
            "native attempt receipt collision",
            task / "attempt.json",
        )
        freeze_failure = OSError(errno.EIO, "injected attempt cleanup freeze failure")
        role_roots = dict(builder.ROLE_ROOTS)
        role_roots[role] = self.root

        def create_outputs(_argv: list[str], *, label: str) -> None:
            del label
            for name, path in outputs.items():
                if not path.exists():
                    raw = b"checkpoint" if name == "checkpoint" else b"{}\n"
                    path.write_bytes(raw)
                    path.chmod(0o444)

        with (
            mock.patch.object(builder, "_plan_for_root", return_value={}),
            mock.patch.object(builder, "validate_plan"),
            mock.patch.object(builder, "_durably_verify_role_layout"),
            mock.patch.object(builder, "_attempt", return_value=(attempt, side)),
            mock.patch.object(builder, "_stage_by_role", return_value=stage),
            mock.patch.object(builder, "_current_job_binding", return_value={}),
            mock.patch.object(builder, "_run_argv", side_effect=create_outputs),
            mock.patch.object(builder, "write_exclusive", side_effect=collision),
            mock.patch.object(
                builder.publication,
                "freeze_tree",
                side_effect=freeze_failure,
            ) as freeze_tree,
            mock.patch.object(builder, "ROLE_ROOTS", role_roots),
            mock.patch.dict(os.environ, {"SLURM_JOB_ID": job_id}),
            self.assertRaises(FileExistsError) as raised,
        ):
            builder.run_attempt(self.root, role, 0, DIRECT_STATE_ARM)
        self.assertIs(raised.exception, collision)
        self.assertTrue(
            any(
                "attempt cleanup freeze failed" in note
                for note in raised.exception.__notes__
            )
        )
        freeze_tree.assert_called_once_with(task)

    def test_atomic_publication_uses_unique_stage_and_preserves_foreign_temp(
        self,
    ) -> None:
        output = self.root / "atomic.bin"
        temporary = builder._protocol_publish_temp(output)
        temporary.write_bytes(b"truncated")
        temporary.chmod(0o444)
        raw = b"complete immutable bytes"
        real_rename = builder.publication.rename_no_replace

        def checked_rename(source: Path, destination: Path) -> None:
            self.assertFalse(os.path.lexists(destination))
            self.assertEqual(Path(source).read_bytes(), raw)
            self.assertTrue(Path(source).name.startswith(".atomic.bin.stage-"))
            real_rename(source, destination)

        with mock.patch.object(
            builder.publication,
            "rename_no_replace",
            side_effect=checked_rename,
        ):
            digest = builder._atomic_publish_bytes(output, raw)
        self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
        self.assertEqual(output.read_bytes(), raw)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o444)
        self.assertEqual(temporary.read_bytes(), b"truncated")
        self.assertEqual(list(self.root.glob(".atomic.bin.stage-*")), [])
        with self.assertRaises(FileExistsError):
            builder._atomic_publish_bytes(output, b"replacement")
        self.assertEqual(output.read_bytes(), raw)

    def test_directory_fsync_propagates_io_and_unexpected_failures(self) -> None:
        for error_number in (errno.EIO, errno.EPERM):
            with (
                self.subTest(error_number=error_number),
                mock.patch.object(
                    builder.publication.os,
                    "fsync",
                    side_effect=OSError(
                        error_number, "injected directory fsync failure"
                    ),
                ),
            ):
                with self.assertRaises(OSError) as raised:
                    builder.publication.fsync_directory(self.root)
            self.assertEqual(raised.exception.errno, error_number)

    def test_atomic_publication_never_reports_success_after_directory_fsync_eio(
        self,
    ) -> None:
        output = self.root / "fsync-failure.bin"
        temporary = builder._protocol_publish_temp(output)
        raw = b"linked but not durably acknowledged"
        failure = OSError(errno.EIO, "injected directory fsync failure")
        with (
            mock.patch.object(
                builder.publication,
                "fsync_directory",
                side_effect=failure,
            ),
            self.assertRaises(OSError) as raised,
        ):
            builder._atomic_publish_bytes(output, raw)
        self.assertEqual(raised.exception.errno, errno.EIO)
        self.assertTrue(output.is_file())
        self.assertEqual(output.read_bytes(), raw)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o444)
        self.assertFalse(os.path.lexists(temporary))
        self.assertEqual(list(self.root.glob(".fsync-failure.bin.stage-*")), [])
        with self.assertRaises(FileExistsError):
            builder._atomic_publish_bytes(output, b"replacement")

    def test_file_publication_drops_write_capability_before_no_replace(self) -> None:
        output = self.root / "no-write-capability.bin"
        raw = b"write capability must end before immutable publication"
        real_mkstemp = builder.publication.tempfile.mkstemp
        real_rename = builder.publication.rename_no_replace
        writable_descriptor = -1
        checked = False

        def capture_descriptor(*args, **kwargs):
            nonlocal writable_descriptor
            writable_descriptor, temporary = real_mkstemp(*args, **kwargs)
            return writable_descriptor, temporary

        def require_no_write_capability(source: Path, destination: Path) -> None:
            nonlocal checked
            with self.assertRaises(OSError) as denied:
                os.write(writable_descriptor, b"unexpected mutation")
            self.assertEqual(denied.exception.errno, errno.EBADF)
            checked = True
            real_rename(source, destination)

        with (
            mock.patch.object(
                builder.publication.tempfile,
                "mkstemp",
                side_effect=capture_descriptor,
            ),
            mock.patch.object(
                builder.publication,
                "rename_no_replace",
                side_effect=require_no_write_capability,
            ),
        ):
            builder._atomic_publish_bytes(output, raw)
        self.assertTrue(checked)
        self.assertEqual(output.read_bytes(), raw)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o444)

    def test_ambient_exception_cannot_suppress_writable_close_failure(self) -> None:
        output = self.root / "ambient-exception-publication.bin"
        real_mkstemp = builder.publication.tempfile.mkstemp
        real_close = os.close
        writable_descriptor = -1
        close_injected = False
        close_failure = OSError(errno.EIO, "injected mandatory writable close failure")

        def capture_descriptor(*args, **kwargs):
            nonlocal writable_descriptor
            writable_descriptor, temporary = real_mkstemp(*args, **kwargs)
            return writable_descriptor, temporary

        def fail_writable_close(descriptor: int) -> None:
            nonlocal close_injected
            if descriptor == writable_descriptor and not close_injected:
                close_injected = True
                raise close_failure
            real_close(descriptor)

        try:
            with (
                mock.patch.object(
                    builder.publication.tempfile,
                    "mkstemp",
                    side_effect=capture_descriptor,
                ),
                mock.patch.object(
                    builder.publication.os,
                    "close",
                    side_effect=fail_writable_close,
                ),
            ):
                try:
                    raise RuntimeError("ambient caller exception")
                except RuntimeError:
                    with self.assertRaises(OSError) as raised:
                        builder._atomic_publish_bytes(output, b"must not publish")
            self.assertIs(raised.exception, close_failure)
            self.assertTrue(close_injected)
            self.assertFalse(os.path.lexists(output))
        finally:
            if writable_descriptor >= 0:
                os.write(writable_descriptor, b"descriptor remained open after failure")
                real_close(writable_descriptor)

    def test_file_publication_rejects_byte_identical_inode_substitution_after_parent_fsync(
        self,
    ) -> None:
        output = self.root / "inode-substitution.bin"
        raw = b"byte-identical replacement"
        original_identity: tuple[int, int] | None = None
        replacement_identity: tuple[int, int] | None = None
        real_fsync_directory = builder.publication.fsync_directory

        def substitute_after_fsync(parent: Path) -> None:
            nonlocal original_identity, replacement_identity
            real_fsync_directory(parent)
            metadata = output.stat()
            original_identity = (metadata.st_dev, metadata.st_ino)
            observed = output.read_bytes()
            output.unlink()
            output.write_bytes(observed)
            output.chmod(0o444)
            metadata = output.stat()
            replacement_identity = (metadata.st_dev, metadata.st_ino)

        with (
            mock.patch.object(
                builder.publication,
                "fsync_directory",
                side_effect=substitute_after_fsync,
            ),
            self.assertRaisesRegex(OSError, "final descriptor readback"),
        ):
            builder._atomic_publish_bytes(output, raw)
        self.assertIsNotNone(original_identity)
        self.assertIsNotNone(replacement_identity)
        self.assertNotEqual(original_identity, replacement_identity)
        self.assertEqual(output.read_bytes(), raw)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o444)
        record = builder.publication._descriptor_record(output, fsync_file=False)
        self.assertEqual(record["st_dev"], output.stat().st_dev)
        self.assertEqual(record["st_ino"], output.stat().st_ino)

    def test_file_publication_rejects_parent_substitution_after_parent_fsync(
        self,
    ) -> None:
        canonical_parent = self.root / "canonical-publication-parent"
        canonical_parent.mkdir()
        replacement_parent = self.root / "replacement-publication-parent"
        replacement_parent.mkdir()
        displaced_parent = self.root / "displaced-publication-parent"
        output = canonical_parent / "artifact.bin"
        raw = b"same published inode under an unsynced replacement parent"
        real_fsync_directory = builder.publication.fsync_directory
        substituted = False
        published_identity: tuple[int, int] | None = None
        installed_identity: tuple[int, int] | None = None

        def substitute_parent_after_fsync(parent: Path) -> None:
            nonlocal substituted, published_identity, installed_identity
            real_fsync_directory(parent)
            if Path(parent) == canonical_parent and not substituted:
                metadata = output.stat()
                published_identity = (metadata.st_dev, metadata.st_ino)
                canonical_parent.rename(displaced_parent)
                replacement_parent.rename(canonical_parent)
                (displaced_parent / output.name).rename(output)
                metadata = output.stat()
                installed_identity = (metadata.st_dev, metadata.st_ino)
                substituted = True

        with (
            mock.patch.object(
                builder.publication,
                "fsync_directory",
                side_effect=substitute_parent_after_fsync,
            ),
            self.assertRaisesRegex(
                OSError,
                "retained ACW|descriptor name changed",
            ),
        ):
            builder._atomic_publish_bytes(output, raw)
        self.assertTrue(substituted)
        self.assertIsNotNone(published_identity)
        self.assertEqual(published_identity, installed_identity)
        self.assertEqual(output.read_bytes(), raw)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o444)

    def test_joint_retained_evidence_rechecks_after_first_name_pass(self) -> None:
        artifact = self.root / "joint-retained-artifact.bin"
        later = self.root / "joint-retained-later.bin"
        replacement = self.root / "joint-retained-replacement.bin"
        raw = b"byte-identical joint retained replacement"
        for path, value in (
            (artifact, raw),
            (later, b"later retained evidence"),
            (replacement, raw),
        ):
            path.write_bytes(value)
            path.chmod(0o444)
        original_identity = (artifact.stat().st_dev, artifact.stat().st_ino)
        swapped = False
        real_verify_names = builder.publication.RetainedEvidenceSnapshot._verify_names

        def swap_after_first_name_pass(
            retained,
            *,
            reverse: bool = False,
        ) -> None:
            nonlocal swapped
            real_verify_names(retained, reverse=reverse)
            if not swapped:
                artifact.unlink()
                replacement.rename(artifact)
                swapped = True

        with (
            mock.patch.object(
                builder.publication.RetainedEvidenceSnapshot,
                "_verify_names",
                autospec=True,
                side_effect=swap_after_first_name_pass,
            ),
            self.assertRaisesRegex(OSError, "retained ACW"),
        ):
            with builder.publication.retained_evidence_snapshot() as retained:
                retained.retain_directory(self.root)
                retained.retain_file(artifact)
                retained.retain_file(later)
        self.assertTrue(swapped)
        self.assertNotEqual(
            original_identity,
            (artifact.stat().st_dev, artifact.stat().st_ino),
        )
        self.assertEqual(artifact.read_bytes(), raw)

    def test_joint_retained_evidence_rechecks_after_descriptor_rotation(self) -> None:
        artifact = self.root / "rotation-retained-artifact.bin"
        replacement = self.root / "rotation-retained-replacement.bin"
        raw = b"byte-identical replacement after terminal retained pass"
        for path in (artifact, replacement):
            path.write_bytes(raw)
            path.chmod(0o444)
        original_identity = (artifact.stat().st_dev, artifact.stat().st_ino)
        rotated = False
        real_rotate = builder.publication.RetainedEvidenceSnapshot.rotate_descriptors

        def rotate_then_replace(retained) -> None:
            nonlocal rotated
            real_rotate(retained)
            artifact.unlink()
            replacement.rename(artifact)
            rotated = True

        with (
            mock.patch.object(
                builder.publication.RetainedEvidenceSnapshot,
                "rotate_descriptors",
                autospec=True,
                side_effect=rotate_then_replace,
            ),
            self.assertRaisesRegex(OSError, "retained ACW"),
        ):
            with builder.publication.retained_evidence_snapshot() as retained:
                retained.retain_directory(self.root)
                retained.retain_file(artifact)
        self.assertTrue(rotated)
        self.assertNotEqual(
            original_identity,
            (artifact.stat().st_dev, artifact.stat().st_ino),
        )

    def test_joint_retained_evidence_rechecks_open_writer_after_rotation(self) -> None:
        artifact = self.root / "rotation-writer-artifact.bin"
        artifact.write_bytes(b"ORIGINAL")
        writable = os.open(artifact, os.O_WRONLY)
        artifact.chmod(0o444)
        mutated = False
        real_rotate = builder.publication.RetainedEvidenceSnapshot.rotate_descriptors

        def rotate_then_mutate(retained) -> None:
            nonlocal mutated
            real_rotate(retained)
            os.lseek(writable, 0, os.SEEK_SET)
            os.write(writable, b"MUTATED!")
            os.fsync(writable)
            mutated = True

        try:
            with (
                mock.patch.object(
                    builder.publication.RetainedEvidenceSnapshot,
                    "rotate_descriptors",
                    autospec=True,
                    side_effect=rotate_then_mutate,
                ),
                self.assertRaisesRegex(OSError, "retained ACW artifact changed"),
            ):
                with builder.publication.retained_evidence_snapshot() as retained:
                    retained.retain_directory(self.root)
                    retained.retain_file(artifact)
        finally:
            os.close(writable)
        self.assertTrue(mutated)

    def test_failed_retained_barrier_preserves_native_collision(self) -> None:
        collision = self.root / "raced-native-collision.bin"
        raised = FileExistsError(errno.EEXIST, "native no-replace collision", collision)
        with self.assertRaises(FileExistsError) as observed:
            with builder.publication.retained_evidence_snapshot() as retained:
                retained.retain_directory(self.root)
                collision.write_bytes(b"competing publication")
                raise raised
        self.assertIs(observed.exception, raised)

    def test_retained_rotation_tracks_original_after_close_failure(self) -> None:
        artifact = self.root / "rotation-close-failure.bin"
        artifact.write_bytes(b"retained close failure")
        retained = builder.publication.RetainedEvidenceSnapshot()
        retained.retain_file(artifact)
        canonical = artifact.absolute()
        original = retained._files[canonical][0]
        failure = OSError(errno.EIO, "injected descriptor rotation close failure")
        real_close = os.close
        injected = False

        def fail_original_close(descriptor: int) -> None:
            nonlocal injected
            if descriptor == original and not injected:
                injected = True
                raise failure
            real_close(descriptor)

        with (
            mock.patch.object(
                builder.publication.os,
                "close",
                side_effect=fail_original_close,
            ),
            self.assertRaises(OSError) as raised,
        ):
            retained.rotate_descriptors()
        self.assertIs(raised.exception, failure)
        self.assertTrue(injected)
        self.assertIn(original, retained._extra_descriptors)
        self.assertNotEqual(retained._files[canonical][0], original)
        retained.close()

    def test_retained_barrier_preserves_dup_error_over_close_error(self) -> None:
        artifact = self.root / "rotation-dup-failure.bin"
        artifact.write_bytes(b"retained dup failure")
        duplicate_failure = OSError(errno.EMFILE, "injected descriptor dup failure")
        cleanup_failure = OSError(errno.EIO, "injected retained cleanup failure")
        real_close = builder.publication.RetainedEvidenceSnapshot.close

        def close_then_fail(retained) -> None:
            real_close(retained)
            raise cleanup_failure

        with (
            mock.patch.object(
                builder.publication.os,
                "dup",
                side_effect=duplicate_failure,
            ),
            mock.patch.object(
                builder.publication.RetainedEvidenceSnapshot,
                "close",
                autospec=True,
                side_effect=close_then_fail,
            ),
            self.assertRaises(OSError) as raised,
        ):
            with builder.publication.retained_evidence_snapshot() as retained:
                retained.retain_file(artifact)
        self.assertIs(raised.exception, duplicate_failure)
        self.assertTrue(
            any(
                "retained ACW descriptor cleanup failed" in note
                for note in raised.exception.__notes__
            )
        )

    def test_native_collision_survives_unlink_refresh_and_close_failures(self) -> None:
        destination = self.root / "triple-fault-native-collision.bin"
        destination.write_bytes(b"competing immutable publication")
        destination.chmod(0o444)
        unlink_failure = OSError(errno.EPERM, "injected staging unlink failure")
        refresh_failure = OSError(errno.EIO, "injected parent refresh failure")
        close_failure = OSError(errno.EIO, "injected staging close failure")
        real_close = os.close
        real_rename = builder.publication.rename_no_replace
        close_injected = False
        collision_attempted = False

        def mark_collision_attempt(source: Path, target: Path) -> None:
            nonlocal collision_attempted
            try:
                real_rename(source, target)
            finally:
                collision_attempted = True

        def fail_regular_close(descriptor: int) -> None:
            nonlocal close_injected
            metadata = os.fstat(descriptor)
            real_close(descriptor)
            if (
                collision_attempted
                and stat.S_ISREG(metadata.st_mode)
                and not close_injected
            ):
                close_injected = True
                raise close_failure

        with (
            mock.patch.object(Path, "unlink", side_effect=unlink_failure),
            mock.patch.object(
                builder.publication.RetainedEvidenceSnapshot,
                "refresh_directory",
                autospec=True,
                side_effect=refresh_failure,
            ),
            mock.patch.object(
                builder.publication.os,
                "close",
                side_effect=fail_regular_close,
            ),
            mock.patch.object(
                builder.publication,
                "rename_no_replace",
                side_effect=mark_collision_attempt,
            ),
            self.assertRaises(FileExistsError) as raised,
        ):
            builder.publication.publish_bytes_no_replace(destination, b"new bytes")
        self.assertEqual(raised.exception.errno, errno.EEXIST)
        self.assertTrue(close_injected)
        notes = raised.exception.__notes__
        self.assertTrue(any("staging unlink failed" in note for note in notes))
        self.assertTrue(any("parent refresh failed" in note for note in notes))
        self.assertTrue(any("descriptor cleanup failed" in note for note in notes))
        for temporary in self.root.glob(f".{destination.name}.stage-*"):
            temporary.unlink()

    def test_file_publication_rejects_byte_identical_inode_substitution_after_final_open(
        self,
    ) -> None:
        output = self.root / "open-inode-substitution.bin"
        raw = b"byte-identical after-open replacement"
        original_identity: tuple[int, int] | None = None
        replacement_identity: tuple[int, int] | None = None
        substituted = False
        real_record = builder.publication._descriptor_record_from_open_file

        def substitute_after_readback(
            descriptor: int,
            path: Path,
            *,
            fsync_file: bool,
        ) -> dict:
            nonlocal original_identity, replacement_identity, substituted
            record = real_record(
                descriptor,
                path,
                fsync_file=fsync_file,
            )
            if Path(path) == output and not substituted:
                substituted = True
                metadata = os.fstat(descriptor)
                original_identity = (metadata.st_dev, metadata.st_ino)
                output.unlink()
                output.write_bytes(raw)
                output.chmod(0o444)
                metadata = output.stat()
                replacement_identity = (metadata.st_dev, metadata.st_ino)
            return record

        with (
            mock.patch.object(
                builder.publication,
                "_descriptor_record_from_open_file",
                side_effect=substitute_after_readback,
            ),
            self.assertRaisesRegex(OSError, "descriptor name changed"),
        ):
            builder._atomic_publish_bytes(output, raw)
        self.assertTrue(substituted)
        self.assertIsNotNone(original_identity)
        self.assertIsNotNone(replacement_identity)
        self.assertNotEqual(original_identity, replacement_identity)
        self.assertEqual(output.read_bytes(), raw)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o444)

    def test_file_publication_rejects_substitution_after_final_record_returns(
        self,
    ) -> None:
        output = self.root / "post-record-inode-substitution.bin"
        raw = b"byte-identical post-record replacement"
        substituted = False
        real_record = builder.publication._descriptor_record

        def substitute_after_final_record(path: Path, *, fsync_file: bool) -> dict:
            nonlocal substituted
            record = real_record(path, fsync_file=fsync_file)
            if Path(path) == output and not substituted:
                substituted = True
                output.unlink()
                output.write_bytes(raw)
                output.chmod(0o444)
            return record

        with (
            mock.patch.object(
                builder.publication,
                "_descriptor_record",
                side_effect=substitute_after_final_record,
            ),
            self.assertRaisesRegex(OSError, "descriptor name changed"),
        ):
            builder._atomic_publish_bytes(output, raw)
        self.assertTrue(substituted)
        self.assertEqual(output.read_bytes(), raw)

    def test_tree_publication_installs_final_modes_inside_durable_contract(
        self,
    ) -> None:
        destination = self.root / "published-tree"
        staging = builder.publication.create_staging_directory(destination)
        artifact = staging / "nested" / "artifact.bin"
        builder.publication.write_file_exclusive(artifact, b"tree bytes")
        with mock.patch.object(
            builder.publication.os,
            "fsync",
            wraps=os.fsync,
        ) as fsync:
            published_snapshot = builder.publication.publish_tree_no_replace(
                staging,
                destination,
                file_mode=0o444,
                directory_mode=0o555,
            )
            snapshot = builder.publication.freeze_tree(destination)
        published = destination / "nested" / "artifact.bin"
        self.assertGreaterEqual(fsync.call_count, 7)
        self.assertEqual(
            published_snapshot["files"]["nested/artifact.bin"]["mode"],
            "0444",
        )
        self.assertEqual(stat.S_IMODE(published.stat().st_mode), 0o444)
        self.assertEqual(stat.S_IMODE((destination / "nested").stat().st_mode), 0o555)
        self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o555)
        self.assertEqual(snapshot["files"]["nested/artifact.bin"]["mode"], "0444")
        self.assertEqual(snapshot["directories"]["nested"]["mode"], "0555")
        self.assertEqual(snapshot["directories"]["."]["mode"], "0555")
        self.assertEqual(
            snapshot["files"]["nested/artifact.bin"]["st_ino"],
            published.stat().st_ino,
        )
        self.assertEqual(published_snapshot, snapshot)

    def test_child_tree_publication_receipt_rejects_post_exit_substitution(
        self,
    ) -> None:
        destination = self.root / "child-published-tree"
        replacement = self.root / "child-published-replacement"
        replacement_staging = builder.publication.create_staging_directory(replacement)
        builder.publication.write_file_exclusive(
            replacement_staging / "artifact.bin",
            b"byte-identical child output",
        )
        builder.publication.publish_tree_no_replace(
            replacement_staging,
            replacement,
            file_mode=0o444,
            directory_mode=0o555,
        )
        displaced = self.root / "child-published-displaced"
        published_identity: tuple[int, int] | None = None
        replacement_identity = (replacement.stat().st_dev, replacement.stat().st_ino)

        def publish_then_substitute(argv, **kwargs):
            nonlocal published_identity
            del argv
            staging = builder.publication.create_staging_directory(destination)
            builder.publication.write_file_exclusive(
                staging / "artifact.bin",
                b"byte-identical child output",
            )
            snapshot = builder.publication.publish_tree_no_replace(
                staging,
                destination,
                file_mode=0o444,
                directory_mode=0o555,
            )
            metadata = destination.stat()
            published_identity = (metadata.st_dev, metadata.st_ino)
            receipt = {
                "protocol": "ACW-TREE-PUBLICATION-RECEIPT-v1",
                "destination": str(destination.absolute()),
                "snapshot": snapshot,
            }
            descriptor = kwargs["pass_fds"][0]
            os.write(descriptor, builder.canonical_json_bytes(receipt) + b"\n")
            os.fsync(descriptor)
            destination.rename(displaced)
            replacement.rename(destination)

        with (
            mock.patch.object(
                builder.subprocess,
                "run",
                side_effect=publish_then_substitute,
            ),
            self.assertRaisesRegex(OSError, "identity changed after child publication"),
        ):
            builder._run_argv(
                [str(builder.PYTHON), "-S", "-P", "publisher.py"],
                label="injected child",
                expected_tree_publication=destination,
            )
        self.assertIsNotNone(published_identity)
        self.assertNotEqual(published_identity, replacement_identity)

    def test_child_receipt_channel_enforces_final_modes_in_real_subprocess(
        self,
    ) -> None:
        destination = self.root / "real-child-published-tree"
        repository = Path(__file__).resolve().parents[1]
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(repository)!r}); "
            "from pathlib import Path; "
            "from pipeline import acw_immutable_publication as publication; "
            "destination = Path(sys.argv[1]); "
            "staging = publication.create_staging_directory(destination); "
            "publication.write_file_exclusive(staging / 'artifact.bin', b'child'); "
            "publication.publish_tree_no_replace(staging, destination)"
        )
        with (
            mock.patch.object(builder, "PYTHON", Path(sys.executable)),
            mock.patch.object(builder, "BASE", self.root),
        ):
            snapshot = builder._run_argv(
                [
                    sys.executable,
                    "-S",
                    "-P",
                    "-c",
                    code,
                    str(destination),
                ],
                label="real child publisher",
                expected_tree_publication=destination,
            )
        self.assertEqual(snapshot["files"]["artifact.bin"]["mode"], "0444")
        self.assertEqual(snapshot["directories"]["."]["mode"], "0555")

    def test_run_inputs_jointly_rejects_replacement_during_later_tree_check(
        self,
    ) -> None:
        role = builder.ROLE_PHASE1
        paths = {
            "dataset": self.root / "inputs/datasets/development_0",
            "cgb": self.root / "inputs/bundles/development_0_cgb",
            "uniform": self.root / "inputs/bundles/development_0_uniform",
        }
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        replacement = paths["dataset"].with_name("development_0_replacement")
        displaced = paths["dataset"].with_name("development_0_displaced")

        def publish_tree(path: Path) -> dict:
            staging = builder.publication.create_staging_directory(path)
            builder.publication.write_file_exclusive(
                staging / "artifact.bin",
                b"jointly retained generated bytes",
            )
            return builder.publication.publish_tree_no_replace(
                staging,
                path,
                file_mode=0o444,
                directory_mode=0o555,
            )

        publish_tree(replacement)
        replacement_identity = (replacement.stat().st_dev, replacement.stat().st_ino)
        original_identity: tuple[int, int] | None = None
        swapped = False

        def run_and_replace(
            argv,
            *,
            label: str,
            expected_tree_publication: Path,
        ) -> dict:
            nonlocal original_identity, swapped
            del argv
            snapshot = publish_tree(expected_tree_publication)
            if label == "uniform bundle argv":
                metadata = paths["dataset"].stat()
                original_identity = (metadata.st_dev, metadata.st_ino)
                paths["dataset"].rename(displaced)
                replacement.rename(paths["dataset"])
                swapped = True
            return snapshot

        argv = [str(builder.PYTHON), "-S", "-P", "publisher.py"]
        plan = {
            "input_table": [
                {
                    "role": role,
                    "index": 0,
                    "paths": {key: str(value) for key, value in paths.items()},
                    "generator_argv": argv,
                    "cgb_bundle_argv": argv,
                    "uniform_bundle_argv": argv,
                }
            ],
            "custody_stages": [
                {
                    "role": role,
                    "held_slurm_job_id": "740999",
                }
            ],
        }
        with (
            mock.patch.dict(os.environ, {"SLURM_JOB_ID": "740999"}, clear=True),
            mock.patch.object(builder, "_plan_for_root", return_value=plan),
            mock.patch.object(builder, "validate_plan", return_value=plan),
            mock.patch.object(builder, "_durably_verify_role_layout"),
            mock.patch.object(builder, "_run_argv", side_effect=run_and_replace),
            self.assertRaisesRegex(OSError, "retained ACW"),
        ):
            builder.run_inputs(self.root, role, 0)
        self.assertTrue(swapped)
        self.assertIsNotNone(original_identity)
        self.assertNotEqual(original_identity, replacement_identity)

    def test_tree_publication_rejects_topology_mutation_during_final_readback(
        self,
    ) -> None:
        destination = self.root / "mutated-tree"
        staging = builder.publication.create_staging_directory(destination)
        builder.publication.write_file_exclusive(
            staging / "nested" / "artifact.bin",
            b"tree bytes",
        )
        injected = False
        real_retain = builder.publication._retain_complete_tree

        def add_file_after_readback(retained, root: Path) -> dict:
            nonlocal injected
            snapshot = real_retain(retained, root)
            if Path(root) == destination and not injected:
                injected = True
                destination.chmod(0o755)
                (destination / "extra.bin").write_bytes(b"unregistered bytes")
            return snapshot

        with (
            mock.patch.object(
                builder.publication,
                "_retain_complete_tree",
                side_effect=add_file_after_readback,
            ),
            self.assertRaisesRegex(OSError, "retained ACW"),
        ):
            builder.publication.publish_tree_no_replace(staging, destination)
        self.assertTrue(injected)
        self.assertTrue((destination / "extra.bin").is_file())

    def test_tree_publication_rejects_late_byte_identical_root_substitution(
        self,
    ) -> None:
        destination = self.root / "published-tree-root-race"
        staging = builder.publication.create_staging_directory(destination)
        builder.publication.write_file_exclusive(
            staging / "nested" / "artifact.bin",
            b"byte-identical published tree",
        )
        replacement = self.root / "published-tree-root-replacement"
        (replacement / "nested").mkdir(parents=True, mode=0o700)
        replacement_artifact = replacement / "nested" / "artifact.bin"
        replacement_artifact.write_bytes(b"byte-identical published tree")
        replacement_artifact.chmod(0o600)
        replacement.chmod(0o700)
        displaced = self.root / "published-tree-root-displaced"
        original_identity: tuple[int, int] | None = None
        replacement_identity = (replacement.stat().st_dev, replacement.stat().st_ino)
        swapped = False
        real_retain = builder.publication._retain_complete_tree

        def replace_after_retained_snapshot(retained, root: Path) -> dict:
            nonlocal original_identity, swapped
            snapshot = real_retain(retained, root)
            if Path(root) == destination and not swapped:
                metadata = destination.stat()
                original_identity = (metadata.st_dev, metadata.st_ino)
                destination.rename(displaced)
                replacement.rename(destination)
                swapped = True
            return snapshot

        with (
            mock.patch.object(
                builder.publication,
                "_retain_complete_tree",
                side_effect=replace_after_retained_snapshot,
            ),
            self.assertRaisesRegex(OSError, "retained ACW"),
        ):
            builder.publication.publish_tree_no_replace(staging, destination)
        self.assertTrue(swapped)
        self.assertIsNotNone(original_identity)
        self.assertNotEqual(original_identity, replacement_identity)
        self.assertEqual(
            (destination / "nested" / "artifact.bin").read_bytes(),
            b"byte-identical published tree",
        )

    def test_freeze_tree_rejects_late_byte_identical_root_substitution(self) -> None:
        tree = self.root / "frozen-tree-root-race"
        (tree / "nested").mkdir(parents=True)
        artifact = tree / "nested" / "artifact.bin"
        artifact.write_bytes(b"byte-identical frozen tree")
        replacement = self.root / "frozen-tree-root-replacement"
        (replacement / "nested").mkdir(parents=True)
        replacement_artifact = replacement / "nested" / "artifact.bin"
        replacement_artifact.write_bytes(b"byte-identical frozen tree")
        builder.publication.freeze_tree(replacement)
        displaced = self.root / "frozen-tree-root-displaced"
        original_identity: tuple[int, int] | None = None
        replacement_identity = (replacement.stat().st_dev, replacement.stat().st_ino)
        swapped = False
        real_retain = builder.publication._retain_complete_tree

        def replace_after_retained_snapshot(retained, root: Path) -> dict:
            nonlocal original_identity, swapped
            snapshot = real_retain(retained, root)
            if Path(root) == tree and not swapped:
                metadata = tree.stat()
                original_identity = (metadata.st_dev, metadata.st_ino)
                tree.rename(displaced)
                replacement.rename(tree)
                swapped = True
            return snapshot

        with (
            mock.patch.object(
                builder.publication,
                "_retain_complete_tree",
                side_effect=replace_after_retained_snapshot,
            ),
            self.assertRaisesRegex(OSError, "retained ACW"),
        ):
            builder.publication.freeze_tree(tree)
        self.assertTrue(swapped)
        self.assertIsNotNone(original_identity)
        self.assertNotEqual(original_identity, replacement_identity)
        self.assertEqual(stat.S_IMODE(tree.stat().st_mode), 0o555)
        self.assertEqual(
            stat.S_IMODE((tree / "nested" / "artifact.bin").stat().st_mode),
            0o444,
        )

    def test_tree_snapshot_rejects_same_name_inode_swap_after_file_readback(
        self,
    ) -> None:
        tree = self.root / "inode-mutated-tree"
        tree.mkdir()
        artifact = tree / "artifact.bin"
        raw = b"byte-identical tree replacement"
        artifact.write_bytes(raw)
        original_identity = (artifact.stat().st_dev, artifact.stat().st_ino)
        replaced = False
        real_record = builder.publication._descriptor_record

        def replace_after_readback(path: Path, *, fsync_file: bool) -> dict:
            nonlocal replaced
            record = real_record(path, fsync_file=fsync_file)
            if Path(path) == artifact and not replaced:
                replaced = True
                artifact.unlink()
                artifact.write_bytes(raw)
            return record

        with (
            mock.patch.object(
                builder.publication,
                "_descriptor_record",
                side_effect=replace_after_readback,
            ),
            self.assertRaisesRegex(OSError, "directory metadata changed"),
        ):
            builder.publication._tree_snapshot(tree, fsync_files=False)
        self.assertTrue(replaced)
        self.assertNotEqual(
            original_identity,
            (artifact.stat().st_dev, artifact.stat().st_ino),
        )
        self.assertEqual(artifact.read_bytes(), raw)

    def test_tree_snapshot_rejects_inode_swap_after_final_file_identity_pass(
        self,
    ) -> None:
        tree = self.root / "late-inode-mutated-tree"
        tree.mkdir()
        artifact = tree / "artifact.bin"
        raw = b"late byte-identical tree replacement"
        artifact.write_bytes(raw)
        original_identity = (artifact.stat().st_dev, artifact.stat().st_ino)
        file_identity_checks = 0
        real_require = builder.publication._require_name_bound_to_descriptor

        def replace_after_final_file_identity(
            path: Path,
            descriptor: int,
            record: dict,
            *,
            directory: bool,
        ) -> None:
            nonlocal file_identity_checks
            real_require(
                path,
                descriptor,
                record,
                directory=directory,
            )
            if not directory:
                file_identity_checks += 1
                if file_identity_checks == 2:
                    artifact.unlink()
                    artifact.write_bytes(raw)

        with (
            mock.patch.object(
                builder.publication,
                "_require_name_bound_to_descriptor",
                side_effect=replace_after_final_file_identity,
            ),
            self.assertRaisesRegex(OSError, "after file identity readback"),
        ):
            builder.publication._tree_snapshot(tree, fsync_files=False)
        self.assertEqual(file_identity_checks, 2)
        self.assertNotEqual(
            original_identity,
            (artifact.stat().st_dev, artifact.stat().st_ino),
        )
        self.assertEqual(artifact.read_bytes(), raw)

    def test_tree_snapshot_rejects_preopened_write_after_final_identity_pass(
        self,
    ) -> None:
        tree = self.root / "late-content-mutated-tree"
        tree.mkdir()
        artifact = tree / "artifact.bin"
        artifact.write_bytes(b"original")
        writable = os.open(artifact, os.O_RDWR)
        file_identity_checks = 0
        real_require = builder.publication._require_name_bound_to_descriptor

        def write_after_final_file_identity(
            path: Path,
            descriptor: int,
            record: dict,
            *,
            directory: bool,
        ) -> None:
            nonlocal file_identity_checks
            real_require(
                path,
                descriptor,
                record,
                directory=directory,
            )
            if not directory:
                file_identity_checks += 1
                if file_identity_checks == 3:
                    os.pwrite(writable, b"MUTATED!", 0)

        try:
            with (
                mock.patch.object(
                    builder.publication,
                    "_require_name_bound_to_descriptor",
                    side_effect=write_after_final_file_identity,
                ),
                self.assertRaisesRegex(
                    OSError,
                    "file contents changed|descriptor metadata changed",
                ),
            ):
                builder.publication.freeze_tree(tree)
        finally:
            os.close(writable)
        self.assertEqual(file_identity_checks, 3)
        self.assertEqual(artifact.read_bytes(), b"MUTATED!")
        self.assertEqual(stat.S_IMODE(artifact.stat().st_mode), 0o444)

    def test_tree_snapshot_rejects_write_to_earlier_file_during_terminal_pass(
        self,
    ) -> None:
        tree = self.root / "multi-file-terminal-race"
        tree.mkdir()
        earlier = tree / "a.bin"
        later = tree / "b.bin"
        earlier.write_bytes(b"ORIGINAL")
        later.write_bytes(b"later")
        writable = os.open(earlier, os.O_RDWR)
        later_reads = 0
        real_record = builder.publication._descriptor_record_from_open_file

        def mutate_earlier_before_later_terminal_read(
            descriptor: int,
            path: Path,
            *,
            fsync_file: bool,
        ) -> dict:
            nonlocal later_reads
            if Path(path) == later:
                later_reads += 1
                if later_reads == 2:
                    os.pwrite(writable, b"MUTATED!", 0)
            return real_record(
                descriptor,
                path,
                fsync_file=fsync_file,
            )

        try:
            with (
                mock.patch.object(
                    builder.publication,
                    "_descriptor_record_from_open_file",
                    side_effect=mutate_earlier_before_later_terminal_read,
                ),
                self.assertRaisesRegex(OSError, "descriptor metadata changed"),
            ):
                builder.publication.snapshot_tree(tree)
        finally:
            os.close(writable)
        self.assertEqual(later_reads, 2)
        self.assertEqual(earlier.read_bytes(), b"MUTATED!")

    def test_tree_snapshot_rejects_entry_added_after_terminal_directory_pass(
        self,
    ) -> None:
        tree = self.root / "late-terminal-topology-race"
        tree.mkdir()
        artifact = tree / "artifact.bin"
        artifact.write_bytes(b"registered")
        artifact_reads = 0
        real_record = builder.publication._descriptor_record_from_open_file

        def add_entry_during_terminal_file_read(
            descriptor: int,
            path: Path,
            *,
            fsync_file: bool,
        ) -> dict:
            nonlocal artifact_reads
            record = real_record(
                descriptor,
                path,
                fsync_file=fsync_file,
            )
            if Path(path) == artifact:
                artifact_reads += 1
                if artifact_reads == 2:
                    (tree / "late-extra.bin").write_bytes(b"extra")
            return record

        with (
            mock.patch.object(
                builder.publication,
                "_descriptor_record_from_open_file",
                side_effect=add_entry_during_terminal_file_read,
            ),
            self.assertRaisesRegex(OSError, "retained ACW directory changed"),
        ):
            builder.publication.snapshot_tree(tree)
        self.assertEqual(artifact_reads, 2)
        self.assertTrue((tree / "late-extra.bin").is_file())

    def test_tree_publication_propagates_fsync_failure(self) -> None:
        destination = self.root / "failed-tree"
        staging = builder.publication.create_staging_directory(destination)
        builder.publication.write_file_exclusive(
            staging / "artifact.bin",
            b"tree bytes",
        )
        failure = OSError(errno.EIO, "injected tree metadata fsync failure")
        with (
            mock.patch.object(
                builder.publication.os,
                "fsync",
                side_effect=failure,
            ),
            self.assertRaises(OSError) as raised,
        ):
            builder.publication.publish_tree_no_replace(
                staging,
                destination,
            )
        self.assertEqual(raised.exception.errno, failure.errno)
        self.assertFalse(os.path.lexists(destination))
        self.assertTrue(staging.is_dir())
        self.assertEqual(stat.S_IMODE(staging.stat().st_mode), 0o700)

    def test_freeze_tree_fsyncs_and_verifies_final_modes(self) -> None:
        tree = self.root / "custody-freeze"
        nested = tree / "nested"
        nested.mkdir(parents=True)
        artifact = nested / "artifact.bin"
        artifact.write_bytes(b"custody bytes")
        with (
            mock.patch.object(
                builder.publication.os,
                "fsync",
                wraps=os.fsync,
            ) as fsync,
            mock.patch.object(
                builder.publication,
                "fsync_directory",
                wraps=builder.publication.fsync_directory,
            ) as fsync_directory,
        ):
            builder._freeze_tree(tree)
        self.assertGreaterEqual(fsync.call_count, 3)
        self.assertIn(mock.call(tree.parent), fsync_directory.call_args_list)
        self.assertEqual(stat.S_IMODE(artifact.stat().st_mode), 0o444)
        self.assertEqual(stat.S_IMODE(nested.stat().st_mode), 0o555)
        self.assertEqual(stat.S_IMODE(tree.stat().st_mode), 0o555)
        snapshot = builder.publication.verify_frozen_tree(tree)
        self.assertEqual(snapshot["files"]["nested/artifact.bin"]["mode"], "0444")
        self.assertEqual(snapshot["directories"]["nested"]["mode"], "0555")

    def test_freeze_file_fsyncs_parent_and_rejects_inode_substitution(self) -> None:
        artifact = self.root / "single-evidence.bin"
        raw = b"single evidence bytes"
        artifact.write_bytes(raw)
        with mock.patch.object(
            builder.publication,
            "fsync_directory",
            wraps=builder.publication.fsync_directory,
        ) as fsync_directory:
            record = builder.publication.freeze_file(artifact)
        self.assertEqual(record["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(stat.S_IMODE(artifact.stat().st_mode), 0o444)
        fsync_directory.assert_called_once_with(artifact.parent)

        replacement = self.root / "replacement-evidence.bin"
        replacement.write_bytes(raw)
        replacement.chmod(0o444)
        real_fsync_directory = builder.publication.fsync_directory

        def substitute(parent: Path) -> None:
            artifact.unlink()
            replacement.rename(artifact)
            real_fsync_directory(parent)

        with (
            mock.patch.object(
                builder.publication,
                "fsync_directory",
                side_effect=substitute,
            ),
            self.assertRaisesRegex(OSError, "changed after parent fsync"),
        ):
            builder.publication.freeze_file(artifact)

    def test_freeze_file_propagates_parent_fsync_failure(self) -> None:
        artifact = self.root / "single-evidence-fsync-failure.bin"
        artifact.write_bytes(b"single evidence bytes")
        failure = OSError(errno.EIO, "injected parent fsync failure")
        with (
            mock.patch.object(
                builder.publication,
                "fsync_directory",
                side_effect=failure,
            ),
            self.assertRaises(OSError) as raised,
        ):
            builder.publication.freeze_file(artifact)
        self.assertEqual(raised.exception.errno, errno.EIO)

    def test_freeze_tree_propagates_mode_and_fsync_failures(self) -> None:
        for index, (operation, failure) in enumerate(
            (
                (
                    "fchmod",
                    OSError(errno.EPERM, "injected custody mode failure"),
                ),
                (
                    "fsync",
                    OSError(errno.EIO, "injected custody fsync failure"),
                ),
            )
        ):
            with self.subTest(operation=operation):
                tree = self.root / f"custody-freeze-failure-{index}"
                tree.mkdir()
                (tree / "artifact.bin").write_bytes(b"custody bytes")
                with (
                    mock.patch.object(
                        builder.publication.os,
                        operation,
                        side_effect=failure,
                    ),
                    self.assertRaises(OSError) as raised,
                ):
                    builder._freeze_tree(tree)
                self.assertEqual(raised.exception.errno, failure.errno)

    def test_freeze_tree_rejects_mode_change_after_descriptor_snapshot(self) -> None:
        tree = self.root / "custody-freeze-mode-race"
        tree.mkdir()
        artifact = tree / "artifact.bin"
        artifact.write_bytes(b"custody bytes")
        changed = False
        real_record = builder.publication._descriptor_record_from_open_file

        def change_mode_after_readback(
            descriptor: int,
            path: Path,
            *,
            fsync_file: bool,
        ) -> dict:
            nonlocal changed
            record = real_record(
                descriptor,
                path,
                fsync_file=fsync_file,
            )
            if Path(path) == artifact and not changed:
                changed = True
                os.fchmod(descriptor, 0o600)
            return record

        with (
            mock.patch.object(
                builder.publication,
                "_descriptor_record_from_open_file",
                side_effect=change_mode_after_readback,
            ),
            self.assertRaisesRegex(OSError, "descriptor metadata changed"),
        ):
            builder._freeze_tree(tree)
        self.assertTrue(changed)
        self.assertEqual(stat.S_IMODE(artifact.stat().st_mode), 0o600)

    def test_attempt_start_binds_the_precommitted_held_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary) / "home"
            repository = home / "shohin_acw"
            root = repository / "artifacts" / "r12" / "acw_development_g2"
            (repository / "pipeline").mkdir(parents=True)
            (root / "runs").mkdir(parents=True)
            plan = json.loads(
                (
                    Path(__file__).resolve().parents[1] / DEVELOPMENT_PLAN_PATH
                ).read_bytes()
            )
            job_ids = ("740999", "741000", "741001", "741002")
            prior = None
            for stage, job_id in zip(plan["custody_stages"], job_ids, strict=True):
                stage["held_slurm_job_id"] = job_id
                stage["dependency"] = (
                    None
                    if prior is None
                    else {"type": "afterok", "held_slurm_job_id": prior}
                )
                prior = job_id
            plan["attempt_registry"]["held_slurm_job_id"] = job_ids[0]
            plan["accounting"]["monitor_stage"]["held_slurm_job_id"] = "741003"
            plan["accounting"]["monitor_stage"]["dependency"] = {
                "type": "afterok",
                "held_slurm_job_id": job_ids[-1],
            }
            plan["accounting"]["monitor_stage"]["script"]["sha256"] = file_sha256(
                builder.REPOSITORY / builder.MONITOR_SCRIPT
            )
            plan["ready_for_g_commit"] = True
            plan["input_table"] = builder.expected_input_table(plan)
            plan["attempt_table"] = builder.expected_attempt_table(plan)
            plan = builder._hash_bound(plan)
            plan_path = root / "development_plan.json"
            plan_path.write_bytes(builder.canonical_json_bytes(plan) + b"\n")
            plan_path.chmod(0o444)
            plan_sha256 = file_sha256(plan_path)
            environment = {
                "SLURM_JOB_ID": "740999",
                "SLURM_JOB_NAME": builder.ROLE_JOB_NAMES[builder.ROLE_PHASE1],
                "SLURM_JOB_NODELIST": "ec51",
                "SLURM_CPUS_PER_TASK": "4",
            }
            with (
                mock.patch.object(
                    builder,
                    "__file__",
                    str(repository / "pipeline" / "build_acw_development_manifest.py"),
                ),
                mock.patch.object(builder, "DEVELOPMENT_PLAN_RAW_SHA256", plan_sha256),
                mock.patch.object(builder, "validate_plan", return_value=plan),
                mock.patch.object(
                    builder, "_validated_g_commit", return_value="d" * 40
                ),
                mock.patch.dict(os.environ, environment, clear=True),
            ):
                attempt = builder.build_attempt_start(root.resolve())
        self.assertEqual(attempt["scientific_commit"], "d" * 40)
        self.assertEqual(attempt["slurm"]["job_id"], "740999")
        self.assertTrue(attempt["created_before_scoring"])
        self.assertEqual(attempt["checkpoint_count_at_creation"], 0)

    def test_plan_drift_and_mutable_artifacts_fail_closed(self) -> None:
        plan = self.root / "development_plan.json"
        plan.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "immutable file"):
            builder._plan_reference(self.root)
        plan.chmod(0o600)
        plan.write_bytes(b"{}\n")
        plan.chmod(0o444)
        with self.assertRaisesRegex(ValueError, "committed bytes"):
            builder._plan_reference(self.root)

        artifact = self.root / "mutable.json"
        artifact.write_bytes(b"{}\n")
        artifact.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "immutable file"):
            builder._reference(artifact, self.root)

    def test_private_closed_world_records_are_relative_to_private_root(self) -> None:
        private = Path(self.temporary.name) / "private"
        private.mkdir()
        artifact = private / "artifact.bin"
        artifact.write_bytes(b"private")
        artifact.chmod(0o444)
        with mock.patch.object(
            builder, "_expected_scan_files", return_value={artifact}
        ):
            summary = builder.closed_world_scan(
                self.root,
                "final",
                scan_root=private,
            )
        self.assertEqual(summary["files"][0]["path"], "artifact.bin")
        self.assertEqual(summary["file_count"], 1)

    def test_closed_world_receipt_rejects_byte_identical_inode_substitution(
        self,
    ) -> None:
        scan_root = Path(self.temporary.name) / "inode-bound-closed-world"
        scan_root.mkdir()
        artifact = scan_root / "artifact.bin"
        replacement = Path(self.temporary.name) / "inode-bound-replacement.bin"
        raw = b"byte-identical closed-world replacement"
        for path in (artifact, replacement):
            path.write_bytes(raw)
            path.chmod(0o444)
        with mock.patch.object(
            builder,
            "_expected_scan_files",
            return_value={artifact},
        ):
            original = builder.closed_world_scan(
                self.root,
                "final",
                scan_root=scan_root,
                bind_inode_identity=True,
            )
            artifact.unlink()
            replacement.rename(artifact)
            substituted = builder.closed_world_scan(
                self.root,
                "final",
                scan_root=scan_root,
                bind_inode_identity=True,
            )
        self.assertEqual(
            original["files"][0]["sha256"], substituted["files"][0]["sha256"]
        )
        self.assertNotEqual(
            original["files"][0]["st_ino"], substituted["files"][0]["st_ino"]
        )
        self.assertNotEqual(original["tree_sha256"], substituted["tree_sha256"])

    def test_closed_world_receipt_binds_directory_inode_substitution(self) -> None:
        scan_root = Path(self.temporary.name) / "directory-bound-closed-world"
        nested = scan_root / "nested"
        nested.mkdir(parents=True)
        artifact = nested / "artifact.bin"
        artifact.write_bytes(b"same file inode under replacement directory")
        artifact.chmod(0o444)
        replacement = Path(self.temporary.name) / "replacement-nested"
        replacement.mkdir()
        displaced = Path(self.temporary.name) / "displaced-nested"
        with mock.patch.object(
            builder,
            "_expected_scan_files",
            return_value={artifact},
        ):
            original = builder.closed_world_scan(
                self.root,
                "final",
                scan_root=scan_root,
                bind_inode_identity=True,
            )
            nested.rename(displaced)
            replacement.rename(nested)
            (displaced / artifact.name).rename(artifact)
            substituted = builder.closed_world_scan(
                self.root,
                "final",
                scan_root=scan_root,
                bind_inode_identity=True,
            )
        self.assertEqual(original["files"], substituted["files"])
        original_directories = {
            record["path"]: record for record in original["directories"]
        }
        substituted_directories = {
            record["path"]: record for record in substituted["directories"]
        }
        self.assertEqual(
            original_directories["."]["st_ino"],
            substituted_directories["."]["st_ino"],
        )
        self.assertNotEqual(
            original_directories["nested"]["st_ino"],
            substituted_directories["nested"]["st_ino"],
        )
        self.assertNotEqual(original["tree_sha256"], substituted["tree_sha256"])

    def test_consumer_closed_world_scan_rejects_unlisted_live_file(self) -> None:
        scan_root = Path(self.temporary.name) / "handoff"
        scan_root.mkdir()
        expected = set()
        for name in ("a.bin", "b.bin", "omitted-hidden.bin"):
            path = scan_root / name
            path.write_bytes(name.encode("ascii"))
            path.chmod(0o444)
            if name != "omitted-hidden.bin":
                expected.add(path)
        with (
            mock.patch.object(builder, "_expected_scan_files", return_value=expected),
            self.assertRaisesRegex(ValueError, "extra=.*omitted-hidden"),
        ):
            builder.closed_world_scan(
                self.root,
                "phase1",
                scan_root=scan_root,
            )

    def test_closed_world_scan_rejects_file_added_during_descriptor_snapshot(
        self,
    ) -> None:
        scan_root = Path(self.temporary.name) / "mutating-handoff"
        scan_root.mkdir()
        artifact = scan_root / "artifact.bin"
        artifact.write_bytes(b"registered")
        artifact.chmod(0o444)
        injected = False
        real_retain = builder.publication._retain_complete_tree

        def add_file_after_readback(retained, root: Path) -> dict:
            nonlocal injected
            snapshot = real_retain(retained, root)
            if Path(root) == scan_root and not injected:
                injected = True
                (scan_root / "late-extra.bin").write_bytes(b"extra")
            return snapshot

        with (
            mock.patch.object(
                builder,
                "_expected_scan_files",
                return_value={artifact},
            ),
            mock.patch.object(
                builder.publication,
                "_retain_complete_tree",
                side_effect=add_file_after_readback,
            ),
            self.assertRaisesRegex(OSError, "retained ACW"),
        ):
            builder.closed_world_scan(
                self.root,
                "phase1",
                scan_root=scan_root,
            )
        self.assertTrue(injected)

    def test_terminal_recovery_allowlist_includes_only_existing_outputs(self) -> None:
        checkpoint = self.root / "best_development_checkpoint.pt"
        checkpoint.write_bytes(b"preserved checkpoint")
        checkpoint.chmod(0o444)
        plan = {"input_table": [], "attempt_table": []}
        with mock.patch.object(builder, "_plan_for_root", return_value=plan):
            expected = builder._expected_scan_files(
                self.root,
                "final",
                self.root,
                pending_completion=None,
                include_current_accounting=True,
                include_terminal_outputs=True,
            )
        self.assertIn(checkpoint, expected)
        self.assertNotIn(self.root / "development_baseline.json", expected)

    def test_baseline_publication_runs_only_after_terminal_validation(self) -> None:
        events: list[str] = []

        def validated(
            _root: Path,
            *,
            include_terminal_outputs: bool,
        ) -> dict:
            events.append("validated")
            self.assertTrue(include_terminal_outputs)
            return {}

        def prepared(_root: Path) -> dict:
            events.append("frozen")
            return {
                "baseline_sha256": "b" * 64,
                "baseline_payload_sha256": "a" * 64,
                "checkpoint_sha256": "c" * 64,
            }

        with (
            mock.patch.object(
                builder, "validate_terminal_prerequisites", side_effect=validated
            ),
            mock.patch.object(
                builder,
                "_prepare_development_baseline_outputs",
                side_effect=prepared,
            ),
        ):
            result = builder.publish_development_baseline_after_terminal_validation(
                self.root
            )
        self.assertEqual(events, ["validated", "frozen"])
        self.assertEqual(result["baseline_sha256"], "b" * 64)
        self.assertEqual(result["checkpoint_sha256"], "c" * 64)

    def test_baseline_preparation_recovers_from_checkpoint_only(self) -> None:
        checkpoint = self.root / "best_development_checkpoint.pt"
        checkpoint.write_bytes(b"candidate weights")
        checkpoint.chmod(0o444)
        checkpoint_sha256 = file_sha256(checkpoint)
        baseline_path = self.root / "development_baseline.json"
        for destination in (checkpoint, baseline_path):
            temporary = builder._protocol_publish_temp(destination)
            temporary.write_bytes(b"interrupted")
            temporary.chmod(0o444)
        written_payload: dict = {}

        def frozen(_manifest: Path, staged: Path) -> dict:
            staged.write_bytes(b"candidate weights")
            staged.chmod(0o444)
            return builder._hash_bound(
                {
                    "copied_checkpoint": {
                        "path": str(staged.resolve(strict=True)),
                        "sha256": checkpoint_sha256,
                        "bytes": staged.stat().st_size,
                        "mode": "0444",
                    }
                }
            )

        def write_baseline(path: Path, payload: dict) -> str:
            written_payload.update(payload)
            raw = builder.canonical_json_bytes(payload) + b"\n"
            path.write_bytes(raw)
            path.chmod(0o444)
            return hashlib.sha256(raw).hexdigest()

        def validated(path: Path) -> dict:
            return {
                "record": {
                    "sha256": file_sha256(path),
                    "payload_sha256": written_payload["payload_sha256"],
                },
                "copied_checkpoint": written_payload["copied_checkpoint"],
            }

        with (
            mock.patch.object(
                builder, "freeze_development_baseline", side_effect=frozen
            ),
            mock.patch.object(
                builder,
                "write_immutable_development_baseline",
                side_effect=write_baseline,
            ),
            mock.patch.object(
                builder,
                "_validate_frozen_development_baseline",
                side_effect=validated,
            ),
        ):
            first = builder._prepare_development_baseline_outputs(self.root)
            second = builder._prepare_development_baseline_outputs(self.root)
        self.assertEqual(checkpoint.read_bytes(), b"candidate weights")
        self.assertEqual(
            written_payload["copied_checkpoint"]["path"],
            str(checkpoint.resolve(strict=True)),
        )
        self.assertEqual(first, second)
        self.assertFalse(os.path.lexists(builder._protocol_publish_temp(checkpoint)))
        self.assertFalse(os.path.lexists(builder._protocol_publish_temp(baseline_path)))

    def test_terminal_envelope_narrows_and_embargoes_claim(self) -> None:
        plan = {"execution_parent_commit": "d" * 40}
        reference = {"path": "artifact", "sha256": "a" * 64}
        semantic = {
            "record": reference,
            "development_manifest": reference,
            "source_checkpoint": reference,
            "copied_checkpoint": reference,
            "selection": {},
        }
        with (
            mock.patch.object(builder, "_plan_for_root", return_value=plan),
            mock.patch.object(builder, "validate_plan", return_value=plan),
            mock.patch.object(
                builder,
                "_require_canonical_monitor_runtime",
                return_value=self._monitor_binding(),
            ),
            mock.patch.object(builder, "_validate_all_stage_receipts_and_accounting"),
            mock.patch.object(
                builder,
                "_validate_frozen_development_baseline",
                return_value=semantic,
            ),
            mock.patch.object(builder, "_reference", return_value=reference),
            mock.patch.object(builder, "_plan_reference", return_value=reference),
            mock.patch.object(builder, "_freeze_tree"),
            mock.patch.object(builder, "closed_world_scan", return_value={}),
            mock.patch.object(builder, "_validated_g_commit", return_value="e" * 40),
        ):
            envelope = builder.build_terminal_accounting(self.root)
        self.assertTrue(envelope["all_four_jobs_terminal_and_step_free"])
        self.assertTrue(envelope["exact_registered_root_verified"])
        self.assertFalse(envelope["same_uid_external_compute_excluded"])
        self.assertFalse(envelope["ordinary_batch_children_independently_attested"])
        self.assertTrue(
            envelope["external_sha256_anchor_required_before_performance_claim"]
        )
        self.assertFalse(envelope["performance_claim_ready"])

    def test_terminal_accounting_polls_until_exact_rows_stabilize(self) -> None:
        stage = {
            "held_slurm_job_id": "740999",
            "job_name": "stage",
            "expected_node": "ec51",
        }
        plan = {"accounting": {"poll_timeout_seconds": 180}}
        first = [{"job_id_raw": "740999", "max_rss": ""}]
        expected = [{"job_id_raw": "740999", "max_rss": "1G"}]
        completed = mock.Mock(stdout="rows")
        with (
            mock.patch.object(builder.subprocess, "run", return_value=completed) as run,
            mock.patch.object(
                builder,
                "_validated_terminal_rows_for_stage",
                side_effect=(first, expected, expected),
            ),
            mock.patch.object(builder.time, "sleep") as sleep,
        ):
            observed = builder._query_terminal_rows_for_stage(plan, stage)
        self.assertEqual(observed, expected)
        self.assertEqual(run.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_terminal_accounting_bounds_each_stalled_sacct_call(self) -> None:
        stage = {
            "held_slurm_job_id": "740999",
            "job_name": "stage",
            "expected_node": "ec51",
        }
        plan = {"accounting": {"poll_timeout_seconds": 5}}
        stalled = builder.subprocess.TimeoutExpired(cmd="sacct", timeout=5.0)
        with (
            mock.patch.object(
                builder.time,
                "monotonic",
                side_effect=(100.0, 100.0, 105.0),
            ),
            mock.patch.object(
                builder.subprocess,
                "run",
                side_effect=stalled,
            ) as run,
            mock.patch.object(builder.time, "sleep") as sleep,
            self.assertRaisesRegex(ValueError, "did not stabilize"),
        ):
            builder._query_terminal_rows_for_stage(plan, stage)
        self.assertEqual(run.call_args.kwargs["timeout"], 5.0)
        sleep.assert_not_called()

    def test_terminal_receipt_validator_recomputes_and_cannot_self_authorize(
        self,
    ) -> None:
        receipt = Path(self.temporary.name) / "terminal.json"
        fresh = builder._hash_bound(
            {
                "schema": "r12_acw_development_terminal_accounting_v1",
                "monitor": self._monitor_binding(),
                "performance_claim_ready": False,
            }
        )
        receipt.write_bytes(builder.canonical_json_bytes(fresh) + b"\n")
        receipt.chmod(0o444)
        canonical_receipt = receipt.resolve(strict=True)
        with (
            mock.patch.object(builder, "TERMINAL_ACCOUNTING_PATH", canonical_receipt),
            mock.patch.object(builder, "build_terminal_accounting", return_value=fresh),
        ):
            validated = builder.validate_terminal_accounting(
                self.root, canonical_receipt
            )
        self.assertFalse(validated["performance_claim_ready"])

        forged = dict(fresh)
        forged["performance_claim_ready"] = True
        forged = builder._hash_bound(forged)
        receipt.chmod(0o600)
        receipt.write_bytes(builder.canonical_json_bytes(forged) + b"\n")
        receipt.chmod(0o444)
        with (
            mock.patch.object(builder, "TERMINAL_ACCOUNTING_PATH", canonical_receipt),
            mock.patch.object(builder, "build_terminal_accounting", return_value=fresh),
        ):
            with self.assertRaisesRegex(ValueError, "differs|self-authorize"):
                builder.validate_terminal_accounting(self.root, canonical_receipt)

    def test_terminal_verification_retains_recomputed_receipt_inode(self) -> None:
        receipt_path = Path(self.temporary.name) / "retained-terminal.json"
        fresh = builder._hash_bound(
            {
                "schema": "r12_acw_development_terminal_accounting_v1",
                "monitor": self._monitor_binding(),
                "performance_claim_ready": False,
            }
        )
        raw = builder.canonical_json_bytes(fresh) + b"\n"
        receipt_path.write_bytes(raw)
        receipt_path.chmod(0o444)
        receipt_path = receipt_path.resolve(strict=True)
        original_identity = receipt_path.stat().st_dev, receipt_path.stat().st_ino
        replacement_identity: tuple[int, int] | None = None

        def substitute_receipt(
            root: Path,
            receipt: dict,
            receipt_record: dict,
            *,
            retained=None,
        ) -> dict:
            self.assertIsNotNone(retained)
            nonlocal replacement_identity
            self.assertEqual(root, self.root)
            self.assertEqual(receipt, fresh)
            self.assertEqual(
                (receipt_record["st_dev"], receipt_record["st_ino"]),
                original_identity,
            )
            receipt_path.unlink()
            receipt_path.write_bytes(raw)
            receipt_path.chmod(0o444)
            replacement_identity = (
                receipt_path.stat().st_dev,
                receipt_path.stat().st_ino,
            )
            return {"sha256": "a" * 64}

        with (
            mock.patch.object(builder, "TERMINAL_ACCOUNTING_PATH", receipt_path),
            mock.patch.object(builder, "build_terminal_accounting", return_value=fresh),
            mock.patch.object(
                builder,
                "publish_terminal_verification_attestation",
                side_effect=substitute_receipt,
            ),
            self.assertRaisesRegex(OSError, "retained ACW artifact changed"),
        ):
            builder.verify_and_publish_terminal_verification_attestation(self.root)
        self.assertIsNotNone(replacement_identity)
        self.assertNotEqual(original_identity, replacement_identity)

    def test_terminal_verification_is_precompletion_immutable_and_retry_safe(
        self,
    ) -> None:
        base = Path(self.temporary.name) / "verification-base"
        log_root = base / "logs"
        log_root.mkdir(parents=True)
        receipt_path = Path(self.temporary.name) / "verification-receipt.json"
        receipt_path.write_bytes(b"verified receipt\n")
        receipt_path.chmod(0o444)
        log_path = log_root / "acw_development_monitor_740999.out"
        log_path.write_bytes(b"monitor output in progress\n")
        verification_root = Path(self.temporary.name) / "verification-evidence"
        reference = {"path": "development_plan.json", "sha256": "a" * 64}
        monitor = self._monitor_binding()
        receipt = {
            "scientific_commit": "d" * 40,
            "monitor": monitor,
            "payload_sha256": "e" * 64,
        }
        plan = {
            "accounting": {
                "monitor_stage": {"held_slurm_job_id": "740999"},
            }
        }
        real_fstat = os.fstat
        receipt_record = builder.publication.snapshot_file(receipt_path)

        def stdout_fstat(descriptor: int) -> os.stat_result:
            if descriptor == 1:
                return log_path.stat()
            return real_fstat(descriptor)

        with (
            mock.patch.dict(os.environ, {"SLURM_JOB_ID": "740999"}, clear=True),
            mock.patch.object(builder, "BASE", base),
            mock.patch.object(builder, "TERMINAL_ACCOUNTING_PATH", receipt_path),
            mock.patch.object(
                builder,
                "TERMINAL_VERIFICATION_ROOT",
                verification_root,
            ),
            mock.patch.object(builder, "_plan_for_root", return_value=plan),
            mock.patch.object(builder, "validate_plan", return_value=plan),
            mock.patch.object(builder, "_plan_reference", return_value=reference),
            mock.patch.object(
                builder,
                "_validate_recorded_monitor_binding",
                return_value=monitor,
            ),
            mock.patch.object(builder.os, "fstat", side_effect=stdout_fstat),
        ):
            first = builder.publish_terminal_verification_attestation(
                self.root,
                receipt,
                receipt_record,
            )
            second = builder.publish_terminal_verification_attestation(
                self.root,
                receipt,
                receipt_record,
            )
            verification_path = verification_root / builder.TERMINAL_VERIFICATION_NAME
            racing_root = verification_root.with_name("racing-verification-evidence")
            replacement = verification_root.with_name("racing-verification-replacement")
            replacement_staging = builder.publication.create_staging_directory(
                replacement
            )
            builder.publication.write_file_exclusive(
                replacement_staging / builder.TERMINAL_VERIFICATION_NAME,
                verification_path.read_bytes(),
            )
            builder.publication.publish_tree_no_replace(
                replacement_staging,
                replacement,
                file_mode=0o444,
                directory_mode=0o555,
            )
            displaced = verification_root.with_name("racing-verification-displaced")
            real_publish = builder.publication.publish_tree_no_replace
            published_identity: tuple[int, int] | None = None

            def publish_then_replace(staging: Path, destination: Path, **kwargs):
                nonlocal published_identity
                snapshot = real_publish(staging, destination, **kwargs)
                metadata = destination.stat()
                published_identity = (metadata.st_dev, metadata.st_ino)
                destination.rename(displaced)
                replacement.rename(destination)
                return snapshot

            with (
                mock.patch.object(
                    builder,
                    "TERMINAL_VERIFICATION_ROOT",
                    racing_root,
                ),
                mock.patch.object(
                    builder.publication,
                    "publish_tree_no_replace",
                    side_effect=publish_then_replace,
                ),
                self.assertRaisesRegex(
                    OSError,
                    "tree changed after immutable publication",
                ),
            ):
                builder.publish_terminal_verification_attestation(
                    self.root,
                    receipt,
                    receipt_record,
                )
            self.assertIsNotNone(published_identity)
            self.assertNotEqual(
                published_identity,
                (racing_root.stat().st_dev, racing_root.stat().st_ino),
            )
        self.assertEqual(first, second)
        self.assertEqual(stat.S_IMODE(verification_root.stat().st_mode), 0o555)
        self.assertEqual(stat.S_IMODE(verification_path.stat().st_mode), 0o444)
        recorded = json.loads(verification_path.read_bytes())
        self.assertTrue(recorded["established_before_monitor_completion"])
        self.assertEqual(
            recorded["monitor_log_descriptor"]["st_ino"],
            log_path.stat().st_ino,
        )

    def test_monitor_anchor_requires_completed_j5_and_binds_receipt_log(self) -> None:
        base = Path(self.temporary.name) / "base"
        log_root = base / "logs"
        log_root.mkdir(parents=True)
        receipt_path = Path(self.temporary.name) / "terminal.json"
        reference = {"path": "artifact", "sha256": "a" * 64}
        receipt = builder._hash_bound(
            {
                "schema": "r12_acw_development_terminal_accounting_v1",
                "protocol": "R12-ACW-DEVELOPMENT-TERMINAL-ACCOUNTING-v1",
                "development_plan": reference,
                "scientific_commit": "d" * 40,
                "monitor": self._monitor_binding(),
                "stages": {},
                "development_manifest": reference,
                "development_baseline": reference,
                "baseline_checkpoint": reference,
                "semantic_baseline_validation": {},
                "closed_world": {},
                "all_four_jobs_terminal_and_step_free": True,
                "exact_registered_root_verified": True,
                "same_uid_external_compute_excluded": False,
                "ordinary_batch_children_independently_attested": False,
                "claim_limited_to_exact_final_rooted_files_and_slurm_rows": True,
                "resource_values_are_diagnostic_only": True,
                "required_before_any_performance_claim": True,
                "external_sha256_anchor_required_before_performance_claim": True,
                "performance_claim_ready": False,
                "confirmation_authorized": False,
                "promotion_authorized": False,
            }
        )
        receipt_raw = builder.canonical_json_bytes(receipt) + b"\n"
        receipt_sha256 = hashlib.sha256(receipt_raw).hexdigest()
        receipt_path.write_bytes(receipt_raw)
        receipt_path.chmod(0o444)
        receipt_reference = {"path": str(receipt_path), "sha256": receipt_sha256}
        plan = {
            "accounting": {
                "terminal_receipt": str(receipt_path),
                "monitor_stage": {"held_slurm_job_id": "740999"},
            }
        }
        log_path = log_root / "acw_development_monitor_740999.out"
        log_path.write_bytes(b"monitor started\n")
        receipt_identity = builder.publication.snapshot_file(receipt_path)
        log_identity = log_path.stat()
        verification_root = Path(self.temporary.name) / "terminal-verification"
        verification_root.mkdir()
        verification_path = verification_root / builder.TERMINAL_VERIFICATION_NAME
        attestation = builder._hash_bound(
            {
                "schema": "r12_acw_development_terminal_verification_v1",
                "protocol": "R12-ACW-DEVELOPMENT-TERMINAL-VERIFICATION-v1",
                "development_plan": reference,
                "scientific_commit": "d" * 40,
                "monitor": self._monitor_binding(),
                "terminal_accounting": {
                    "path": str(receipt_path),
                    **receipt_identity,
                },
                "terminal_accounting_payload_sha256": receipt["payload_sha256"],
                "monitor_log_descriptor": {
                    "path": str(log_path),
                    "st_dev": log_identity.st_dev,
                    "st_ino": log_identity.st_ino,
                },
                "terminal_accounting_fully_recomputed_before_attestation": True,
                "established_before_monitor_completion": True,
                "confirmation_authorized": False,
                "promotion_authorized": False,
            }
        )
        verification_path.write_bytes(builder.canonical_json_bytes(attestation) + b"\n")
        verification_path.chmod(0o444)
        verification_root.chmod(0o555)
        verification_record = builder.publication.snapshot_file(verification_path)
        log_path.write_bytes(
            (
                "[acw-development-terminal-verify] "
                f"sha256={receipt_sha256} "
                f"payload_sha256={receipt['payload_sha256']} "
                f"verification_sha256={verification_record['sha256']} "
                "external_anchor_required=1 performance_claim_ready=0\n"
                f"[acw-development-monitor] complete root={self.root} "
                "performance_claim_ready=0\n"
            ).encode("ascii")
        )
        rows = [{"job_id_raw": "740999", "state": "COMPLETED"}]
        anchor_path = Path(self.temporary.name) / "monitor-anchor.json"
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(builder, "BASE", base),
            mock.patch.object(builder, "TERMINAL_ACCOUNTING_PATH", receipt_path),
            mock.patch.object(
                builder,
                "TERMINAL_VERIFICATION_ROOT",
                verification_root,
            ),
            mock.patch.object(builder, "_plan_for_root", return_value=plan),
            mock.patch.object(builder, "validate_plan", return_value=plan),
            mock.patch.object(builder, "_validated_g_commit", return_value="d" * 40),
            mock.patch.object(builder, "_plan_reference", return_value=reference),
            mock.patch.object(
                builder,
                "_validate_recorded_monitor_binding",
                return_value=self._monitor_binding(),
            ),
            mock.patch.object(
                builder, "_query_monitor_terminal_rows", return_value=rows
            ),
            mock.patch.object(
                builder,
                "_reference",
                return_value=receipt_reference,
            ),
            mock.patch.object(builder, "MONITOR_ANCHOR_PATH", anchor_path),
            mock.patch.object(builder, "_retain_terminal_closed_world_from_receipt"),
        ):
            envelope, anchor_digest = builder.publish_monitor_anchor_ready_envelope(
                self.root,
                externally_anchored_verification_sha256=verification_record["sha256"],
            )
        self.assertEqual(anchor_digest, file_sha256(anchor_path))
        self.assertEqual(stat.S_IMODE(anchor_path.stat().st_mode), 0o444)
        self.assertEqual(envelope["monitor_terminal_rows"], rows)
        self.assertTrue(envelope["monitor_completed_zero_exit_and_step_free"])
        self.assertFalse(envelope["performance_claim_ready"])
        self.assertEqual(envelope["monitor_log"]["sha256"], file_sha256(log_path))
        self.assertEqual(envelope["monitor_log"]["mode"], "0444")
        self.assertEqual(stat.S_IMODE(log_path.stat().st_mode), 0o444)
        self.assertEqual(
            envelope["terminal_verification"]["verification"]["sha256"],
            verification_record["sha256"],
        )
        self.assertEqual(
            envelope["externally_anchored_terminal_verification_sha256"],
            verification_record["sha256"],
        )

        collision_anchor = Path(self.temporary.name) / "raced-monitor-anchor.json"
        real_rename_no_replace = builder.publication.rename_no_replace
        collision_injected = False

        def collide_at_native_anchor_rename(source: Path, destination: Path) -> None:
            nonlocal collision_injected
            if Path(destination) == collision_anchor and not collision_injected:
                collision_anchor.write_bytes(
                    builder.canonical_json_bytes(envelope) + b"\n"
                )
                collision_anchor.chmod(0o444)
                collision_injected = True
            real_rename_no_replace(source, destination)

        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(builder, "BASE", base),
            mock.patch.object(builder, "TERMINAL_ACCOUNTING_PATH", receipt_path),
            mock.patch.object(
                builder,
                "TERMINAL_VERIFICATION_ROOT",
                verification_root,
            ),
            mock.patch.object(builder, "_plan_for_root", return_value=plan),
            mock.patch.object(builder, "validate_plan", return_value=plan),
            mock.patch.object(builder, "_validated_g_commit", return_value="d" * 40),
            mock.patch.object(builder, "_plan_reference", return_value=reference),
            mock.patch.object(
                builder,
                "_validate_recorded_monitor_binding",
                return_value=self._monitor_binding(),
            ),
            mock.patch.object(
                builder,
                "_query_monitor_terminal_rows",
                return_value=rows,
            ),
            mock.patch.object(
                builder,
                "_reference",
                return_value=receipt_reference,
            ),
            mock.patch.object(builder, "MONITOR_ANCHOR_PATH", collision_anchor),
            mock.patch.object(builder, "_retain_terminal_closed_world_from_receipt"),
            mock.patch.object(
                builder.publication,
                "rename_no_replace",
                side_effect=collide_at_native_anchor_rename,
            ),
            self.assertRaises(FileExistsError) as collision,
        ):
            builder.publish_monitor_anchor_ready_envelope(
                self.root,
                externally_anchored_verification_sha256=verification_record["sha256"],
            )
        self.assertTrue(collision_injected)
        self.assertEqual(collision.exception.errno, errno.EEXIST)

        def assert_late_anchor_swap_rejected(
            target: Path,
            *,
            label: str,
        ) -> None:
            replacement = target.with_name(f"{target.name}.{label}.replacement")
            replacement.write_bytes(builder.canonical_json_bytes(envelope) + b"\n")
            replacement.chmod(0o444)
            real_retained_evidence_snapshot = (
                builder.publication.retained_evidence_snapshot
            )
            swapped = False
            original_identity: tuple[int, int] | None = None
            replacement_identity: tuple[int, int] | None = None

            @contextmanager
            def retain_and_swap():
                nonlocal swapped, original_identity, replacement_identity
                with real_retained_evidence_snapshot() as retained:
                    try:
                        yield retained
                    finally:
                        retained_files = set(retained._files)
                        if (
                            receipt_path.absolute() in retained_files
                            and target.absolute() in retained_files
                            and not swapped
                        ):
                            original = target.stat()
                            original_identity = (original.st_dev, original.st_ino)
                            target.unlink()
                            replacement.rename(target)
                            installed = target.stat()
                            replacement_identity = (
                                installed.st_dev,
                                installed.st_ino,
                            )
                            swapped = True

            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(builder, "BASE", base),
                mock.patch.object(
                    builder,
                    "TERMINAL_ACCOUNTING_PATH",
                    receipt_path,
                ),
                mock.patch.object(
                    builder,
                    "TERMINAL_VERIFICATION_ROOT",
                    verification_root,
                ),
                mock.patch.object(builder, "_plan_for_root", return_value=plan),
                mock.patch.object(builder, "validate_plan", return_value=plan),
                mock.patch.object(
                    builder,
                    "_validated_g_commit",
                    return_value="d" * 40,
                ),
                mock.patch.object(
                    builder,
                    "_plan_reference",
                    return_value=reference,
                ),
                mock.patch.object(
                    builder,
                    "_validate_recorded_monitor_binding",
                    return_value=self._monitor_binding(),
                ),
                mock.patch.object(
                    builder,
                    "_query_monitor_terminal_rows",
                    return_value=rows,
                ),
                mock.patch.object(
                    builder,
                    "_reference",
                    return_value=receipt_reference,
                ),
                mock.patch.object(builder, "MONITOR_ANCHOR_PATH", target),
                mock.patch.object(
                    builder,
                    "_retain_terminal_closed_world_from_receipt",
                ),
                mock.patch.object(
                    builder.publication,
                    "retained_evidence_snapshot",
                    side_effect=retain_and_swap,
                ),
                self.assertRaisesRegex(OSError, "retained ACW"),
            ):
                builder.publish_monitor_anchor_ready_envelope(
                    self.root,
                    externally_anchored_verification_sha256=(
                        verification_record["sha256"]
                    ),
                )
            self.assertTrue(swapped)
            self.assertIsNotNone(original_identity)
            self.assertIsNotNone(replacement_identity)
            self.assertNotEqual(original_identity, replacement_identity)
            self.assertEqual(
                target.read_bytes(),
                builder.canonical_json_bytes(envelope) + b"\n",
            )
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o444)

        assert_late_anchor_swap_rejected(
            Path(self.temporary.name) / "new-anchor-return-race.json",
            label="new",
        )
        assert_late_anchor_swap_rejected(anchor_path, label="retry")

        receipt_replacement = Path(self.temporary.name) / "late-receipt-replacement"
        receipt_replacement.write_bytes(receipt_raw)
        receipt_replacement.chmod(0o444)
        displaced_receipt = Path(self.temporary.name) / "late-receipt-displaced"
        original_receipt_identity = (
            receipt_path.stat().st_dev,
            receipt_path.stat().st_ino,
        )
        replaced_receipt_identity: tuple[int, int] | None = None
        receipt_swapped = False
        real_verify_names = builder.publication.RetainedEvidenceSnapshot._verify_names

        def swap_receipt_after_first_joint_name_pass(
            retained,
            *,
            reverse: bool = False,
        ) -> None:
            nonlocal receipt_swapped, replaced_receipt_identity
            real_verify_names(retained, reverse=reverse)
            retained_files = set(retained._files)
            if (
                not reverse
                and receipt_path.absolute() in retained_files
                and anchor_path.absolute() in retained_files
                and not receipt_swapped
            ):
                receipt_path.rename(displaced_receipt)
                receipt_replacement.rename(receipt_path)
                metadata = receipt_path.stat()
                replaced_receipt_identity = (metadata.st_dev, metadata.st_ino)
                receipt_swapped = True

        try:
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(builder, "BASE", base),
                mock.patch.object(
                    builder,
                    "TERMINAL_ACCOUNTING_PATH",
                    receipt_path,
                ),
                mock.patch.object(
                    builder,
                    "TERMINAL_VERIFICATION_ROOT",
                    verification_root,
                ),
                mock.patch.object(builder, "_plan_for_root", return_value=plan),
                mock.patch.object(builder, "validate_plan", return_value=plan),
                mock.patch.object(
                    builder,
                    "_validated_g_commit",
                    return_value="d" * 40,
                ),
                mock.patch.object(
                    builder,
                    "_plan_reference",
                    return_value=reference,
                ),
                mock.patch.object(
                    builder,
                    "_validate_recorded_monitor_binding",
                    return_value=self._monitor_binding(),
                ),
                mock.patch.object(
                    builder,
                    "_query_monitor_terminal_rows",
                    return_value=rows,
                ),
                mock.patch.object(
                    builder,
                    "_reference",
                    return_value=receipt_reference,
                ),
                mock.patch.object(builder, "MONITOR_ANCHOR_PATH", anchor_path),
                mock.patch.object(
                    builder,
                    "_retain_terminal_closed_world_from_receipt",
                ),
                mock.patch.object(
                    builder.publication.RetainedEvidenceSnapshot,
                    "_verify_names",
                    autospec=True,
                    side_effect=swap_receipt_after_first_joint_name_pass,
                ),
                self.assertRaisesRegex(OSError, "retained ACW"),
            ):
                builder.publish_monitor_anchor_ready_envelope(
                    self.root,
                    externally_anchored_verification_sha256=(
                        verification_record["sha256"]
                    ),
                )
        finally:
            if displaced_receipt.exists():
                receipt_path.unlink()
                displaced_receipt.rename(receipt_path)
        self.assertTrue(receipt_swapped)
        self.assertIsNotNone(replaced_receipt_identity)
        self.assertNotEqual(original_receipt_identity, replaced_receipt_identity)
        self.assertEqual(
            (receipt_path.stat().st_dev, receipt_path.stat().st_ino),
            original_receipt_identity,
        )

        forged = builder._hash_bound({**receipt, "stages": {"forged": {}}})
        forged_raw = builder.canonical_json_bytes(forged) + b"\n"
        forged_sha256 = hashlib.sha256(forged_raw).hexdigest()
        forged_receipt_path = Path(self.temporary.name) / "forged-terminal-stage.json"
        forged_receipt_path.write_bytes(forged_raw)
        forged_receipt_path.chmod(0o444)
        forged_log_path = Path(self.temporary.name) / "forged-monitor-stage.out"
        forged_log_path.write_bytes(b"forged monitor output in progress\n")
        forged_receipt_identity = builder.publication.snapshot_file(forged_receipt_path)
        forged_log_identity = forged_log_path.stat()
        displaced_verification = verification_root.with_name(
            f"{verification_root.name}-displaced"
        )
        forged_verification_root = verification_root.with_name(
            f"{verification_root.name}-forged-stage"
        )
        forged_verification_root.mkdir()
        forged_verification_path = (
            forged_verification_root / builder.TERMINAL_VERIFICATION_NAME
        )
        forged_attestation = builder._hash_bound(
            {
                "schema": "r12_acw_development_terminal_verification_v1",
                "protocol": "R12-ACW-DEVELOPMENT-TERMINAL-VERIFICATION-v1",
                "development_plan": reference,
                "scientific_commit": "d" * 40,
                "monitor": self._monitor_binding(),
                "terminal_accounting": {
                    "path": str(receipt_path),
                    **forged_receipt_identity,
                },
                "terminal_accounting_payload_sha256": forged["payload_sha256"],
                "monitor_log_descriptor": {
                    "path": str(log_path),
                    "st_dev": forged_log_identity.st_dev,
                    "st_ino": forged_log_identity.st_ino,
                },
                "terminal_accounting_fully_recomputed_before_attestation": True,
                "established_before_monitor_completion": True,
                "confirmation_authorized": False,
                "promotion_authorized": False,
            }
        )
        forged_verification_path.write_bytes(
            builder.canonical_json_bytes(forged_attestation) + b"\n"
        )
        builder.publication.freeze_tree(forged_verification_root)
        forged_verification_record = builder.publication.snapshot_file(
            forged_verification_path
        )
        forged_log_path.write_bytes(
            (
                "[acw-development-terminal-verify] "
                f"sha256={forged_sha256} "
                f"payload_sha256={forged['payload_sha256']} "
                f"verification_sha256={forged_verification_record['sha256']} "
                "external_anchor_required=1 performance_claim_ready=0\n"
                f"[acw-development-monitor] complete root={self.root} "
                "performance_claim_ready=0\n"
            ).encode("ascii")
        )
        forged_log_path.chmod(0o444)

        race_anchor_path = Path(self.temporary.name) / "racing-monitor-anchor.json"
        real_retained_bytes_publication = builder.publication.retained_bytes_publication

        @contextmanager
        def publish_then_replace(
            path: Path,
            raw: bytes,
            *,
            mode: int = 0o444,
        ):
            with real_retained_bytes_publication(path, raw, mode=mode) as record:
                verification_root.rename(displaced_verification)
                forged_verification_root.rename(verification_root)
                log_path.unlink()
                forged_log_path.rename(log_path)
                receipt_path.unlink()
                forged_receipt_path.rename(receipt_path)
                yield record

        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(builder, "BASE", base),
            mock.patch.object(builder, "TERMINAL_ACCOUNTING_PATH", receipt_path),
            mock.patch.object(
                builder,
                "TERMINAL_VERIFICATION_ROOT",
                verification_root,
            ),
            mock.patch.object(builder, "MONITOR_ANCHOR_PATH", race_anchor_path),
            mock.patch.object(builder, "_retain_terminal_closed_world_from_receipt"),
            mock.patch.object(builder, "_plan_for_root", return_value=plan),
            mock.patch.object(builder, "validate_plan", return_value=plan),
            mock.patch.object(builder, "_validated_g_commit", return_value="d" * 40),
            mock.patch.object(builder, "_plan_reference", return_value=reference),
            mock.patch.object(
                builder,
                "_validate_recorded_monitor_binding",
                return_value=self._monitor_binding(),
            ),
            mock.patch.object(
                builder, "_query_monitor_terminal_rows", return_value=rows
            ),
            mock.patch.object(
                builder,
                "_reference",
                return_value=receipt_reference,
            ),
            mock.patch.object(
                builder.publication,
                "retained_bytes_publication",
                side_effect=publish_then_replace,
            ),
            self.assertRaisesRegex(OSError, "retained ACW"),
        ):
            builder.publish_monitor_anchor_ready_envelope(
                self.root,
                externally_anchored_verification_sha256=verification_record["sha256"],
            )
        self.assertTrue(race_anchor_path.is_file())
        self.assertEqual(stat.S_IMODE(race_anchor_path.stat().st_mode), 0o444)

        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(builder, "BASE", base),
            mock.patch.object(builder, "TERMINAL_ACCOUNTING_PATH", receipt_path),
            mock.patch.object(
                builder,
                "TERMINAL_VERIFICATION_ROOT",
                verification_root,
            ),
            mock.patch.object(builder, "_plan_for_root", return_value=plan),
            mock.patch.object(builder, "validate_plan", return_value=plan),
            mock.patch.object(builder, "_validated_g_commit", return_value="d" * 40),
            mock.patch.object(builder, "_plan_reference", return_value=reference),
            mock.patch.object(
                builder,
                "_validate_recorded_monitor_binding",
                return_value=self._monitor_binding(),
            ),
            mock.patch.object(
                builder, "_query_monitor_terminal_rows", return_value=rows
            ),
            self.assertRaisesRegex(
                ValueError,
                "external terminal verification anchor differs",
            ),
        ):
            builder.build_monitor_anchor_ready_envelope(
                self.root,
                externally_anchored_verification_sha256=verification_record["sha256"],
            )


if __name__ == "__main__":
    unittest.main()
