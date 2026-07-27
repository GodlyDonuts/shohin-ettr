from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import BinaryIO, Callable

import pytest

import publish_ettr_il_v3_hf as publisher


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_corpus(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "dataset"
    (root / "shards" / "train").mkdir(parents=True)
    (root / "shards" / "development").mkdir(parents=True)
    card = b"---\npretty_name: Shohin ETTR IL v3\n---\n\nFrozen test card.\n"
    train = b'{"split":"train","value":1}\n'
    development = b'{"split":"development","value":2}\n'
    (root / "README.md").write_bytes(card)
    (root / "shards" / "train" / "part-00000.jsonl").write_bytes(train)
    (root / "shards" / "development" / "part-00000.jsonl").write_bytes(
        development
    )
    manifest = {
        "card": {
            "path": "README.md",
            "sha256": _sha256(card),
            "size_bytes": len(card),
        },
        "dataset_protocol": publisher.DATASET_PROTOCOL,
        "schema": publisher.MANIFEST_SCHEMA,
        "shards": [
            {
                "path": "shards/train/part-00000.jsonl",
                "sha256": _sha256(train),
                "size_bytes": len(train),
                "split": "train",
            },
            {
                "path": "shards/development/part-00000.jsonl",
                "sha256": _sha256(development),
                "size_bytes": len(development),
                "split": "development",
            },
        ],
    }
    manifest_path = root / "publication_manifest.json"
    manifest_path.write_bytes(publisher._canonical_json_bytes(manifest))
    return root, manifest_path


class FakeGateway:
    def __init__(self) -> None:
        self.private_requests: list[tuple[str, bool]] = []
        self.revisions: set[str] = set()
        self.files: dict[tuple[str, str], bytes] = {}
        self.uploads: list[str] = []
        self.unverifiable: set[tuple[str, str]] = set()
        self.on_upload: Callable[[str], None] | None = None

    def ensure_dataset_repo(self, repo_id: str, *, private: bool) -> None:
        self.private_requests.append((repo_id, private))

    def revision_exists(self, repo_id: str, revision: str) -> bool:
        del repo_id
        return revision in self.revisions

    def create_revision(self, repo_id: str, revision: str) -> None:
        del repo_id
        self.revisions.add(revision)

    def list_files(self, repo_id: str, revision: str) -> set[str]:
        del repo_id
        return {
            path
            for (stored_revision, path), _payload in self.files.items()
            if stored_revision == revision
        }

    def remote_sha256(
        self,
        repo_id: str,
        revision: str,
        remote_path: str,
    ) -> str | None:
        del repo_id
        key = (revision, remote_path)
        if key in self.unverifiable:
            return None
        payload = self.files.get(key)
        return None if payload is None else _sha256(payload)

    def upload_file(
        self,
        repo_id: str,
        revision: str,
        remote_path: str,
        fileobj: BinaryIO,
    ) -> None:
        del repo_id
        self.uploads.append(remote_path)
        self.files[(revision, remote_path)] = fileobj.read()
        if self.on_upload is not None:
            self.on_upload(remote_path)


def _plan(tmp_path: Path) -> tuple[publisher.PublicationPlan, Path]:
    root, manifest = _write_corpus(tmp_path)
    return publisher.build_publication_plan(root, manifest), root


def test_dry_run_is_private_credential_free_and_content_addressed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _root = _plan(tmp_path)
    monkeypatch.delenv(publisher.TOKEN_ENVIRONMENT_VARIABLE, raising=False)
    receipt_path = tmp_path / "dry-run-receipt.json"
    receipt = publisher.publish_plan(
        plan,
        repo_id=publisher.DEFAULT_REPO_ID,
        revision="release/ettr-v3-test",
        receipt_path=receipt_path,
        dry_run=True,
    )

    assert receipt["status"] == "dry_run_validated"
    assert receipt["private"] is True
    assert receipt["repo_id"] == "Godlydonuts/shohin-ettr-il-v3"
    assert all(item["action"] == "would_upload" for item in receipt["operations"])
    shard_operations = [
        item for item in receipt["operations"] if item["role"] == "shard"
    ]
    assert shard_operations
    assert all(
        item["remote_path"].startswith(
            f"shards/sha256/{item['sha256'][:2]}/{item['sha256']}/"
        )
        for item in shard_operations
    )
    assert {
        item["remote_path"]
        for item in receipt["operations"]
        if item["role"] in {"card", "manifest"}
    } == {
        f"{plan.release_prefix}/README.md",
        f"{plan.release_prefix}/manifest.json",
    }
    assert json.loads(receipt_path.read_text("ascii")) == receipt
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o444


def test_main_repo_rejects_raw_confirmation_payload(tmp_path: Path) -> None:
    root, manifest_path = _write_corpus(tmp_path)
    confirmation_dir = root / "shards" / "confirmation"
    confirmation_dir.mkdir()
    confirmation = b'{"split":"confirmation","value":3}\n'
    confirmation_path = confirmation_dir / "part-00000.jsonl"
    confirmation_path.write_bytes(confirmation)
    manifest = json.loads(manifest_path.read_text("ascii"))
    manifest["shards"].append(
        {
            "path": "shards/confirmation/part-00000.jsonl",
            "sha256": _sha256(confirmation),
            "size_bytes": len(confirmation),
            "split": "confirmation",
        }
    )
    manifest_path.write_bytes(publisher._canonical_json_bytes(manifest))

    with pytest.raises(
        publisher.PublicationError,
        match="invalid split path",
    ):
        publisher.build_publication_plan(root, manifest_path)


def test_new_revision_uploads_then_existing_revision_skips_exact_matches(
    tmp_path: Path,
) -> None:
    plan, _root = _plan(tmp_path)
    gateway = FakeGateway()
    first = publisher.publish_plan(
        plan,
        repo_id=publisher.DEFAULT_REPO_ID,
        revision="release/immutable-a",
        receipt_path=tmp_path / "first.json",
        new_revision=True,
        gateway=gateway,
    )
    assert gateway.private_requests == [(publisher.DEFAULT_REPO_ID, True)]
    assert len(gateway.uploads) == len(plan.files)
    assert all(
        operation["action"] == "uploaded_and_verified"
        for operation in first["operations"]
    )
    assert first["operations"][-1]["role"] == "manifest"

    gateway.uploads.clear()
    second = publisher.publish_plan(
        plan,
        repo_id=publisher.DEFAULT_REPO_ID,
        revision="release/immutable-a",
        receipt_path=tmp_path / "second.json",
        gateway=gateway,
    )
    assert gateway.uploads == []
    assert all(
        operation["action"] == "skipped_matching_remote"
        for operation in second["operations"]
    )


def test_remote_mismatch_fails_before_any_upload(tmp_path: Path) -> None:
    plan, _root = _plan(tmp_path)
    gateway = FakeGateway()
    revision = "release/collision"
    gateway.revisions.add(revision)
    gateway.files[(revision, plan.files[-1].remote_path)] = b"different-content"

    with pytest.raises(
        publisher.PublicationError,
        match="different content",
    ):
        publisher.publish_plan(
            plan,
            repo_id=publisher.DEFAULT_REPO_ID,
            revision=revision,
            receipt_path=tmp_path / "receipt.json",
            gateway=gateway,
        )
    assert gateway.uploads == []
    assert not (tmp_path / "receipt.json").exists()


def test_unverifiable_existing_remote_fails_closed(tmp_path: Path) -> None:
    plan, _root = _plan(tmp_path)
    gateway = FakeGateway()
    revision = "release/unverifiable"
    gateway.revisions.add(revision)
    remote_path = plan.files[0].remote_path
    gateway.files[(revision, remote_path)] = b"opaque"
    gateway.unverifiable.add((revision, remote_path))

    with pytest.raises(
        publisher.PublicationError,
        match="cannot be verified",
    ):
        publisher.publish_plan(
            plan,
            repo_id=publisher.DEFAULT_REPO_ID,
            revision=revision,
            receipt_path=tmp_path / "receipt.json",
            gateway=gateway,
        )
    assert gateway.uploads == []


def test_revision_creation_is_explicit_and_never_reuses_new_name(
    tmp_path: Path,
) -> None:
    plan, _root = _plan(tmp_path)
    gateway = FakeGateway()
    with pytest.raises(publisher.PublicationError, match="does not exist"):
        publisher.publish_plan(
            plan,
            repo_id=publisher.DEFAULT_REPO_ID,
            revision="release/not-created",
            receipt_path=tmp_path / "missing.json",
            gateway=gateway,
        )

    gateway.revisions.add("release/already-there")
    with pytest.raises(publisher.PublicationError, match="already exists"):
        publisher.publish_plan(
            plan,
            repo_id=publisher.DEFAULT_REPO_ID,
            revision="release/already-there",
            receipt_path=tmp_path / "existing.json",
            new_revision=True,
            gateway=gateway,
        )
    assert gateway.uploads == []


def test_symlinked_declared_file_is_rejected(tmp_path: Path) -> None:
    root, manifest = _write_corpus(tmp_path)
    shard = root / "shards" / "train" / "part-00000.jsonl"
    target = tmp_path / "outside.jsonl"
    target.write_bytes(shard.read_bytes())
    shard.unlink()
    shard.symlink_to(target)

    with pytest.raises(publisher.PublicationError, match="symlink rejected"):
        publisher.build_publication_plan(root, manifest)


def test_hard_linked_declared_file_is_rejected(tmp_path: Path) -> None:
    root, manifest = _write_corpus(tmp_path)
    shard = root / "shards" / "train" / "part-00000.jsonl"
    os.link(shard, tmp_path / "alias.jsonl")
    with pytest.raises(publisher.PublicationError, match="hard-linked"):
        publisher.build_publication_plan(root, manifest)


@pytest.mark.parametrize("failure", ("missing", "hash", "unexpected"))
def test_missing_hash_mismatch_and_unexpected_files_fail_closed(
    tmp_path: Path,
    failure: str,
) -> None:
    root, manifest = _write_corpus(tmp_path)
    shard = root / "shards" / "train" / "part-00000.jsonl"
    if failure == "missing":
        shard.unlink()
        match = "missing"
    elif failure == "hash":
        original = shard.read_bytes()
        shard.write_bytes(b"x" * len(original))
        match = "SHA-256 mismatch"
    else:
        (root / "notes.txt").write_text("not authorized\n", encoding="ascii")
        match = "unexpected files"
    with pytest.raises(publisher.PublicationError, match=match):
        publisher.build_publication_plan(root, manifest)


def test_unsafe_manifest_shard_path_is_rejected(tmp_path: Path) -> None:
    root, manifest_path = _write_corpus(tmp_path)
    manifest = json.loads(manifest_path.read_text("ascii"))
    manifest["shards"][0]["path"] = "../outside.jsonl"
    manifest_path.write_bytes(publisher._canonical_json_bytes(manifest))
    with pytest.raises(publisher.PublicationError, match="unsafe path component"):
        publisher.build_publication_plan(root, manifest_path)


def test_symlinked_parent_directory_is_rejected_before_file_read(
    tmp_path: Path,
) -> None:
    root, manifest_path = _write_corpus(tmp_path)
    train_directory = root / "shards" / "train"
    outside_directory = tmp_path / "outside-train"
    train_directory.rename(outside_directory)
    train_directory.symlink_to(outside_directory, target_is_directory=True)
    with pytest.raises(publisher.PublicationError, match="safely|symlink"):
        publisher.build_publication_plan(root, manifest_path)


def test_unexpected_remote_release_member_is_rejected(tmp_path: Path) -> None:
    plan, _root = _plan(tmp_path)
    gateway = FakeGateway()
    revision = "release/occupied"
    gateway.revisions.add(revision)
    gateway.files[(revision, f"{plan.release_prefix}/surprise.txt")] = b"surprise"
    with pytest.raises(publisher.PublicationError, match="unexpected files"):
        publisher.publish_plan(
            plan,
            repo_id=publisher.DEFAULT_REPO_ID,
            revision=revision,
            receipt_path=tmp_path / "receipt.json",
            gateway=gateway,
        )
    assert gateway.uploads == []


def test_receipt_never_contains_token_and_cannot_be_overwritten(
    tmp_path: Path,
) -> None:
    plan, _root = _plan(tmp_path)
    secret = "hf_secret-that-must-never-appear"
    assert publisher._token_from_environment({"HF_TOKEN": secret}) == secret
    with pytest.raises(publisher.PublicationError, match="must be set"):
        publisher._token_from_environment({})

    receipt_path = tmp_path / "receipt.json"
    publisher.publish_plan(
        plan,
        repo_id=publisher.DEFAULT_REPO_ID,
        revision="release/no-secret",
        receipt_path=receipt_path,
        dry_run=True,
    )
    assert secret not in receipt_path.read_text("ascii")
    with pytest.raises(publisher.PublicationError, match="overwrite"):
        publisher.publish_plan(
            plan,
            repo_id=publisher.DEFAULT_REPO_ID,
            revision="release/no-secret",
            receipt_path=receipt_path,
            dry_run=True,
        )


def test_local_mutation_during_upload_is_detected(tmp_path: Path) -> None:
    plan, root = _plan(tmp_path)
    gateway = FakeGateway()
    revision = "release/race"
    gateway.revisions.add(revision)
    shard = root / "shards" / "train" / "part-00000.jsonl"

    def mutate_after_first_upload(_remote_path: str) -> None:
        if shard.exists():
            shard.write_bytes(shard.read_bytes() + b"changed\n")
        gateway.on_upload = None

    gateway.on_upload = mutate_after_first_upload
    with pytest.raises(publisher.PublicationError, match="changed during publication"):
        publisher.publish_plan(
            plan,
            repo_id=publisher.DEFAULT_REPO_ID,
            revision=revision,
            receipt_path=tmp_path / "receipt.json",
            gateway=gateway,
        )
    assert not (tmp_path / "receipt.json").exists()


@pytest.mark.parametrize(
    "repo_id,revision",
    (
        ("owner", "release/v1"),
        ("owner/repo/extra", "release/v1"),
        ("owner/repo", "main"),
        ("owner/repo", "../release"),
        ("owner/repo", "release//v1"),
    ),
)
def test_unsafe_repo_ids_and_revisions_are_rejected(
    tmp_path: Path,
    repo_id: str,
    revision: str,
) -> None:
    plan, _root = _plan(tmp_path)
    with pytest.raises(publisher.PublicationError):
        publisher.publish_plan(
            plan,
            repo_id=repo_id,
            revision=revision,
            receipt_path=tmp_path / "receipt.json",
            dry_run=True,
        )


def test_cli_dry_run_uses_safe_default_without_hf_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, manifest = _write_corpus(tmp_path)
    receipt = tmp_path / "cli-receipt.json"
    monkeypatch.delenv("HF_TOKEN", raising=False)
    result = publisher.main(
        [
            "--dataset-root",
            str(root),
            "--manifest",
            str(manifest),
            "--revision",
            "release/cli-dry-run",
            "--receipt",
            str(receipt),
            "--dry-run",
        ]
    )
    output = capsys.readouterr()
    assert result == 0
    assert output.err == ""
    assert "dry_run_validated" in output.out
    parsed = json.loads(receipt.read_text("ascii"))
    assert parsed["repo_id"] == publisher.DEFAULT_REPO_ID
    assert parsed["private"] is True
