from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from ettr_factorial_authority import (
    ETTRCustodyAuthorityError,
    load_ettr_custody_root,
    make_root_signed_ettr_custody_authority,
    read_root_signed_ettr_custody_authority,
    write_ettr_custody_authority_once,
)


BOARD_SHA256 = "a" * 64
MANIFEST_SHA256 = "b" * 64


def _public_key_bytes(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _write_root(path: Path, private_key: Ed25519PrivateKey) -> str:
    payload = _public_key_bytes(private_key)
    path.write_bytes(payload)
    path.chmod(0o444)
    return hashlib.sha256(payload).hexdigest()


def _authority(
    root: Ed25519PrivateKey,
    signer: Ed25519PrivateKey,
):
    return make_root_signed_ettr_custody_authority(
        root_private_key=root,
        custody_public_key_hex=_public_key_bytes(signer).hex(),
        board_sha256=BOARD_SHA256,
        execution_manifest_sha256=MANIFEST_SHA256,
    )


def test_root_signed_authority_round_trips_from_immutable_file(
    tmp_path: Path,
) -> None:
    root = Ed25519PrivateKey.generate()
    signer = Ed25519PrivateKey.generate()
    root_path = tmp_path / "root.pub"
    root_sha256 = _write_root(root_path, root)
    authority = _authority(root, signer)
    authority_path = tmp_path / "authority.json"
    assert (
        write_ettr_custody_authority_once(authority_path, authority)
        == authority.sha256()
    )

    trust = load_ettr_custody_root(
        root_path,
        pinned_public_key_sha256=root_sha256,
    )
    assert (
        read_root_signed_ettr_custody_authority(
            authority_path,
            root_trust=trust,
            expected_record_sha256=authority.sha256(),
            expected_board_sha256=BOARD_SHA256,
            expected_execution_manifest_sha256=MANIFEST_SHA256,
        )
        == authority
    )


def test_authority_rejects_writable_symlink_and_hardlink_files(
    tmp_path: Path,
) -> None:
    root = Ed25519PrivateKey.generate()
    signer = Ed25519PrivateKey.generate()
    root_path = tmp_path / "root.pub"
    root_sha256 = _write_root(root_path, root)
    authority = _authority(root, signer)
    authority_path = tmp_path / "authority.json"
    write_ettr_custody_authority_once(authority_path, authority)
    trust = load_ettr_custody_root(
        root_path,
        pinned_public_key_sha256=root_sha256,
    )

    authority_path.chmod(0o644)
    with pytest.raises(ETTRCustodyAuthorityError):
        read_root_signed_ettr_custody_authority(
            authority_path,
            root_trust=trust,
            expected_record_sha256=authority.sha256(),
            expected_board_sha256=BOARD_SHA256,
            expected_execution_manifest_sha256=MANIFEST_SHA256,
        )
    authority_path.chmod(0o444)
    hardlink = tmp_path / "authority-hardlink.json"
    hardlink.hardlink_to(authority_path)
    with pytest.raises(ETTRCustodyAuthorityError):
        read_root_signed_ettr_custody_authority(
            authority_path,
            root_trust=trust,
            expected_record_sha256=authority.sha256(),
            expected_board_sha256=BOARD_SHA256,
            expected_execution_manifest_sha256=MANIFEST_SHA256,
        )
    hardlink.unlink()
    symlink = tmp_path / "authority-symlink.json"
    symlink.symlink_to(authority_path)
    with pytest.raises(ETTRCustodyAuthorityError):
        read_root_signed_ettr_custody_authority(
            symlink,
            root_trust=trust,
            expected_record_sha256=authority.sha256(),
            expected_board_sha256=BOARD_SHA256,
            expected_execution_manifest_sha256=MANIFEST_SHA256,
        )


def test_authority_rejects_same_root_and_signer_key() -> None:
    root = Ed25519PrivateKey.generate()
    with pytest.raises(ETTRCustodyAuthorityError):
        _authority(root, root)


def test_authority_rejects_noncanonical_json_and_extra_field(
    tmp_path: Path,
) -> None:
    root = Ed25519PrivateKey.generate()
    signer = Ed25519PrivateKey.generate()
    root_path = tmp_path / "root.pub"
    root_sha256 = _write_root(root_path, root)
    authority = _authority(root, signer)
    trust = load_ettr_custody_root(
        root_path,
        pinned_public_key_sha256=root_sha256,
    )
    for name, payload in (
        (
            "pretty.json",
            json.dumps(
                {
                    "schema": authority.schema,
                    "authorized_seal_schema": (
                        authority.authorized_seal_schema
                    ),
                    "root_public_key_sha256": (
                        authority.root_public_key_sha256
                    ),
                    "custody_public_key_hex": (
                        authority.custody_public_key_hex
                    ),
                    "board_sha256": authority.board_sha256,
                    "execution_manifest_sha256": (
                        authority.execution_manifest_sha256
                    ),
                    "root_signature_hex": authority.root_signature_hex,
                },
                indent=2,
            ).encode("ascii"),
        ),
        (
            "extra.json",
            authority.canonical_bytes()[:-2]
            + b',\"claimant_extra\":true}\\n',
        ),
    ):
        path = tmp_path / name
        path.write_bytes(payload)
        path.chmod(0o444)
        with pytest.raises(ETTRCustodyAuthorityError):
            read_root_signed_ettr_custody_authority(
                path,
                root_trust=trust,
                expected_record_sha256=hashlib.sha256(payload).hexdigest(),
                expected_board_sha256=BOARD_SHA256,
                expected_execution_manifest_sha256=MANIFEST_SHA256,
            )


def test_authority_rejects_wrong_board_manifest_and_signature(
    tmp_path: Path,
) -> None:
    root = Ed25519PrivateKey.generate()
    signer = Ed25519PrivateKey.generate()
    root_path = tmp_path / "root.pub"
    root_sha256 = _write_root(root_path, root)
    authority = _authority(root, signer)
    authority_path = tmp_path / "authority.json"
    write_ettr_custody_authority_once(authority_path, authority)
    trust = load_ettr_custody_root(
        root_path,
        pinned_public_key_sha256=root_sha256,
    )
    for board, manifest in (
        ("c" * 64, MANIFEST_SHA256),
        (BOARD_SHA256, "d" * 64),
    ):
        with pytest.raises(ETTRCustodyAuthorityError):
            read_root_signed_ettr_custody_authority(
                authority_path,
                root_trust=trust,
                expected_record_sha256=authority.sha256(),
                expected_board_sha256=board,
                expected_execution_manifest_sha256=manifest,
            )

    forged = replace(authority, board_sha256="c" * 64)
    authority_path.unlink()
    write_ettr_custody_authority_once(authority_path, forged)
    with pytest.raises(ETTRCustodyAuthorityError):
        read_root_signed_ettr_custody_authority(
            authority_path,
            root_trust=trust,
            expected_record_sha256=forged.sha256(),
            expected_board_sha256="c" * 64,
            expected_execution_manifest_sha256=MANIFEST_SHA256,
        )


def test_claimant_self_rooted_authority_fails_pinned_offline_root(
    tmp_path: Path,
) -> None:
    trusted_root = Ed25519PrivateKey.generate()
    attacker_root = Ed25519PrivateKey.generate()
    attacker_signer = Ed25519PrivateKey.generate()
    trusted_root_path = tmp_path / "trusted-root.pub"
    trusted_root_sha256 = _write_root(trusted_root_path, trusted_root)
    attacker_root_path = tmp_path / "attacker-root.pub"
    _write_root(attacker_root_path, attacker_root)
    attacker_authority = _authority(attacker_root, attacker_signer)
    authority_path = tmp_path / "authority.json"
    write_ettr_custody_authority_once(authority_path, attacker_authority)

    with pytest.raises(ETTRCustodyAuthorityError):
        load_ettr_custody_root(
            attacker_root_path,
            pinned_public_key_sha256=trusted_root_sha256,
        )
