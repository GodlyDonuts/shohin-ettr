from __future__ import annotations

from dataclasses import replace

import pytest

from endogenous_typed_theory_reactor import TheoryReactorError
from ettr_data_contract import (
    ETTRPacketSufficiencyIndex,
    ETTRPacketSufficiencyVerifier,
)
from ettr_episode import ETTREpisodeSegment
from ettr_packet_index import (
    ETTRDiskPacketSufficiencyIndex,
    ETTRPacketIndexError,
    build_disk_packet_index,
)
from test_ettr_train_step import _trainer


def _batches():
    _trainer_instance, train = _trainer(accumulation=1)
    validation_tokens = train.episodes.query.tokens.clone()
    validation_tokens[:, 0] = 9
    validation = replace(
        train,
        episodes=replace(
            train.episodes,
            query=ETTREpisodeSegment.from_tokens(validation_tokens),
        ),
    )
    return train, validation


def _make_index(tmp_path):
    train, validation = _batches()
    root = tmp_path / "packet-index"
    manifest = build_disk_packet_index(
        root,
        train_batches=(train,),
        validation_batches=(validation,),
    )
    return root, manifest, train, validation


def _make_writable(root):
    root.chmod(0o700)
    for path in root.iterdir():
        path.chmod(0o600)


def test_disk_index_matches_in_memory_receipt_and_verifies_splits(tmp_path):
    root, manifest, train, validation = _make_index(tmp_path)
    expected = ETTRPacketSufficiencyIndex.from_splits(
        (train,),
        (validation,),
    )
    with ETTRDiskPacketSufficiencyIndex(root) as index:
        assert isinstance(index, ETTRPacketSufficiencyVerifier)
        assert index.receipt == expected.receipt
        assert index.train_payload_sha256 == expected.train_payload_sha256
        assert (
            index.validation_payload_sha256
            == expected.validation_payload_sha256
        )
        assert manifest["manifest_payload_sha256"]
        index.verify_train((train,))
        index.verify_validation((validation,))
        with pytest.raises(TheoryReactorError, match="frozen train"):
            index.verify_train((validation,))
        with pytest.raises(TheoryReactorError, match="frozen validation"):
            index.verify_validation((train,))
    _make_writable(root)


def test_context_target_mutation_fails_disk_verification(tmp_path):
    root, _manifest, train, _validation = _make_index(tmp_path)
    mutated_targets = train.episodes.query.targets.clone()
    row = 0
    read = int(train.episodes.query_read_index[row])
    mutated_targets[row, read] = (
        int(mutated_targets[row, read]) + 1
    ) % 64
    mutated = replace(
        train,
        episodes=replace(
            train.episodes,
            query=replace(
                train.episodes.query,
                targets=mutated_targets,
            ),
        ),
    )
    with ETTRDiskPacketSufficiencyIndex(root) as index:
        with pytest.raises(TheoryReactorError, match="frozen train"):
            index.verify_train((mutated,))
    _make_writable(root)


def test_index_file_substitution_fails_load(tmp_path):
    root, _manifest, _train, _validation = _make_index(tmp_path)
    contexts = root / "contexts.bin"
    root.chmod(0o700)
    contexts.chmod(0o600)
    payload = bytearray(contexts.read_bytes())
    payload[-1] ^= 1
    contexts.write_bytes(payload)
    contexts.chmod(0o400)
    root.chmod(0o500)
    with pytest.raises(ETTRPacketIndexError, match="identity differs"):
        ETTRDiskPacketSufficiencyIndex(root)
    _make_writable(root)


def test_duplicate_batch_payload_is_rejected(tmp_path):
    train, validation = _batches()
    root = tmp_path / "duplicate-index"
    with pytest.raises(ETTRPacketIndexError, match="duplicate batch"):
        build_disk_packet_index(
            root,
            train_batches=(train, train),
            validation_batches=(validation,),
        )
    _make_writable(root)
