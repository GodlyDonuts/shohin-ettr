from __future__ import annotations

from pathlib import Path
import subprocess

SCRIPT = Path(__file__).parent / "jobs" / "q36_restore_nemotron_ultra.sbatch"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_ultra_restore_shell_is_valid() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_ultra_restore_is_bound_to_preserved_exact_snapshot() -> None:
    source = _source()
    expected = {
        "REPOSITORY=nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
        "REVISION=183968f87ae4cedce3039313cac1fd43d112c578",
        "EXPECTED_UPSTREAM_FILES=243",
        "EXPECTED_UPSTREAM_BYTES=352381245521",
        "EXPECTED_MANIFEST_ENTRIES=245",
        "EXPECTED_WEIGHT_FILES=113",
        "EXPECTED_NONWEIGHT_FILES=132",
        "EXPECTED_FINAL_FILES=246",
        "EXPECTED_CAPSULE_FILES=134",
        "EXPECTED_CAPSULE_MANIFEST_ENTRIES=133",
        "EXPECTED_CAPSULE_MANIFEST_SHA256=4f2b30d3c5090f81765b32d8afc51c9d6379999379a5247c8d9497fe9cb68641",
        "EXPECTED_MODEL_MANIFEST_SHA256=e20d8aad607dd1146ee83d49c31f65e7fc24033348137a57a93615436484751a",
        "EXPECTED_ORIGINAL_RECEIPT_SHA256=6b9b50ed6bdb644c0bf059c3c9f8cff2a2775d936fb8c6d1b65f0c953aba731a",
        "EXPECTED_SOURCE_REVISION_SHA256=72b9c31cfe862b4bb903d0ba7da4452df43875fdf32cd2c388ca836164338c8c",
    }
    for value in expected:
        assert value in source


def test_ultra_restore_fails_before_mutation_without_capsule_or_quota() -> None:
    source = _source()
    capsule_hash = source.index('verify_sha256 "$EVIDENCE_ROOT/SHA256SUMS.capsule"')
    capsule_replay = source.index("/usr/bin/sha256sum -c SHA256SUMS.capsule")
    manifest_projection = source.index(
        "Q36 Ultra preserved manifest projection differs"
    )
    quota = source.index("quota_raw=$(/usr/bin/lfs quota")
    quota_failure = source.index("Q36 Ultra durable-storage headroom is unsafe")
    stage = source.index('/usr/bin/mkdir -- "$STAGE_ROOT" "$RESTORE_STAGE"')
    assert capsule_hash < capsule_replay < manifest_projection < quota
    assert quota < quota_failure < stage
    assert "readonly REQUIRED_HEADROOM_BYTES=421100722257" in source
    assert "readonly REQUIRED_HEADROOM_INODES=20000" in source


def test_ultra_restore_downloads_only_weight_shards() -> None:
    source = _source()
    assert "allow_patterns=weights" in source
    assert 'value.rfilename.endswith(".safetensors")' in source
    assert "if len(weights) != expected_weights" in source
    assert '"nonweight_files_downloaded": 0' in source
    assert '"weight_files_downloaded": 113' in source
    # Preserved non-weight bytes are copied before the only network download.
    copy = source.index("shutil.copyfile(source / name, destination / name)")
    download = source.index("snapshot_download(")
    assert copy < download


def test_ultra_restore_replays_manifest_and_rejects_tree_extras() -> None:
    source = _source()
    # Capsule replay plus restored-tree replay before and after publication.
    assert source.count("/usr/bin/sha256sum -c SHA256SUMS") == 3
    assert 'observed != sorted([*names, "SHA256SUMS"])' in source
    assert "restored tree unexpectedly contains directories" in source
    assert '/usr/bin/chmod -R a-w "$STAGE_ROOT" "$RESTORE_STAGE"' in source
    assert '/usr/bin/mv -- "$STAGE_ROOT" "$FINAL_ROOT"' in source
    assert '/usr/bin/mv -- "$RESTORE_STAGE" "$RESTORE_ROOT"' in source


def test_ultra_restore_receipt_is_outside_model_tree_and_non_scientific() -> None:
    source = _source()
    assert '"$RESTORE_STAGE/restore_receipt.json" "$FINAL_ROOT"' in source
    assert '"schema": "shohin-q36-nemotron-ultra-exact-restore-v1"' in source
    assert '"byte_exact_original_tree": True' in source
    assert "#SBATCH --no-requeue" in source
    for forbidden in (
        "--gres=gpu",
        "score_completion",
        "assessor_root",
        "benchmark_rows",
    ):
        assert forbidden not in source.lower()
