from __future__ import annotations

from pathlib import Path

SCRIPT = Path(__file__).parent / "jobs" / "q36_stage_nemotron_ultra.sbatch"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_ultra_snapshot_contract_is_exact() -> None:
    source = _source()
    expected = {
        "REPOSITORY=nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
        "REVISION=183968f87ae4cedce3039313cac1fd43d112c578",
        "EXPECTED_FILES=243",
        "EXPECTED_BYTES=352381245521",
        "EXPECTED_CONFIG_SHA256=0c939f324c8910f5ebdafbe2a56d7e4e074c50042a3b4f26326bf71a3fe33929",
        "EXPECTED_INDEX_SHA256=41951cc6db98b717cf45c73dd6f0503c050e99c70a9662abe01d9511bc2bed06",
        "EXPECTED_TOKENIZER_SHA256=623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7",
        "EXPECTED_QUANT_CONFIG_SHA256=68f0c66d927e5e9115ad475a7c1dbfe40f7e7e72d9ab9c3b6a33d8c52da2c596",
        "EXPECTED_SOURCE_REVISION_SHA256=72b9c31cfe862b4bb903d0ba7da4452df43875fdf32cd2c388ca836164338c8c",
    }
    for value in expected:
        assert value in source


def test_ultra_stage_fails_before_mutation_without_storage_margin() -> None:
    source = _source()
    quota = source.index("quota_raw=$(/usr/bin/lfs quota")
    quota_failure = source.index("Q36 Ultra durable-storage headroom is unsafe")
    temporary = source.index("cache_root=$(/usr/bin/mktemp")
    stage = source.index('/usr/bin/mkdir -- "$STAGE_ROOT"')
    assert quota < quota_failure < temporary < stage
    assert "readonly SAFETY_BYTES=68719476736" in source
    assert "readonly REQUIRED_HEADROOM_BYTES=421100722257" in source


def test_ultra_stage_is_write_once_hash_bound_and_non_scientific() -> None:
    source = _source()
    assert "#SBATCH --no-requeue" in source
    assert '[[ ! -e "$FINAL_ROOT" && ! -e "$STAGE_ROOT" ]]' in source
    assert "files_metadata=True" in source
    assert "sha256sum -c SHA256SUMS" in source
    assert "snapshot_receipt.json" in source
    receipt = source.index('path = root / "snapshot_receipt.json"')
    manifest = source.index("/usr/bin/find . -type f ! -name SHA256SUMS")
    assert receipt < manifest
    assert "== 245 ]]" in source
    assert '/usr/bin/chmod -R a-w "$STAGE_ROOT"' in source
    assert '/usr/bin/mv -- "$STAGE_ROOT" "$FINAL_ROOT"' in source
    for forbidden in (
        "--gres=gpu",
        "score_completion",
        "assessor_root",
        "benchmark_rows",
    ):
        assert forbidden not in source.lower()
