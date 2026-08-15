"""Static checks for the first 141B-A39B scientific fit."""

from pathlib import Path

SCRIPT = Path(__file__).with_name("mixtral_8x22b_train_revision.sbatch")


def test_mixtral_revision_fit_is_two_h100_nonrequeueing_and_dependency_ready() -> None:
    source = SCRIPT.read_text()
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:2" in source
    assert "#SBATCH --no-requeue" in source
    assert "#SBATCH --time=06:00:00" in source
    assert "hf_mixtral_8x22b_train_revision.py" in source
    assert '"${MECHANICS_REPORT:?MECHANICS_REPORT is required}"' in source
    assert '"${DATA:?DATA is required}"' in source
    assert (
        '"${EXPECTED_MODEL_MANIFEST_SHA256:?EXPECTED_MODEL_MANIFEST_SHA256 is required}"'
        in source
    )
    assert "--mechanics-report" in source
    assert "--data" in source
    assert "TRANSFORMERS_OFFLINE=1" in source
    assert "q36_init_local_tmp" in source
    assert "trap q36_cleanup_local_tmp EXIT" in source
    worker = (
        SCRIPT.parents[1].joinpath("hf_mixtral_8x22b_train_revision.py").read_text()
    )
    assert 'bnb_4bit_quant_type="nf4"' in worker
    assert "bnb_4bit_use_double_quant=True" in worker
    assert "trust_remote_code=False" in worker
    assert "set(device_map.values()) != {0, 1}" in worker
    assert "MODEL_MANIFEST_SHA256" not in worker


def test_mixtral_revision_fit_freezes_trainable_only_outputs() -> None:
    source = SCRIPT.read_text()
    assert '[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]]' in source
    assert 'checkpoint="$OUTPUT/checkpoint_0000256.pt"' in source
    assert 'report="$OUTPUT/report.json"' in source
    assert 'chmod a-w "$checkpoint" "$report" "$OUTPUT"' in source
