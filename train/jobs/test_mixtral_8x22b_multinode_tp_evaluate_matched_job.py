from pathlib import Path

SCRIPT = Path(__file__).with_name("mixtral_8x22b_multinode_tp_evaluate_matched.sbatch")


def test_postcondition_uses_two_digit_shard_names() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "seq -w" not in source
    assert "printf -v shard '%02d'" in source
    assert "shard_$shard/candidates.jsonl" in source
    assert "shard_$shard/report.json" in source
