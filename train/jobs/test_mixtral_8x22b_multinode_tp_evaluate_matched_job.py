from pathlib import Path

SCRIPT = Path(__file__).with_name("mixtral_8x22b_multinode_tp_evaluate_matched.sbatch")
SUBMITTER = Path(__file__).with_name("submit_mixtral_8x22b_tp4_validation_groups.sh")


def test_postcondition_uses_two_digit_shard_names() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "seq -w" not in source
    assert "printf -v shard '%02d'" in source
    assert "shard_$shard/candidates.jsonl" in source
    assert "shard_$shard/report.json" in source


def test_validation_submitter_exports_only_each_groups_four_drafts() -> None:
    source = SUBMITTER.read_text(encoding="utf-8")
    assert 'first_draft=$((group * 4))' in source
    assert 'group_drafts=$(IFS=:; echo "${draft_paths[*]:first_draft:4}")' in source
    assert '[[ ${#selected_drafts[@]} -eq 4 ]]' in source
    assert '"${selected_drafts[$offset]}" == "${draft_paths[$((first_draft + offset))]}"' in source
    assert 'DRAFT_CANDIDATES=$group_drafts' in source
    assert 'DRAFT_CANDIDATES=$DRAFT_CANDIDATES' not in source
