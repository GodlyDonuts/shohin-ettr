import json
from pathlib import Path

import pytest

from pipeline.shard_taco_audit_candidates import TacoShardError
from pipeline.shard_taco_audit_candidates import merge_verified
from pipeline.shard_taco_audit_candidates import split_candidates


def _write_candidates(path: Path, count: int = 7) -> Path:
    path.write_text(
        "".join(
            json.dumps({"problem_id": index, "response": f"print({index})"}) + "\n"
            for index in range(count)
        )
    )
    return path


def _verified(row: dict) -> dict:
    return {
        **row,
        "full_verified_cases": 2,
        "training_group": "code",
        "verification": "execution_verified",
    }


def test_split_and_merge_preserve_candidate_order(tmp_path: Path) -> None:
    source = _write_candidates(tmp_path / "input.jsonl")
    manifest_path = tmp_path / "manifest.json"
    manifest = split_candidates(source, tmp_path / "shards", manifest_path, 3)
    verified_paths = []
    for record in manifest["shards"]:
        shard_rows = [json.loads(line) for line in Path(record["path"]).read_text().splitlines()]
        accepted = [_verified(row) for row in shard_rows if int(row["problem_id"]) != 4]
        path = tmp_path / f"verified-{record['index']}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in accepted))
        verified_paths.append(path)

    report = merge_verified(
        input_path=source,
        manifest_path=manifest_path,
        verified_paths=verified_paths,
        output_path=tmp_path / "merged.jsonl",
        report_path=tmp_path / "report.json",
    )

    merged = [json.loads(line) for line in (tmp_path / "merged.jsonl").read_text().splitlines()]
    assert [row["problem_id"] for row in merged] == [0, 1, 2, 3, 5, 6]
    assert report["candidate_rows"] == 7
    assert report["verified_rows"] == 6
    assert report["dropped_rows"] == 1


def test_merge_rejects_response_drift(tmp_path: Path) -> None:
    source = _write_candidates(tmp_path / "input.jsonl", 2)
    manifest_path = tmp_path / "manifest.json"
    manifest = split_candidates(source, tmp_path / "shards", manifest_path, 2)
    verified_paths = []
    for record in manifest["shards"]:
        row = json.loads(Path(record["path"]).read_text())
        if row["problem_id"] == 1:
            row["response"] = "changed"
        path = tmp_path / f"verified-{record['index']}.jsonl"
        path.write_text(json.dumps(_verified(row)) + "\n")
        verified_paths.append(path)

    with pytest.raises(TacoShardError, match="response differs"):
        merge_verified(
            input_path=source,
            manifest_path=manifest_path,
            verified_paths=verified_paths,
            output_path=tmp_path / "merged.jsonl",
            report_path=tmp_path / "report.json",
        )
