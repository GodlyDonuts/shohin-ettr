import argparse
import json
from pathlib import Path
import sys
import types

from pipeline.build_ndr1_natural_revision_data import build, sha256_file


class FakeTokenizer:
    name_or_path = "fake"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(range(max(1, len(str(text).split()))))

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return "\n".join(message["content"] for message in messages)


def write_lines(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return sha256_file(path)


def test_ndr1_build_preserves_targets_and_shuffles_drafts(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    source_rows = [
        {
            "question": f"question {index}",
            "response": f"verified solution {index}",
            "training_group": "math" if index < 2 else "science",
        }
        for index in range(4)
    ]
    source_sha = write_lines(source, source_rows)
    source_report = tmp_path / "source.report.json"
    source_report.write_text(
        json.dumps(
            {
                "schema": "shohin-token-balanced-reasoning-mix-v1",
                "status": "complete",
                "output": str(source.resolve()),
                "output_sha256": source_sha,
                "max_sequence_length": 1536,
            }
        )
    )

    reports = []
    for shard in range(2):
        draft_path = tmp_path / f"draft_{shard}.jsonl"
        rows = []
        for index in range(shard * 2, shard * 2 + 2):
            group = "math" if index < 2 else "science"
            identity = __import__(
                "pipeline.build_ndr1_natural_revision_data", fromlist=["source_identity"]
            ).source_identity(source_rows[index])
            rows.append(
                {
                    "schema": "shohin-ndr1-natural-drafts-v1",
                    "source_identity_sha256": identity,
                    "source_index": index,
                    "training_group": group,
                    "completion": f"natural draft {index}",
                    "generated_tokens": 3,
                    "max_token_exhausted": False,
                }
            )
        draft_sha = write_lines(draft_path, rows)
        report_path = tmp_path / f"draft_{shard}.report.json"
        report_path.write_text(
            json.dumps(
                {
                    "schema": "shohin-ndr1-natural-draft-report-v1",
                    "status": "complete",
                    "source_sha256": source_sha,
                    "adapter_checkpoint_sha256": "adapter",
                    "shard_index": shard,
                    "shard_count": 2,
                    "max_new_tokens": 768,
                    "seed": 2026080919,
                    "output": str(draft_path.resolve()),
                    "output_sha256": draft_sha,
                }
            )
        )
        reports.append(report_path)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: FakeTokenizer()
            )
        ),
    )
    output = tmp_path / "output"
    report = build(
        argparse.Namespace(
            source=source,
            source_report=source_report,
            draft_report=reports,
            adapter_checkpoint_sha256="adapter",
            model_root=tmp_path,
            output=output,
        )
    )
    aligned = [json.loads(line) for line in (output / "train_aligned.jsonl").read_text().splitlines()]
    shuffled = [json.loads(line) for line in (output / "train_shuffled.jsonl").read_text().splitlines()]
    assert report["admitted_rows_per_arm"] == 4
    assert [row["response"] for row in aligned] == [row["response"] for row in shuffled]
    assert all(
        row["source_identity_sha256"] != row["draft_donor_identity_sha256"]
        for row in shuffled
    )
    assert report["synthetic_faults_used"] is False
    assert report["clean_copy_presentations_used"] is False
