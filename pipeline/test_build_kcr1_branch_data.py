import argparse
import json
from pathlib import Path
import sys
import types

from build_kcr1_branch_data import build
from build_ndr1_natural_revision_data import sha256_file, source_identity
from kcr1_branch_transducer import execute_transaction


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(range(max(1, len(str(text).split()))))

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return "\n".join(message["content"] for message in messages)


def _write_lines(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return sha256_file(path)


def test_builds_three_exact_source_local_branches(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.jsonl"
    sources = [
        {
            "question": f"question {index}",
            "response": f"First derive value {index}. Then finish carefully. \\boxed{{{index}}}",
            "training_group": "math",
            "expected_answer_normalized": str(index),
        }
        for index in range(4)
    ]
    source_sha = _write_lines(source, sources)
    source_report = tmp_path / "source.report.json"
    source_report.write_text(
        json.dumps(
            {
                "schema": "shohin-token-balanced-reasoning-mix-v1",
                "status": "complete",
                "output_sha256": source_sha,
                "max_sequence_length": 1536,
            }
        ),
        encoding="utf-8",
    )
    draft_reports = []
    for shard in range(2):
        draft_path = tmp_path / f"draft_{shard}.jsonl"
        draft_rows = []
        for index in range(shard * 2, shard * 2 + 2):
            completion = (
                f"Reasoning. Final answer: \\boxed{{{index}}}"
                if index == 0
                else "unfinished work"
            )
            draft_rows.append(
                {
                    "schema": "shohin-ndr1-natural-drafts-v1",
                    "source_identity_sha256": source_identity(sources[index]),
                    "source_index": index,
                    "training_group": "math",
                    "completion": completion,
                    "generated_tokens": 8,
                    "max_token_exhausted": index == 1,
                }
            )
        draft_sha = _write_lines(draft_path, draft_rows)
        report = tmp_path / f"draft_{shard}.report.json"
        report.write_text(
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
            ),
            encoding="utf-8",
        )
        draft_reports.append(report)

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
            draft_report=draft_reports,
            adapter_checkpoint_sha256="adapter",
            model_root=tmp_path,
            max_sequence_length=4096,
            output=output,
        )
    )
    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert report["admitted_sources"] == 4
    assert report["presentations"] == 12
    assert report["transaction_roundtrip_rows"] == 12
    assert report["scan_counters"]["natural_action_<KEEP>"] == 1
    assert report["scan_counters"]["natural_action_<RESTART>"] == 3
    source_by_identity = {source_identity(row): row for row in sources}
    for row in rows:
        prompt = row["question"]
        draft = prompt.split("DRAFT:\n", 1)[1]
        expected = (
            draft
            if row["action"] == "<KEEP>"
            else source_by_identity[row["source_identity_sha256"]]["response"]
        )
        assert execute_transaction(draft, row["response"]) == expected
