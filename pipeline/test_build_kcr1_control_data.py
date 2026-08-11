import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

from build_kcr1_control_data import build


class FakeTokenizer:
    chat_template = None

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(range(max(1, len(str(text).split()))))


def test_builds_complete_matched_controls(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "aligned.jsonl"
    rows = []
    for index in range(4):
        identity = hashlib.sha256(str(index).encode()).hexdigest()
        target = f"first {index}. finish {index}"
        presentations = (
            ("verified_keep", target, "<KEEP>"),
            ("verified_continue", f"first {index}.", f"<CONTINUE>\n finish {index}"),
            ("natural_owner", "wrong", f"<RESTART>\n{target}"),
        )
        for presentation, draft, response in presentations:
            rows.append(
                {
                    "schema": "shohin-kcr1-branch-train-v1",
                    "source_identity_sha256": identity,
                    "training_group": "math",
                    "presentation": presentation,
                    "question": f"SOURCE:\nq\n\nDRAFT:\n{draft}",
                    "response": response,
                    "executed_target_sha256": hashlib.sha256(target.encode()).hexdigest(),
                }
            )
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema": "shohin-kcr1-branch-data-report-v1",
                "status": "complete",
                "zero_truncation": True,
                "holdout_used": False,
                "output": {"path": str(source.resolve()), "sha256": source_sha},
            }
        )
    )
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
    result = build(
        argparse.Namespace(
            source=source,
            source_report=report,
            model_root=tmp_path,
            max_sequence_length=4096,
            output=output,
        )
    )
    assert result["source_rows"] == 12
    assert result["source_groups"] == 4
    assert result["outputs"]["action_permuted"]["rows"] == 12
    assert result["outputs"]["constant_restart"]["rows"] == 12
    assert result["outputs"]["constant_restart"]["scan_counters"] == {
        "action_<RESTART>": 12
    }
    assert result["zero_truncation"] is True
