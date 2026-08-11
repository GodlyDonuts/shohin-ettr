import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

from build_kcr1_development_canary import build


class FakeTokenizer:
    chat_template = None

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(range(max(1, len(str(text).split()))))


def write_lines(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_builds_three_state_source_disjoint_canary(tmp_path: Path, monkeypatch) -> None:
    development = tmp_path / "development.jsonl"
    rows = []
    for index in range(1289):
        identity = hashlib.sha256(str(index).encode()).hexdigest()
        completion = f"Reason {index}. Then finish. \\boxed{{{index}}}"
        rows.append(
            {
                "schema": "shohin-idr1-revision-eval-v1",
                "identity_sha256": identity,
                "split": "development",
                "task": "math500" if index else "mbpp",
                "question": "prompt",
                "runtime_fields": ["question"],
                "internal_draft": {
                    "identity_sha256": identity,
                    "completion": "wrong",
                    "correct": False,
                    "max_token_exhausted": True,
                },
                "candidates": [
                    {"lineage": "base", "completion": completion, "correct": True},
                    {"lineage": "expert", "completion": "bad", "correct": False},
                ],
                "assessor": {
                    "task": "mbpp" if index == 0 else "math500",
                    "text": "write f" if index == 0 else None,
                    "test_list": ["assert f() == 1"] if index == 0 else None,
                    "question": f"question {index}",
                },
            }
        )
    development_sha = write_lines(development, rows)
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema": "shohin-idr1-revision-data-report-v1",
                "status": "complete",
                "outputs": {
                    "development": {
                        "path": str(development.resolve()),
                        "sha256": development_sha,
                        "rows": 1289,
                    }
                },
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
            development=development,
            development_report=report,
            model_root=tmp_path,
            max_sequence_length=4096,
            output=output,
        )
    )
    assert result["admitted_sources"] == 1289
    assert result["presentations"] == 3867
    assert result["scan_counters"]["action_<KEEP>"] == 1289
    assert result["scan_counters"]["action_<CONTINUE>"] == 1289
    assert result["scan_counters"]["action_<RESTART>"] == 1289
    assert result["zero_truncation"] is True
