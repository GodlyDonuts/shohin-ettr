from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_dense_public_ruler_data import run


def test_builds_thirteen_task_ruler_screen(tmp_path: Path) -> None:
    paths = []
    for index in range(13):
        path = tmp_path / "4096" / f"task_{index}" / "validation.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps({"index": 0, "input": f"prompt {index}", "outputs": ["x"]})
            + "\n"
        )
        paths.append(path)
    args = argparse.Namespace(
        ruler_jsonl=paths,
        ruler_commit="c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a",
        output_root=tmp_path / "output",
    )
    report = run(args)
    assert report["benchmarks"]["ruler"]["rows"] == 13
    assert report["assessors_visible_to_model"] is False
