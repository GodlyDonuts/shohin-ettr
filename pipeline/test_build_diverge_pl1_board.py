#!/usr/bin/env python3
"""Small materialization smoke for the DIVERGE-PL1 board builder."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    builder = root / "pipeline" / "build_diverge_pl1_board.py"
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "board"
        subprocess.run(
            [
                sys.executable,
                str(builder),
                "--output",
                str(output),
                "--train-count",
                "4",
                "--development-count",
                "3",
                "--confirmation-count",
                "2",
            ],
            cwd=root,
            check=True,
            env={**dict(__import__("os").environ), "PYTHONPATH": str(root / "train")},
            capture_output=True,
            text=True,
        )
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        assert report["model_score_used_for_selection"] is False
        assert not any(report["overlap"].values())
        assert report["split_reports"]["train"]["episodes"] == 4
        public = json.loads(
            (output / "development_public.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        assessor = json.loads(
            (output / "development_assessor.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        assert "symbol_to_operation" not in public
        assert "trace" not in public["acquisition"][0]
        assert "symbol_to_operation" in assessor
        assert "trace" in assessor["acquisition"][0]
    print("DIVERGE-PL1 board builder tests passed")


if __name__ == "__main__":
    main()

