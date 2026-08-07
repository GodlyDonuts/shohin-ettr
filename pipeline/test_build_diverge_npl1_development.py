#!/usr/bin/env python3
"""Small self-contained smoke for the NPL1 development-only builder."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from diverge_pl1_data import DEVELOPMENT_SEED, build_split


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    builder = root / "pipeline" / "build_diverge_npl1_development.py"
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        prior_path = directory / "prior.jsonl"
        prior = build_split(split="development", seed=DEVELOPMENT_SEED, count=3)
        prior_path.write_text(
            "".join(
                json.dumps(episode.assessor_record(), sort_keys=True) + "\n"
                for episode in prior
            ),
            encoding="utf-8",
        )
        output = directory / "board"
        subprocess.run(
            [
                sys.executable,
                str(builder),
                "--output",
                str(output),
                "--prior-assessor",
                str(prior_path),
                "--development-count",
                "4",
            ],
            cwd=root,
            check=True,
            env={**os.environ, "PYTHONPATH": str(root / "train")},
            capture_output=True,
            text=True,
        )
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        assert report["episodes"] == 4
        assert report["confirmation_generated"] is False
        assert report["model_score_used_for_selection"] is False
        assert not any(report["overlap"].values())
        public = json.loads(
            (output / "development_public.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        assessor = json.loads(
            (output / "development_assessor.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        assert "oracle" not in public
        assert "oracle" in assessor
    print("DIVERGE-NPL1 development builder tests passed")


if __name__ == "__main__":
    main()
