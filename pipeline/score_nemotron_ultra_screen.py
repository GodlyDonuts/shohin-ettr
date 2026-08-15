#!/usr/bin/env python3
"""Score the matched zero-label-transfer Nemotron Ultra screen once."""

from __future__ import annotations

import json

from score_nemotron_super_screen import parse_args, run

CANDIDATE_SCHEMA = "shohin-nemotron-ultra-fixed-draft-candidate-v1"
REPORT_SCHEMA = "shohin-nemotron-ultra-fixed-draft-screen-score-v1"
HOST = "NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
TOTAL_PARAMETERS = 550_000_000_000
ACTIVE_PARAMETERS = 55_000_000_000


if __name__ == "__main__":
    result = run(
        parse_args(),
        candidate_schema=CANDIDATE_SCHEMA,
        report_schema=REPORT_SCHEMA,
        host=HOST,
        total_parameters=TOTAL_PARAMETERS,
        active_parameters=ACTIVE_PARAMETERS,
    )
    print(
        json.dumps(
            {arm: result["arms"][arm]["correct"] for arm in result["arms"]},
            sort_keys=True,
        )
    )
