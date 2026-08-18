#!/usr/bin/env python3
"""Score the matched fixed-draft GPT-OSS-120B screen once."""

from __future__ import annotations

import json

from score_nemotron_super_screen import parse_args, run

CANDIDATE_SCHEMA = "shohin-gpt-oss-120b-fixed-draft-candidate-v1"
REPORT_SCHEMA = "shohin-gpt-oss-120b-fixed-draft-screen-score-v1"
HOST = "openai/gpt-oss-120b"
TOTAL_PARAMETERS = 117_000_000_000
ACTIVE_PARAMETERS = 5_100_000_000


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
