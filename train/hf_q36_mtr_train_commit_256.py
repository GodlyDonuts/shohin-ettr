#!/usr/bin/env python3
"""Train the Q36 semantic commit for 256 engineering updates."""

from __future__ import annotations

import json

import hf_q36_mtr_train_commit as implementation

ENGINEERING_UPDATES = 256


def main() -> int:
    implementation.UPDATES = ENGINEERING_UPDATES
    args = implementation.parse_args()
    if args.updates != ENGINEERING_UPDATES:
        raise implementation.Q36MTRCommitError(
            "Q36 engineering commit update count differs"
        )
    report = implementation.train(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "updates": report["updates"],
                "pair_presentations": report["pair_presentations"],
                "checkpoint_sha256": report["checkpoint_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
