#!/usr/bin/env python3
"""Small filesystem test for the frozen DIVERGE-TOL1 board builder."""

from pathlib import Path
import tempfile

from build_diverge_tol1_board import build, sha256_path


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "tol1"
        report = build(output, train_count=32, development_count=16, ood_count=16)
        assert report["identity_overlaps"] == {
            "train_development": 0,
            "train_ood": 0,
            "development_ood": 0,
        }
        assert report["reserved_bigram_counts"]["train"] == {
            "GUARD->SWAP": 0,
            "SWAP->MULTIPLY": 0,
        }
        assert report["reserved_bigram_counts"]["ood"] == {
            "GUARD->SWAP": 16,
            "SWAP->MULTIPLY": 16,
        }
        for split, digest in report["hashes"].items():
            assert sha256_path(output / f"{split}.jsonl") == digest
    print("diverge TOL1 board builder tests passed")


if __name__ == "__main__":
    main()
