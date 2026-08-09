import json

from train_dset1_span_edit import DATA_REPORT_SCHEMA, DATA_SCHEMA, load_pairs


def test_load_pairs_requires_clean_and_fault(tmp_path) -> None:
    data = tmp_path / "train.jsonl"
    rows = [
        {"schema": DATA_SCHEMA, "pair_identity_sha256": "p", "pair_member": "clean", "source_identity_sha256": "s"},
        {"schema": DATA_SCHEMA, "pair_identity_sha256": "p", "pair_member": "fault", "source_identity_sha256": "s"},
    ]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    import hashlib

    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema": DATA_REPORT_SCHEMA,
                "status": "complete",
                "holdout_used": False,
                "complete_retention": True,
                "train_diagnostic_source_overlap": 0,
                "max_script_tokens": 32,
                "outputs": {
                    "train": {
                        "path": str(data.resolve()),
                        "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                        "sources": 1,
                    }
                },
            }
        )
    )
    pairs, _ = load_pairs(data, report)
    assert len(pairs) == 1
