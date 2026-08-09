from hf_vfr1_evaluate_reviser import summarize


def _rows() -> list[dict[str, object]]:
    rows = []
    for task, count in (("math500", 223), ("bbh_logic", 349), ("mbpp", 31)):
        for index in range(count):
            rows.append(
                {
                    "identity_sha256": f"{len(rows):064x}",
                    "task": task,
                    "outcome_class": "both_wrong",
                    "assessor": {},
                    "candidates": [
                        {"lineage": "base", "correct": False},
                        {"lineage": "expert", "correct": False},
                    ],
                }
            )
    return rows


def test_absolute_gate_passes_at_exact_floors() -> None:
    rows = _rows()
    results = [
        {"identity_sha256": row["identity_sha256"], "correct": True, "parse_error": None}
        for row in rows
    ]
    result = summarize(rows, results)
    assert len(rows) == 603
    assert result["absolute_gate_pass"] is True
