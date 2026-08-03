from pipeline.freeze_unique_eval_board import freeze_rows


def test_freeze_rows_keeps_lowest_id_and_source_order() -> None:
    rows = [
        {"task_id": 9, "text": "duplicate", "test_list": ["second"]},
        {"task_id": 2, "text": "unique", "test_list": ["only"]},
        {"task_id": 3, "text": "duplicate", "test_list": ["first"]},
    ]

    frozen, groups = freeze_rows(rows, "text", "task_id")

    assert [row["task_id"] for row in frozen] == [2, 3]
    assert len(groups) == 1
    assert groups[0]["kept_id"] == 3
    assert groups[0]["dropped_ids"] == [9]


def test_freeze_rows_preserves_distinct_prompts() -> None:
    rows = [
        {"task_id": 1, "text": "left"},
        {"task_id": 2, "text": "right"},
    ]

    frozen, groups = freeze_rows(rows, "text", "task_id")

    assert frozen == rows
    assert groups == []
