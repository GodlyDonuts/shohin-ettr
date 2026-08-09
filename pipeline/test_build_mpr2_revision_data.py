from build_mpr2_revision_data import donor_map


def test_donor_map_is_same_task_nearest_and_nonself():
    rows = [
        {"task": "math500", "source_id": "a", "draft_tokens": 10},
        {"task": "math500", "source_id": "b", "draft_tokens": 12},
        {"task": "math500", "source_id": "c", "draft_tokens": 30},
        {"task": "bbh_logic", "source_id": "d", "draft_tokens": 7},
        {"task": "bbh_logic", "source_id": "e", "draft_tokens": 8},
    ]
    assert donor_map(rows) == {"a": "b", "b": "a", "c": "b", "d": "e", "e": "d"}


def test_repeated_presentations_use_canonical_draft_lengths():
    canonical = {
        "a": {"draft_tokens": 10},
        "b": {"draft_tokens": 12},
    }
    presentation = {"source_id": "a"}
    donor = canonical["b"]
    assert abs(canonical[presentation["source_id"]]["draft_tokens"] - donor["draft_tokens"]) == 2
