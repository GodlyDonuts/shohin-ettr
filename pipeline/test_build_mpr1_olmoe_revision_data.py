from build_mpr1_olmoe_revision_data import MPR1DataError, donor_map, split_draft


def test_split_draft_roundtrip():
    prompt = (
        "prefix\n\nInternal draft:\na draft\n\nReturn a complete corrected solution "
        "with the exact final answer in \\boxed{}.\n\nOriginal problem:\nsource"
    )
    prefix, draft, suffix = split_draft(prompt)
    assert prefix + draft + suffix == prompt
    assert draft == "a draft"


def test_split_draft_rejects_ambiguous_marker():
    try:
        split_draft("no markers")
    except MPR1DataError:
        pass
    else:
        raise AssertionError("missing markers accepted")


def test_donor_map_is_same_task_and_nonself_nearest():
    rows = [
        {"task": "math500", "source_id": "a", "draft_tokens": 10},
        {"task": "math500", "source_id": "b", "draft_tokens": 12},
        {"task": "math500", "source_id": "c", "draft_tokens": 30},
        {"task": "bbh_logic", "source_id": "d", "draft_tokens": 8},
        {"task": "bbh_logic", "source_id": "e", "draft_tokens": 9},
    ]
    donors = donor_map(rows)
    assert donors["a"] == "b"
    assert donors["b"] == "a"
    assert donors["c"] == "b"
    assert donors["d"] == "e" and donors["e"] == "d"


def test_donor_map_uses_canonical_sources_not_presentations():
    rows = [
        {"task": "math500", "source_id": "a", "draft_tokens": 10},
        {"task": "math500", "source_id": "b", "draft_tokens": 11},
    ]
    assert donor_map(rows) == {"a": "b", "b": "a"}
