from hf_mpr2_generate_drafts import MPR2DraftError, canonical_sources


def test_canonical_sources_deduplicates_identical_presentations():
    rows = [
        {"source_identity_sha256": "a" * 64, "question": "same"},
        {"source_identity_sha256": "a" * 64, "question": "same"},
        {"source_identity_sha256": "b" * 64, "question": "other"},
    ]
    assert [row["source_identity_sha256"] for row in canonical_sources(rows)] == [
        "a" * 64,
        "b" * 64,
    ]


def test_canonical_sources_rejects_prompt_drift():
    rows = [
        {"source_identity_sha256": "a" * 64, "question": "one"},
        {"source_identity_sha256": "a" * 64, "question": "two"},
    ]
    try:
        canonical_sources(rows)
    except MPR2DraftError:
        pass
    else:
        raise AssertionError("prompt drift accepted")

