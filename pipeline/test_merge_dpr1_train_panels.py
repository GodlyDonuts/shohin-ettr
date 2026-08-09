from hf_mpr2_generate_drafts import canonical_sources


def test_dpr1_train_panel_source_canonicalization():
    rows=[{"source_identity_sha256":"a"*64,"question":"q"},{"source_identity_sha256":"a"*64,"question":"q"}]
    assert len(canonical_sources(rows))==1

