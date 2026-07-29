import json
from pathlib import Path

import pytest

from pipeline.profile_tokenized_corpus_domains import (
    DomainProfileError,
    profile_domains,
)
from pipeline.test_materialize_domain_balanced_residual import _make_source
from pipeline.tokenize_shards import canonical_payload_sha256


def test_profile_binds_domain_concentration_and_cap_projection(
    tmp_path: Path,
) -> None:
    source, selection = _make_source(tmp_path)
    output = tmp_path / "profile.json"
    profile = profile_domains(
        source_dir=source,
        source_selection_code=selection,
        output_path=output,
        cap_projections=(3,),
    )
    assert profile["documents"] == 4
    assert profile["tokens"] == 8
    assert profile["domains"] == 3
    assert profile["missing_domain_tokens"] == 2
    assert profile["concentration"]["top_1_token_fraction"] == 0.5
    projection = profile["cap_projections"][0]
    assert projection[
        "projected_tokens_after_cap_and_missing_domain_rejection"
    ] == 5
    assert profile["payload_sha256"] == canonical_payload_sha256(
        {key: value for key, value in profile.items() if key != "payload_sha256"}
    )
    assert json.loads(output.read_text()) == profile


def test_profile_refuses_existing_output(tmp_path: Path) -> None:
    source, selection = _make_source(tmp_path)
    output = tmp_path / "profile.json"
    output.write_text("existing")
    with pytest.raises(DomainProfileError, match="arguments"):
        profile_domains(
            source_dir=source,
            source_selection_code=selection,
            output_path=output,
        )


def test_profile_rejects_duplicate_projection_caps(tmp_path: Path) -> None:
    source, selection = _make_source(tmp_path)
    with pytest.raises(DomainProfileError, match="projection"):
        profile_domains(
            source_dir=source,
            source_selection_code=selection,
            output_path=tmp_path / "profile.json",
            cap_projections=(3, 3),
        )
