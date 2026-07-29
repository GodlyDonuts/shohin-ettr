"""Static gates for the pinned Essential-Web reasoning-core candidate."""

import json
from pathlib import Path

from pipeline.verify_pinned_hf_manifest import validate_manifest


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "pipeline" / "jobs" / "essential_web_reasoning_core.sbatch"
MANIFEST = (
    ROOT
    / "pipeline"
    / "source_manifests"
    / "essential_web_2024_38_uniform256_ce4eccc.json"
)
DATASET = "EssentialAI/essential-web-v1.0"
REVISION = "ce4eccc7e9604667b6d7f32cb6274b8b41f3113d"


def test_manifest_is_pinned_deterministic_and_bounded():
    payload = json.loads(MANIFEST.read_text())
    records = validate_manifest(
        payload,
        expected_dataset=DATASET,
        expected_revision=REVISION,
    )
    assert len(records) == 256
    assert payload["selection"] == {
        "method": "sha256_priority_subset",
        "prefix": "data/crawl=CC-MAIN-2024-38",
        "suffix": ".parquet",
        "candidate_count": 3291,
        "selected_count": 256,
        "priority_material": "sha256(dataset\\x1frevision\\x1fpath)",
    }


def test_job_requires_exact_reasoning_quality_and_cleanliness_labels():
    script = JOB.read_text()
    for predicate in (
        "quality_signals.fasttext.english=0.90",
        "eai_taxonomy.reasoning_depth.primary.code=3",
        "eai_taxonomy.reasoning_depth.primary.code=4",
        "eai_taxonomy.reasoning_depth.primary.code=5",
        "eai_taxonomy.technical_correctness.primary.code=4",
        "eai_taxonomy.technical_correctness.primary.code=5",
        "eai_taxonomy.extraction_artifacts.primary.code=0",
        "eai_taxonomy.missing_content.primary.code=0",
    ):
        assert predicate in script
    for document_type in ("3", "8", "9", "10", "18", "20", "21", "23"):
        assert (
            "--required-allowed-value "
            f"eai_taxonomy.document_type_v2.primary.code={document_type}"
        ) in script


def test_job_is_candidate_only_with_physical_inputs_and_domain_cap():
    script = JOB.read_text()
    assert "--input-files" in script
    assert "--domain-field metadata.source_domain" in script
    assert "DOMAIN_TOKEN_CAP=${DOMAIN_TOKEN_CAP:-5000000}" in script
    assert "MAX_TOKENS=${MAX_TOKENS:-1000000000}" in script
    assert "not training-admitted" in script
    assert "train.py" not in script
