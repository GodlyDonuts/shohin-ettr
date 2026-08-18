from pathlib import Path

from hf_gpt_oss_120b_evaluate import (
    ARMS,
    CANDIDATE_SCHEMA,
    MAX_NEW_TOKENS,
    REPORT_SCHEMA,
    ROWS,
    SHARDS,
)


def test_matched_screen_geometry_is_frozen() -> None:
    assert ARMS == ("unchanged", "self_refinement", "revision")
    assert ROWS == 256
    assert SHARDS == 4
    assert MAX_NEW_TOKENS == 768
    assert CANDIDATE_SCHEMA == "shohin-gpt-oss-120b-fixed-draft-candidate-v1"
    assert REPORT_SCHEMA == "shohin-gpt-oss-120b-fixed-draft-evaluation-v1"


def test_evaluator_projects_harmony_final_and_keeps_assessors_closed() -> None:
    source = Path(__file__).with_name("hf_gpt_oss_120b_evaluate.py").read_text()
    assert "extract_final_completion" in source
    assert 'generation_projection": "final_channel_only"' in source
    assert 'harmony_reasoning_effort": "low"' in source
    assert '"assessor_access_count": 0' in source
    assert '"development_labels_read": 0' in source
    assert '"sealed_access": {"holdout": 0, "product": 0, "public": 0}' in source


def test_each_evaluation_loads_one_native_mxfp4_host() -> None:
    source = Path(__file__).with_name("hf_gpt_oss_120b_evaluate.py").read_text()
    assert "_load_backbone(model_root)" in source
    assert "torch.cuda.device_count() != 1" in source
    assert "GptOssRevisionModel(backbone)" in source
