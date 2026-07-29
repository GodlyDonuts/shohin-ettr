import json

import pytest
import torch

from model import GPT, GPTConfig
from pipeline.test_materialize_v3_holdout_split import (
    SEED,
    SELECTION_CODE,
    _build_source,
)
from pipeline.materialize_v3_holdout_split import materialize_holdout_split
from pipeline.tokenize_shards import sha256_file
from eval_corpus_nll import (
    CorpusNllError,
    evaluate_corpus_nll,
    selected_window_indices,
)
from assess_paired_corpus_nll import assess_paired_nll


def test_midpoint_windows_are_unique_and_spread():
    assert selected_window_indices(10, 4) == [1, 3, 6, 8]
    assert selected_window_indices(4, 4) == [0, 1, 2, 3]
    with pytest.raises(CorpusNllError):
        selected_window_indices(3, 4)


def test_fixed_corpus_nll_is_hash_bound_and_excludes_zloss(tmp_path):
    source, source_selection, _rows = _build_source(tmp_path)
    split = tmp_path / "split"
    materialize_holdout_split(
        source_dir=source,
        source_selection_code=source_selection,
        selection_code=SELECTION_CODE,
        output_dir=split,
        seed=SEED,
        document_validation_bps=2_500,
        domain_validation_bps=2_500,
        shard_tokens=3,
    )
    config = GPTConfig(
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=1,
        d_model=16,
        d_ff=32,
        seq_len=3,
        zloss=0.5,
    )
    torch.manual_seed(7)
    model = GPT(config)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {"model": model.state_dict(), "cfg": config.__dict__, "step": 0},
        checkpoint,
    )
    output = tmp_path / "nll.json"
    report = evaluate_corpus_nll(
        checkpoint=checkpoint,
        checkpoint_sha256=sha256_file(checkpoint),
        corpus_dir=split / "train",
        selection_code=SELECTION_CODE,
        output=output,
        max_target_tokens=6,
        batch_size=2,
        device="cpu",
    )
    assert report["metric"]["training_zloss_excluded"] is True
    assert report["sampling"]["target_tokens"] == 6
    assert report["metric"]["mean_nll"] > 0
    assert len(report["metric"]["window_mean_nll"]) == 2
    assert json.loads(output.read_text())["payload_sha256"] == report[
        "payload_sha256"
    ]


def test_checkpoint_substitution_fails_before_report(tmp_path):
    source, source_selection, _rows = _build_source(tmp_path)
    split = tmp_path / "split"
    materialize_holdout_split(
        source_dir=source,
        source_selection_code=source_selection,
        selection_code=SELECTION_CODE,
        output_dir=split,
        seed=SEED,
        document_validation_bps=2_500,
        domain_validation_bps=2_500,
        shard_tokens=3,
    )
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"not": "a model"}, checkpoint)
    output = tmp_path / "nll.json"
    with pytest.raises(CorpusNllError, match="SHA-256 differs"):
        evaluate_corpus_nll(
            checkpoint=checkpoint,
            checkpoint_sha256="0" * 64,
            corpus_dir=split / "train",
            selection_code=SELECTION_CODE,
            output=output,
            max_target_tokens=3,
            batch_size=1,
            device="cpu",
        )
    assert not output.exists()


def test_paired_nll_detects_an_identical_checkpoint_tie(tmp_path):
    source, source_selection, _rows = _build_source(tmp_path)
    split = tmp_path / "split"
    materialize_holdout_split(
        source_dir=source,
        source_selection_code=source_selection,
        selection_code=SELECTION_CODE,
        output_dir=split,
        seed=SEED,
        document_validation_bps=2_500,
        domain_validation_bps=2_500,
        shard_tokens=3,
    )
    config = GPTConfig(
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_kv_head=1,
        d_model=16,
        d_ff=32,
        seq_len=3,
    )
    model = GPT(config)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {"model": model.state_dict(), "cfg": config.__dict__, "step": 0},
        checkpoint,
    )
    reports = []
    for name in ("baseline", "candidate"):
        report_path = tmp_path / f"{name}.json"
        evaluate_corpus_nll(
            checkpoint=checkpoint,
            checkpoint_sha256=sha256_file(checkpoint),
            corpus_dir=split / "train",
            selection_code=SELECTION_CODE,
            output=report_path,
            max_target_tokens=6,
            batch_size=2,
            device="cpu",
        )
        reports.append(report_path)
    assessment = assess_paired_nll(
        baseline_path=reports[0],
        candidate_path=reports[1],
        output=tmp_path / "paired.json",
    )
    assert assessment["statistics"]["mean_delta_nll"] == 0
    assert not assessment["gate"][
        "strict_improvement_upper_95pct_ci_below_zero"
    ]
