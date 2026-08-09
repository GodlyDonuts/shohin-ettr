import torch

from mtr1_router_accounting import summarize_router_logits


def test_router_summary_excludes_padding_and_counts_topk() -> None:
    logits = torch.tensor(
        [
            [9.0, 8.0, 0.0, 0.0],
            [0.0, 9.0, 8.0, 0.0],
            [0.0, 0.0, 9.0, 8.0],
            [9.0, 0.0, 0.0, 8.0],
        ]
    )
    mask = torch.tensor([[0, 1], [1, 1]])
    summary = summarize_router_logits((logits,), mask, top_k=2)[0]
    assert summary["tokens"] == 3
    assert summary["assignments"] == 6
    assert summary["expert_counts"] == [1, 1, 2, 2]
    assert summary["active_experts"] == 4
    assert abs(sum(summary["expert_weight_share"]) - 1.0) < 1e-6


def test_router_summary_rejects_empty_mask() -> None:
    try:
        summarize_router_logits((torch.zeros(2, 4),), torch.zeros(1, 2), top_k=2)
    except RuntimeError as error:
        assert "empty" in str(error)
    else:
        raise AssertionError("empty accounting must fail")


def test_router_summary_respects_unnormalized_topk_weights() -> None:
    logits = torch.tensor([[10.0, 9.0, -10.0], [0.0, 1.0, 0.0]])
    mask = torch.ones(1, 2)
    raw = summarize_router_logits((logits,), mask, top_k=1, normalize_topk=False)[0]
    normalized = summarize_router_logits(
        (logits,), mask, top_k=1, normalize_topk=True
    )[0]
    assert raw["expert_weight_share"][0] != normalized["expert_weight_share"][0]
