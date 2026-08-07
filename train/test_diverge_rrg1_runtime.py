from __future__ import annotations

import torch

from diverge_iem1_runtime import tensorize_queries
from diverge_rrg1_runtime import (
    RRG1Config,
    RelationalReferentMachine,
    RelationalRoleGrounder,
    permutation_scores,
    permutation_targets,
    validate_owner_contract,
)


def _record(text: str, symbols: list[str]) -> dict[str, object]:
    return {
        "source_text": text,
        "symbols": symbols,
        "symbol_role_ids": [0, 1],
    }


def test_rrg1_name_bytes_and_lengths_are_unobservable() -> None:
    torch.manual_seed(9)
    model = RelationalRoleGrounder(RRG1Config()).eval()
    base = _record(
        "Report register ash; ignore decoy register cinder.",
        ["ash", "cinder", "elm", "fir", "grove"],
    )
    renamed = _record(
        "Report register extraordinarilylongreferent; ignore decoy register ox.",
        ["extraordinarilylongreferent", "ox", "elm", "fir", "grove"],
    )
    ids, mask, symbols, _ = tensorize_queries([base, renamed], torch.device("cpu"))
    embedded, compact_mask, compact_groups = model.canonicalize(ids, mask, symbols)
    assert torch.equal(embedded[0], embedded[1])
    assert torch.equal(compact_mask[0], compact_mask[1])
    assert torch.equal(compact_groups[0], compact_groups[1])
    logits = model(ids, mask, symbols)
    assert torch.equal(logits[0], logits[1])


def test_rrg1_group_and_role_slot_equivariance() -> None:
    torch.manual_seed(10)
    model = RelationalRoleGrounder(RRG1Config()).eval()
    record = _record(
        "Report register alpha; ignore decoy register beta.",
        ["alpha", "beta", "gamma", "delta", "epsilon"],
    )
    ids, mask, symbols, targets = tensorize_queries([record], torch.device("cpu"))
    normal = model(ids, mask, symbols)
    group_swapped = model(ids, mask, symbols.flip(1))
    slot_swapped = model(ids, mask, symbols, control="role_slot_swap")
    deleted = model(ids, mask, symbols, control="marker_delete")
    assert torch.equal(group_swapped, normal.flip(1))
    assert torch.equal(slot_swapped, normal.flip(2))
    assert not torch.equal(deleted, normal)
    assert permutation_scores(normal).shape == (1, 2)
    assert permutation_targets(targets).tolist() == [0]


def test_rrg1_owner_storage_and_protection() -> None:
    model = RelationalReferentMachine(RRG1Config())
    model.freeze_qualified_owners()
    validate_owner_contract(model)
    hashes = model.owner_hashes()
    assert hashes["WORLD"]
    assert hashes["NUMERIC_EVIDENCE"]
    assert hashes["QUERY"] == hashes["REFERENT"]


def main() -> None:
    test_rrg1_name_bytes_and_lengths_are_unobservable()
    test_rrg1_group_and_role_slot_equivariance()
    test_rrg1_owner_storage_and_protection()
    print("DIVERGE-RRG1 runtime tests passed")


if __name__ == "__main__":
    main()
