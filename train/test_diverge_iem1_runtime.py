#!/usr/bin/env python3
"""Focused shared-interface tests for DIVERGE-IEM1."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import torch

from diverge_iem1_data import generate_query_training_records
from diverge_iem1_runtime import (
    IEM1Config,
    IEM1RuntimeError,
    IntegratedEpistemicMachine,
    NaturalQueryReceipt,
    canonical_sha256,
    compile_query_batch,
    load_nve1_state,
    mutate_query_receipt,
    seal_natural_query,
    tensorize_local_texts,
    tensorize_queries,
)
from diverge_nve1_data import generate_training_records
from diverge_nve1_runtime import (
    EvidenceCompilerConfig,
    NaturalEvidenceCompiler,
    tensorize_sources,
)


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cpu")
    source = NaturalEvidenceCompiler(EvidenceCompilerConfig()).to(device)
    model = IntegratedEpistemicMachine(IEM1Config()).to(device)
    load_nve1_state(model, source.state_dict())

    evidence_rows = generate_training_records()[:3]
    evidence = tensorize_sources(evidence_rows, device)
    with torch.inference_mode():
        expected = source(*evidence[:4])
        actual = model.forward_evidence(*evidence[:4])
    assert torch.equal(expected[0], actual[0])
    assert torch.equal(expected[1], actual[1])

    query_rows = generate_query_training_records()[:3]
    query = tensorize_queries(query_rows, device)
    query_logits = model.forward_query(*query[:3])
    train_evidence = model.forward_evidence(*evidence[:4])
    assert query_logits.shape == (3, 2, 2)

    local_ids, local_mask = tensorize_local_texts(
        ("set", "increase", "strictly below"), device
    )
    operation, comparator = model.forward_local(local_ids, local_mask)
    hard_operation, hard_comparator = model.forward_local(
        local_ids,
        local_mask,
        hard_transport=True,
    )
    assert operation.shape == (3, 7)
    assert comparator.shape == (3, 6)
    assert hard_operation.shape == operation.shape
    assert hard_comparator.shape == comparator.shape
    operation_transport, comparator_transport = model.transports()
    for transport in (operation_transport, comparator_transport):
        assert torch.allclose(
            transport.sum(-1), torch.ones(transport.shape[0]), atol=1e-5
        )
        assert torch.allclose(
            transport.sum(-2), torch.ones(transport.shape[1]), atol=1e-5
        )
    loss = (
        operation.square().mean()
        + comparator.square().mean()
        + query_logits.square().mean()
        + train_evidence[0].square().mean()
        + train_evidence[1].square().mean()
    )
    loss.backward()
    invalid_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is None or not torch.isfinite(parameter.grad).all()
    ]
    assert not invalid_gradients, invalid_gradients
    assert (
        compile_query_batch(
            model,
            (),
            (),
            compiler_commitment="c" * 64,
            device=device,
        )
        == ()
    )

    packet = SimpleNamespace(commitment="a" * 64, symbols=("alpha", "beta"))
    provisional = NaturalQueryReceipt(
        packet.commitment,
        "b" * 64,
        "c" * 64,
        "alpha",
        "beta",
        (
            ("TARGET", "alpha", ((4, 9),)),
            ("DISTRACTOR", "beta", ((18, 22),)),
        ),
        "",
    )
    receipt = replace(
        provisional,
        commitment=canonical_sha256(provisional.payload()),
    )
    query = seal_natural_query(
        packet,  # type: ignore[arg-type]
        receipt,
        expected_compiler_commitment="c" * 64,
    )
    assert query.register == "alpha"
    for field in ("packet", "source", "compiler", "target", "distractor", "commitment"):
        try:
            seal_natural_query(
                packet,  # type: ignore[arg-type]
                mutate_query_receipt(receipt, field),
                expected_compiler_commitment="c" * 64,
            )
        except IEM1RuntimeError:
            pass
        else:
            raise AssertionError(f"mutated query receipt accepted: {field}")
    print("diverge IEM1 runtime tests passed")


if __name__ == "__main__":
    main()
