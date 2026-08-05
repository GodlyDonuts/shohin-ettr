import torch

from pcsd_conservation_shift import (
    ConservationReasoner,
    ConservationShiftError,
    LedgerConfig,
    evaluation_specs,
    execute_program,
    generate_batch,
    parameter_count,
)


def test_evaluation_splits_keep_confirmation_sealed() -> None:
    assert evaluation_specs("development") == ((8, 41008), (12, 41012))
    assert evaluation_specs("confirmation") == ((16, 91016), (32, 91032))
    try:
        evaluation_specs("invalid")
    except ConservationShiftError:
        pass
    else:
        raise AssertionError("invalid evaluation split was accepted")


def test_programs_preserve_modular_sum() -> None:
    config = LedgerConfig(width=24, heads=3, maximum_depth=8)
    batch = generate_batch(64, 8, config, seed=17, device=torch.device("cpu"))
    target = execute_program(batch.initial, batch.operations, config.modulus)
    assert torch.equal(target, batch.target)
    assert torch.equal(
        batch.initial.sum(-1).remainder(config.modulus),
        batch.target.sum(-1).remainder(config.modulus),
    )


def test_matched_arms_have_sub_one_percent_parameter_mismatch() -> None:
    config = LedgerConfig(width=32, heads=4, maximum_depth=8)
    pcsd = parameter_count(ConservationReasoner(config, "pcsd"))
    dense = parameter_count(ConservationReasoner(config, "dense"))
    assert abs(pcsd - dense) / max(pcsd, dense) < 0.01


def test_reasoner_forward_and_ablation_shapes() -> None:
    torch.manual_seed(23)
    config = LedgerConfig(width=24, heads=3, maximum_depth=8, checks=3)
    batch = generate_batch(5, 4, config, seed=29, device=torch.device("cpu"))
    model = ConservationReasoner(config, "pcsd")
    normal = model(batch)
    zero = model(batch, disable_projection=True)
    shuffled = model(batch, shuffled_checks=True)
    assert normal["state_logits"].shape == (5, config.registers, config.modulus)
    assert normal["answer_logits"].shape == (5, config.modulus)
    assert normal["pre_syndrome"].shape == (5, 4)
    assert normal["post_syndrome"].mean() < normal["pre_syndrome"].mean() / 100
    assert not torch.allclose(normal["state"], zero["state"])
    assert not torch.allclose(normal["state"], shuffled["state"])


def test_dense_control_is_finite_and_differentiable() -> None:
    config = LedgerConfig(width=24, heads=3, maximum_depth=8, checks=3)
    batch = generate_batch(4, 3, config, seed=31, device=torch.device("cpu"))
    model = ConservationReasoner(config, "dense")
    output = model(batch)
    loss = output["state_logits"].mean() + output["answer_logits"].mean()
    loss.backward()
    assert torch.isfinite(output["state"]).all()
    assert model.dense_corrector is not None
    assert model.dense_corrector.basis.grad is not None
