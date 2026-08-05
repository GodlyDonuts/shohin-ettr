from dataclasses import replace
import inspect

import torch

from learned_pspa_language_reasoning import (
    LanguageConfig,
    LearnedPSPAGate,
    PresentationCompiler,
    compiler_loss,
    greedy_permutation,
    render_source,
)
from pspa_presented_reasoning import PresentedAlgebraConfig, generate_batch


def test_renderer_is_deterministic_and_query_free() -> None:
    config = PresentedAlgebraConfig()
    batch = generate_batch(9, 4, config, seed=151)
    first = render_source(batch, config, seed=157)
    second = render_source(
        replace(batch, query_start=batch.query_start.roll(1)), config, seed=157
    )
    assert torch.equal(first.tokens, second.tokens)
    assert torch.equal(first.token_mask, second.token_mask)
    assert "query" not in inspect.signature(render_source).parameters


def test_greedy_projection_is_a_whole_permutation() -> None:
    torch.manual_seed(163)
    scores = torch.rand(7, 3, 11, 11)
    tables = greedy_permutation(scores)
    assert torch.equal(tables.sum(-1), torch.ones_like(tables.sum(-1)))
    assert torch.equal(tables.sum(-2), torch.ones_like(tables.sum(-2)))


def test_presented_compiler_trains_one_finite_source_only_step() -> None:
    torch.manual_seed(167)
    algebra = PresentedAlgebraConfig()
    language = LanguageConfig(width=24, heads=3, layers=1, ff_multiplier=2)
    compiler = PresentationCompiler(algebra, language, projection="presented")
    batch = generate_batch(6, 3, algebra, seed=173)
    source = render_source(batch, algebra, seed=179)
    _, tables = compiler(source, batch.generator_mask, hard=False)
    loss = compiler_loss(tables, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert compiler.output.weight.grad is not None
    assert torch.isfinite(compiler.output.weight.grad).all()


def test_all_three_learned_arms_have_finite_outputs() -> None:
    torch.manual_seed(181)
    algebra = PresentedAlgebraConfig()
    language = LanguageConfig(width=24, heads=3, layers=1, ff_multiplier=2)
    model = LearnedPSPAGate(algebra, language)
    batch = generate_batch(5, 3, algebra, seed=191)
    source = render_source(batch, algebra, seed=193)
    _, presented = model.presented(source, batch.generator_mask, hard=False)
    _, row_soft = model.row_soft(source, batch.generator_mask, hard=False)
    direct = model.direct(batch, source)
    assert torch.isfinite(presented).all()
    assert torch.isfinite(row_soft).all()
    assert torch.isfinite(direct).all()

