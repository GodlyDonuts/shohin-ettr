import inspect

import torch

from csdc_semantic_bridge import (
    SemanticBridgeConfig,
    SemanticChallengeParser,
    render_semantic_source,
    semantic_loss,
)
from prompt_selected_presented_algebra import PresentedAlgebraConfig
from pspa_presented_reasoning import generate_batch


def test_semantic_renderer_is_query_free_and_labels_all_challenges() -> None:
    algebra = PresentedAlgebraConfig()
    batch = generate_batch(7, 4, algebra, seed=211)
    source = render_semantic_source(
        batch, algebra, seed=223, templates=(0, 1, 2)
    )
    assert source.challenge_record.sum().item() == 7 * algebra.maximum_challenges
    assert source.challenge_length[source.challenge_record].min().item() >= 2
    assert "query" not in inspect.signature(render_semantic_source).parameters


def test_semantic_parser_has_finite_source_only_gradient() -> None:
    torch.manual_seed(227)
    algebra = PresentedAlgebraConfig()
    config = SemanticBridgeConfig(width=24, heads=3, layers=1)
    parser = SemanticChallengeParser(algebra, config)
    batch = generate_batch(4, 3, algebra, seed=229)
    source = render_semantic_source(batch, algebra, seed=233, templates=(0, 1, 2))
    logits = parser(source.rendered)
    loss, _ = semantic_loss(logits, source)
    loss.backward()
    assert torch.isfinite(loss)
    assert parser.word.weight.grad is not None
    assert torch.isfinite(parser.word.weight.grad).all()


def test_decode_returns_exact_fixed_challenge_geometry() -> None:
    torch.manual_seed(239)
    algebra = PresentedAlgebraConfig()
    config = SemanticBridgeConfig(width=24, heads=3, layers=1)
    parser = SemanticChallengeParser(algebra, config)
    batch = generate_batch(3, 2, algebra, seed=241)
    source = render_semantic_source(batch, algebra, seed=251, templates=(3,))
    decoded = parser.decode(parser(source.rendered), source.rendered)
    assert decoded.start.shape == (3, algebra.maximum_challenges)
    assert decoded.word.shape == (
        3,
        algebra.maximum_challenges,
        algebra.maximum_word_length,
    )
    assert decoded.word_mask.shape == decoded.word.shape

