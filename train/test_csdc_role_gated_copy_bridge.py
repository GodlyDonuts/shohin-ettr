import torch

from csdc_role_gated_copy_bridge import (
    CopyBridgeConfig,
    CopyLogits,
    OUTCOME_ROLE,
    RoleGatedCopyParser,
    START_ROLE,
    WORD_ROLE,
    copy_loss,
    label_source_roles,
)
from csdc_semantic_bridge import gather_targets, render_semantic_source
from prompt_selected_presented_algebra import PresentedAlgebraConfig
from pspa_presented_reasoning import generate_batch


def test_role_labels_cover_exact_source_fields() -> None:
    algebra = PresentedAlgebraConfig()
    batch = generate_batch(5, 4, algebra, seed=271)
    source = render_semantic_source(
        batch, algebra, seed=277, templates=(0, 1, 2, 3)
    )
    labeled = label_source_roles(source, algebra)
    challenge = source.challenge_record
    assert labeled.token_role.eq(START_ROLE)[challenge].sum().item() == 5 * 8
    assert labeled.token_role.eq(OUTCOME_ROLE)[challenge].sum().item() == 5 * 8
    assert labeled.token_role.eq(WORD_ROLE)[challenge].sum().item() == int(
        source.challenge_length[challenge].sum().item()
    )


def test_copy_parser_has_finite_source_only_gradient() -> None:
    torch.manual_seed(281)
    algebra = PresentedAlgebraConfig()
    parser = RoleGatedCopyParser(
        algebra, CopyBridgeConfig(width=24, heads=3, layers=1)
    )
    batch = generate_batch(4, 3, algebra, seed=283)
    source = render_semantic_source(
        batch, algebra, seed=293, templates=(0, 1, 2)
    )
    labeled = label_source_roles(source, algebra)
    loss, _ = copy_loss(parser(source.rendered), labeled)
    loss.backward()
    assert torch.isfinite(loss)
    assert parser.role.weight.grad is not None
    assert torch.isfinite(parser.role.weight.grad).all()


def test_oracle_role_logits_copy_every_challenge_exactly() -> None:
    torch.manual_seed(307)
    algebra = PresentedAlgebraConfig()
    parser = RoleGatedCopyParser(
        algebra, CopyBridgeConfig(width=24, heads=3, layers=1)
    )
    batch = generate_batch(6, 4, algebra, seed=311)
    source = render_semantic_source(
        batch, algebra, seed=313, templates=(0, 1, 2, 3)
    )
    labeled = label_source_roles(source, algebra)
    raw = parser(source.rendered)
    kind = torch.full_like(raw.kind, -20.0)
    kind[..., 0] = 20.0
    kind[..., 0][source.challenge_record] = -20.0
    kind[..., 1][source.challenge_record] = 20.0
    role = torch.full_like(raw.role, -20.0)
    role.scatter_(-1, labeled.token_role[..., None], 20.0)
    decoded = parser.decode(CopyLogits(kind=kind, role=role), source.rendered)
    true_record, true_start, true_outcome, true_length, true_word = gather_targets(
        source, decoded.record_index
    )
    positions = torch.arange(algebra.maximum_word_length)
    word_mask = positions[None, None] < true_length[..., None]
    assert true_record.all()
    assert decoded.start.eq(true_start).all()
    assert decoded.outcome.eq(true_outcome).all()
    assert decoded.length.eq(true_length).all()
    assert (decoded.word.eq(true_word) | ~word_mask).all()
