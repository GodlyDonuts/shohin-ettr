from dataclasses import replace
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace

from csdc_role_gated_copy_bridge import OTHER_ROLE, START_ROLE
from csdc_smollm2_span_quotient_bridge import (
    ERROR_PARTIAL,
    SHIFT_GENERATOR_ALIASES,
    SHIFT_STATE_ALIASES,
    TRAIN_GENERATOR_ALIASES,
    TRAIN_STATE_ALIASES,
    SpanQuotientChallengeParser,
    SpanQuotientLogits,
    aggregate_evaluations,
    assess_gate,
    decode_span_logits,
    render_span_lexical_source,
    span_quotient_loss,
)
from csdc_smollm2_lexical_bridge import LexicalBridgeConfig, gather_lexical_targets
from prompt_selected_presented_algebra import PresentedAlgebraConfig
from pspa_presented_reasoning import generate_batch


def split_alias_tokenizer() -> Tokenizer:
    vocab = {"[UNK]": 0}
    for word in (
        *TRAIN_STATE_ALIASES,
        *SHIFT_STATE_ALIASES,
        *TRAIN_GENERATOR_ALIASES,
        *SHIFT_GENERATOR_ALIASES,
    ):
        prefix = word[:2]
        suffix = "##" + word[2:]
        if prefix not in vocab:
            vocab[prefix] = len(vocab)
        if suffix not in vocab:
            vocab[suffix] = len(vocab)
    tokenizer = Tokenizer(WordPiece(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def rendered_source(count: int = 4):
    algebra = PresentedAlgebraConfig()
    batch = generate_batch(count, 4, algebra, seed=701, family=2)
    source = render_span_lexical_source(
        batch,
        algebra,
        split_alias_tokenizer(),
        seed=709,
        templates=(0, 1, 2, 3),
        shifted_aliases=True,
        seq_len=2048,
    )
    return algebra, batch, source


def oracle_logits(source) -> SpanQuotientLogits:
    kind = torch.full((*source.record_mask.shape, 2), -20.0)
    kind[..., 0] = 20.0
    kind[..., 0][source.challenge_record] = -20.0
    kind[..., 1][source.challenge_record] = 20.0
    role = torch.full((source.candidate_batch.numel(), 4), -20.0)
    role[:, OTHER_ROLE] = 20.0
    exact = source.candidate_target_role.ne(OTHER_ROLE)
    role[exact, OTHER_ROLE] = -20.0
    role[exact, source.candidate_target_role[exact]] = 20.0
    return SpanQuotientLogits(kind=kind, role=role)


def test_span_renderer_represents_all_multitoken_mentions() -> None:
    _, _, source = rendered_source()
    exact = source.candidate_target_role.ne(OTHER_ROLE)
    widths = source.candidate_end[exact] - source.candidate_start[exact] + 1
    assert widths.min().item() == 2
    assert widths.max().item() == 2
    assert source.labeled_mentions.eq(source.represented_mentions).all()
    assert source.candidate_class.max().item() + 1 < source.candidate_class.numel()


def test_oracle_whole_spans_decode_every_challenge_exactly() -> None:
    algebra, _, source = rendered_source(7)
    decoded, audit = decode_span_logits(oracle_logits(source), source, algebra)
    true_record, true_start, true_outcome, true_length, true_word = (
        gather_lexical_targets(source, decoded.record_index)
    )
    positions = torch.arange(algebra.maximum_word_length)
    word_mask = positions[None, None] < true_length[..., None]
    assert true_record.all()
    assert audit.valid.all()
    assert decoded.start.eq(true_start).all()
    assert decoded.outcome.eq(true_outcome).all()
    assert decoded.length.eq(true_length).all()
    assert (decoded.word.eq(true_word) | ~word_mask).all()


def test_partial_alias_span_fails_closed() -> None:
    algebra, _, source = rendered_source(1)
    logits = oracle_logits(source)
    challenge_records = torch.nonzero(source.challenge_record[0]).flatten()
    record = int(challenge_records[0].item())
    exact = torch.nonzero(
        source.candidate_batch.eq(0)
        & source.candidate_record.eq(record)
        & source.candidate_target_role.eq(START_ROLE),
        as_tuple=False,
    ).flatten()
    assert exact.numel() == 1
    start = int(source.candidate_start[exact].item())
    end = int(source.candidate_end[exact].item())
    partial = torch.nonzero(
        source.candidate_batch.eq(0)
        & source.candidate_record.eq(record)
        & source.candidate_error_kind.eq(ERROR_PARTIAL)
        & source.candidate_start.ge(start)
        & source.candidate_end.le(end),
        as_tuple=False,
    ).flatten()
    assert partial.numel() > 0
    role = logits.role.clone()
    role[exact, START_ROLE] = -20.0
    role[exact, OTHER_ROLE] = 20.0
    role[partial[0], OTHER_ROLE] = -20.0
    role[partial[0], START_ROLE] = 20.0
    _, audit = decode_span_logits(
        replace(logits, role=role), source, algebra
    )
    selected_records = audit.selected_candidate[..., 0].clamp_min(0)
    selected_record_ids = source.candidate_record[selected_records]
    affected = selected_record_ids.eq(record)
    assert affected.any()
    assert not audit.valid[affected].any()


class FakeBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(n_loop=1, d_model=16)
        self.blocks = torch.nn.ModuleList([torch.nn.Identity()])


def test_exact_surface_class_reindex_is_logit_invariant() -> None:
    _, _, source = rendered_source(2)
    parser = SpanQuotientChallengeParser(
        FakeBackbone(),
        LexicalBridgeConfig(
            layer=0,
            width=8,
            heads=2,
            encoder_layers=1,
            ff=16,
        ),
    ).eval()
    memory = torch.randn(source.ids.shape[0], source.ids.shape[1], 8)
    first = parser.score_memory(memory, source)
    maximum = int(source.candidate_class.max().item())
    second = parser.score_memory(
        memory,
        replace(source, candidate_class=maximum - source.candidate_class),
    )
    assert torch.equal(first.kind, second.kind)
    assert torch.equal(first.role, second.role)


def test_span_loss_is_finite_and_differentiable() -> None:
    _, _, source = rendered_source(2)
    logits = SpanQuotientLogits(
        kind=torch.randn(*source.record_mask.shape, 2, requires_grad=True),
        role=torch.randn(source.candidate_batch.numel(), 4, requires_grad=True),
    )
    loss, metrics = span_quotient_loss(logits, source)
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["kind_loss"] > 0
    assert metrics["span_role_loss"] > 0
    assert logits.kind.grad is not None
    assert logits.role.grad is not None


def test_span_gate_requires_every_causal_condition() -> None:
    rows = []
    for split in ("development", "lexical_shift"):
        for family in ("cyclic", "dihedral", "random"):
            rows.append(
                {
                    "split": split,
                    "family": family,
                    "length": 8,
                    "count": 100,
                    "learned_accuracy": 0.97 if split == "development" else 0.93,
                    "oracle_accuracy": 0.995,
                    "shuffle_outcome_accuracy": 0.55,
                    "lineage_swap_accuracy": 0.15,
                    "challenge_tuple_exact": 0.97 if split == "development" else 0.92,
                    "selected_table_exact": 0.97 if split == "development" else 0.92,
                    "decoded_all_valid": 0.96,
                    "class_zero_answer_accuracy": 0.70,
                    "class_zero_tuple_exact": 0.60,
                    "class_reindex_bit_identical": 1.0,
                    "start_mention_exact": 0.98,
                    "outcome_mention_exact": 0.98,
                    "word_mention_exact": 0.97,
                    "gold_mention_exact": 0.97 if split == "development" else 0.92,
                    "representability": 1.0,
                    "selected_partial": 0,
                    "selected_superset": 0,
                    "selected_overlap": 0,
                    "accepted_partial": 0,
                    "accepted_superset": 0,
                    "accepted_overlap": 0,
                    "candidate_classes": 100,
                    "selected_classes": 20,
                    "missing_start": 0,
                    "duplicate_start": 0,
                    "missing_outcome": 0,
                    "duplicate_outcome": 0,
                    "missing_word": 0,
                    "excess_word": 0,
                    "nonexact_identity": 0,
                }
            )
    summary = aggregate_evaluations(rows)
    assert all(assess_gate(summary).values())
    rows[-1]["class_reindex_bit_identical"] = 0.99
    assert not assess_gate(aggregate_evaluations(rows))[
        "class_reindex_bit_identical"
    ]
