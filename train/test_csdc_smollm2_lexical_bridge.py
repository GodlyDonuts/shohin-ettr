from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
import torch

from csdc_role_gated_copy_bridge import OUTCOME_ROLE, START_ROLE, WORD_ROLE
from csdc_smollm2_lexical_bridge import (
    LexicalChallengeParser,
    LexicalLogits,
    aggregate_evaluations,
    assess_gate,
    gather_lexical_targets,
    lexical_loss,
    render_lexical_source,
)
from prompt_selected_presented_algebra import PresentedAlgebraConfig
from pspa_presented_reasoning import generate_batch


def tiny_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def test_lexical_rendering_aligns_every_copied_field() -> None:
    algebra = PresentedAlgebraConfig()
    batch = generate_batch(5, 4, algebra, seed=401, family=2)
    source = render_lexical_source(
        batch,
        algebra,
        tiny_tokenizer(),
        seed=409,
        templates=(0, 1, 2, 3),
        shifted_aliases=True,
        seq_len=2048,
    )
    challenge_tokens = source.challenge_record.gather(
        1, source.token_record.clamp_min(0)
    ) & source.valid_mask
    assert source.challenge_record.sum().item() == 5 * algebra.maximum_challenges
    assert source.token_role.eq(START_ROLE)[challenge_tokens].sum().item() == 5 * 8
    assert source.token_role.eq(OUTCOME_ROLE)[challenge_tokens].sum().item() == 5 * 8
    assert source.token_role.eq(WORD_ROLE)[challenge_tokens].sum().item() == int(
        source.challenge_length[source.challenge_record].sum().item()
    )
    assert source.ids.shape == source.valid_mask.shape == source.token_record.shape
    assert source.valid_mask.sum().item() > 0


def test_oracle_logits_copy_exact_lexical_challenges() -> None:
    algebra = PresentedAlgebraConfig()
    batch = generate_batch(7, 4, algebra, seed=419, family=1)
    source = render_lexical_source(
        batch,
        algebra,
        tiny_tokenizer(),
        seed=421,
        templates=(0, 1, 2, 3),
        shifted_aliases=False,
        seq_len=2048,
    )
    kind = torch.full((*source.record_mask.shape, 2), -20.0)
    kind[..., 0] = 20.0
    kind[..., 0][source.challenge_record] = -20.0
    kind[..., 1][source.challenge_record] = 20.0
    role = torch.full((*source.ids.shape, 4), -20.0)
    role.scatter_(-1, source.token_role[..., None], 20.0)
    decoded, valid = LexicalChallengeParser.decode(
        None,
        LexicalLogits(kind=kind, role=role),
        source,
        algebra,
    )
    true_record, true_start, true_outcome, true_length, true_word = (
        gather_lexical_targets(source, decoded.record_index)
    )
    positions = torch.arange(algebra.maximum_word_length)
    word_mask = positions[None, None] < true_length[..., None]
    assert true_record.all()
    assert valid.all()
    assert decoded.start.eq(true_start).all()
    assert decoded.outcome.eq(true_outcome).all()
    assert decoded.length.eq(true_length).all()
    assert (decoded.word.eq(true_word) | ~word_mask).all()


def test_lexical_loss_is_finite_for_source_only_logits() -> None:
    algebra = PresentedAlgebraConfig()
    batch = generate_batch(3, 2, algebra, seed=431, family=0)
    source = render_lexical_source(
        batch,
        algebra,
        tiny_tokenizer(),
        seed=433,
        templates=(0, 1, 2),
        shifted_aliases=False,
        seq_len=2048,
    )
    logits = LexicalLogits(
        kind=torch.randn(*source.record_mask.shape, 2, requires_grad=True),
        role=torch.randn(*source.ids.shape, 4, requires_grad=True),
    )
    loss, metrics = lexical_loss(logits, source)
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["kind_loss"] > 0
    assert metrics["role_loss"] > 0
    assert logits.kind.grad is not None
    assert logits.role.grad is not None


def test_gate_assessor_requires_every_causal_condition() -> None:
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
                    "decoded_all_valid": 1.0,
                }
            )
    summary = aggregate_evaluations(rows)
    assert all(assess_gate(summary).values())
    rows[-1]["learned_accuracy"] = 0.89
    failed = assess_gate(aggregate_evaluations(rows))
    assert not failed["every_shift_cohort_answer_ge_90"]

