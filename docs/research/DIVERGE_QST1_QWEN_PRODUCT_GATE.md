# DIVERGE-QST1: Qwen Stage-Owned Transaction Gate

Status: frozen before CUDA training or benchmark scores, 2026-08-07.

## Objective

Test whether the strongest mechanism learned in controlled DIVERGE experiments
can improve a current, useful language model. The host is the exact pinned
`Qwen/Qwen3.5-0.8B` revision
`2fc06364715b967f1860aea9cf38778875588b17`. It has 852,985,920 parameters and
is loaded from the existing immutable Newton artifact. QST1 is model-owned: no
external solver, teacher model, tool, answer router, or host arithmetic runs at
inference.

This is a product-capability experiment. It does not claim that the controlled
OPB1/SNL1 interfaces already solve open-language compilation.

## Architecture

QST1 replaces the old globally active recurrent workspace with three disjoint
owners and one coherent state lineage:

1. **Source owner:** contextual Qwen prompt states are compiled once into eight
   immutable source slots with episode-local slot identities.
2. **Transaction owner:** eight separate state slots evolve for eight tied
   recurrent steps. Cross-attention may read the source packet, but source
   slots are never overwritten. Per-field write gates implement copy-on-write,
   and cumulative adaptive stopping monotonically reduces later writes.
3. **Query owner:** four late query slots read the final state and immutable
   source only after recurrence.
4. **Commit:** the complete source packet, one final state lineage, and late
   query packet form a 20-slot soft prefix. There are no exchangeable particles,
   fieldwise hypothesis averages, hard straight-through interfaces, or an
   external selection policy.

The workspace receives two causal auxiliary signals in addition to response
language loss. Batch-contrastive provenance requires each evolved state to
remain bound to its own source packet, and a reset margin requires the evolved
state to become more source-consistent than the pre-transaction state. A small
late-stop loss discourages endless updates. These signals target the measured
cross-owner interference and identical-trajectory collapse of prior lanes.

Default trainable geometry is width 384, 8 source slots, 8 state slots, 4 query
slots, 8 recurrent steps, 8 attention heads, and the same final-four-layer
rank-8/alpha-16 LoRA used by the protected baseline.

## Matched Development Contract

| Field | Frozen value |
|---|---:|
| Host revision | `2fc06364715b967f1860aea9cf38778875588b17` |
| Data | V10 verified-priority stream |
| Data SHA-256 | `2461d6f70b44a142854d56c24e1fb42d600065e5788a2c4e055ba47b12696549` |
| Selected rows | 26,387 |
| Updates | 1,000 |
| Logical examples/update | 16 |
| Maximum sequence | 1,024 |
| Learning rate | `2e-4` cosine decay |
| Seed / data seed | `31` / `20260802` |
| LoRA | final 4 layers, rank 8, alpha 16 |

The protected matched baseline is
`baseline_v10_verified_u1000_2461d6f/checkpoint_0001000.pt`. Checkpoint 200 is a
learning-curve diagnostic only; checkpoint 1,000 decides promotion.

## Evaluation And Decision

Use the unchanged evaluator, prompts, deterministic decoding, answer extraction,
and boards for GSM8K, MATH-500, HumanEval, MBPP, GPQA-Diamond, BBH logic, and
AIME-2024. First score the existing fixed development subsets; a broader or
fresh milestone opens only after the fixed gate passes.

Promote QST1 to the next product milestone only if checkpoint 1,000:

- improves five-domain macro by at least 3.0 points over the matched B1 result;
- solves at least 15 additional fixed-board examples;
- improves at least three scored domains;
- regresses no domain by more than 2.0 points;
- preserves a finite, stable transaction trace and unchanged protected-weight
  hash; and
- loses its gain under source-packet swap or state-reset intervention on a
  bounded causal replay.

If QST1 misses, close this exact geometry and training rule. Do not rescue it
with width, duration, seed, threshold, or board selection. The next change must
alter the mechanism or supervision, not merely spend more updates.
