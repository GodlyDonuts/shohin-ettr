# MPR1: Practical MoE Temporal Revision

Status: closed negative on development, 2026-08-09. Holdout and larger-MoE
transfer remain sealed.

## Objective

MPR1 asks the practical causal question left open by the closed MoE-native
lanes: can a small frozen OLMoE improve materially by training a simple
all-layer post-MoE residual to revise its own exact draft? It does not require
router novelty or dense-model parity.

## Host and changed factor

- `allenai/OLMoE-1B-7B-0125-Instruct` at
  `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e`;
- frozen attention, routers, experts, embeddings, and language head;
- one shared rank-18 residual after each of all 16 sparse blocks,
  `delta = B A h`, alpha 18;
- exactly 1,179,648 trainable parameters and 73,728 adapter MAC/token/layer;
- 256 AdamW updates, LR `2e-5`, batch 1, accumulation 8, seed `2026080901`,
  data seed `2026080814`, and maximum sequence length 4,096.

This is the strongest simple OLMoE intervention already observed: its prior
matched-control instance scored 248/1,289. MPR1 changes the causal supervision,
not its geometry.

## Data and arms

The immutable OLMoE draft bank and source-disjoint split from MTR1 are rebound
through the pinned OLMoE tokenizer. Every admitted row must preserve the full
source, complete exact model-owned draft, and target at 4,096 tokens. Rows are
admitted identically across all trained arms; target multisets and update/FLOP
budgets are exact matches.

1. **Aligned:** source plus its exact OLMoE-owned draft.
2. **Shuffled:** identical source and target, but a different source's
   same-domain nearest-token-length OLMoE draft.
3. **Hidden/source-only:** identical aligned token IDs and positions, but all
   draft-token keys are causally hidden throughout the full model during
   training and generation.
4. **Unchanged:** frozen OLMoE second pass with the exact aligned prompt.

The hidden arm is the equal-compute source-only control. No Qwen-tokenized row,
external teacher, verifier, benchmark label, or target is available at
inference.

## Frozen development gate

Across the exact 1,289-row development board, promotion requires every item:

1. aligned at least `230/1,289`, exactly 39 answers above unchanged `191`;
2. aligned at least 13 answers above shuffled;
3. aligned at least 13 answers above hidden/source-only;
4. aligned domains do not regress below unchanged: MATH `40`, logic/science
   `145`, executable code `5`;
5. conservative semantic attribution reports at least 13 net possible-semantic
   repairs: possible-semantic repairs minus strict breaks `>=13`;
6. complete model/data/runtime hashes, zero protected trainables, exact
   parameter/update/token/FLOP receipts, and complete 4,096-token custody.

One-update mechanics must pass before the three fits. A conjunctive development
pass opens exactly one sealed holdout evaluation with the same arms and
requires the same margins and nonnegative domain deltas. A development or
holdout miss closes exact MPR1 without rank, depth, duration, seed, prompt,
threshold, or decoding rescue. A development-plus-holdout pass establishes
Shohin temporal revision as operational on small OLMoE and authorizes staged
larger-MoE transfer.

## Result

The OLMoE-tokenized corpus contains 9,651 matched rows and 1,575,873 target
tokens per arm, with 5,824 unique source identities, zero source/donor identity
matches, same-task donors, donor token-length delta p95 one, and maximum total
length 2,616. The known four overlength presentations were excluded from every
arm. Mechanics passed, and each fit completed exactly 256 updates and 338,931
charged tokens with 1,179,648 trainables.

Aligned scores `233/1,289`, shuffled `247`, hidden/source-only `247`, and
unchanged `191`. Aligned therefore gains 42 answers over unchanged but loses 14
to each causal control. Its domains are MATH `52`, logic/science `177`, and code
`4`, so code also misses the unchanged floor of five. Conservative attribution
finds 107 strict repairs, 13 certified serialization-only repairs, 94 remaining
possible-semantic repairs, and 65 strict breaks, for semantic net `+29`.

The exact conclusion is that all-layer shared post-MoE SFT is useful, but the
current weak OLMoE draft is not a useful causal input: an unrelated draft and a
fully hidden draft both do better. MPR1 is closed without geometry, duration,
seed, or threshold rescue. Result:
`docs/research/SHOHIN_MPR1_RESULT.json`.
