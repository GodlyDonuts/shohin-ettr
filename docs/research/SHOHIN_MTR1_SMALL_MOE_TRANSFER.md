# Shohin MTR1 Small-MoE Transfer Gate

Status: frozen before model-owned draft generation, 2026-08-09.

## Question

Does trained same-family temporal revision transfer to a small open mixture of
experts while its router and experts remain frozen?

MTR1 is the next scale/family point after the Qwen dense scale curve,
SmolLM3-3B aggregate transfer, and the OLMo2-7B negative. It is not scratch
pretraining and does not authorize the larger Qwen3.6 MoE campaign.

## Host

- model: `allenai/OLMoE-1B-7B-0125-Instruct`;
- revision: `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e`;
- license: Apache-2.0;
- topology: 16 decoder layers, width 2,048, 64 experts, 8 active experts per
  token, 7B total parameters and approximately 1B active parameters;
- context: 4,096 tokens.

The pinned snapshot, its config hash, and its complete file-manifest hash must
be recorded before capability work starts.

## Changed Factor

1. The unchanged OLMoE owner writes one complete greedy draft from the source.
2. A separate state of the same OLMoE model reads the identical source and
   exact internal draft.
3. Only rank-8 LoRA projections in shared attention token mixers of the final
   four layers are trained for 256 updates. Router and expert parameters remain
   frozen and their names/hashes are checked.
4. The reviser emits one coherent complete trajectory. No fieldwise averaging,
   external solver, verifier, answer router, or teacher runs at inference.

Training uses batch 1, accumulation 8, maximum sequence length 4,096, AdamW,
learning rate `2e-5`, alpha 16, and the existing data/order seeds. Draft and
final generation are greedy with a 768-token budget and batch 4.

## Data And Evaluator

MTR1 reuses the exact source-disjoint IDR/TTR geometry:

- 4,096 MATH identities, SHA-256 `e0ede832...dbe5`;
- 4,096 logic/science identities, SHA-256 `5a96859f...017`;
- 200 execution-verified MBPP identities, SHA-256 `0b6d068b...398`;
- 9,655 training presentations, 1,289 development identities, and 1,279
  sealed holdout identities after model-owned drafts are joined;
- unchanged exact-answer and executable-code assessors.

No target or assessor field is visible to the runtime. Holdout remains sealed
until the development conjunction passes.

## Matched Arms

All arms use the same source, internal draft, final prompt, evaluator, decoding
budget, and inference accounting where applicable.

1. trained temporal revision;
2. unchanged second pass;
3. generic self-refinement;
4. longer source-only generation with the same two-pass token ceiling;
5. best-of-two with deterministic tie handling;
6. parameter/update-matched independent commitment with the draft span masked.

The report must include prompt/generated tokens, wall time, peak memory,
trainable and active parameters, estimated FLOPs, router load distribution,
expert utilization, and accuracy per generated token and per estimated FLOP.

## Pass And Stop Rules

Mechanics must first prove a finite backward update, exact trainable-name
inventory, zero trainable router/expert parameters, nonempty model-owned drafts,
and complete provenance.

Development passes only if all conditions hold across all 1,289 identities:

1. treatment exceeds unchanged second pass by at least 5 absolute points;
2. treatment exceeds the strongest fully matched standard control by at least
   3 absolute points;
3. MATH, logic/science, and executable-code correct counts are each
   nondecreasing versus unchanged;
4. every arm has complete generation and compute receipts;
5. router/expert accounting is complete and no protected parameter changed.

A development miss closes exact MTR1 without seed, rank, layer, duration,
prompt, threshold, or decoding rescue. A pass opens the one sealed holdout
evaluation. Only a development-plus-holdout pass can authorize the larger
Qwen3.6 MoE campaign.

