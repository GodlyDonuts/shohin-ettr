# Shohin IDR08 Capacity Boundary

Status: frozen before 0.8B draft generation, training, or source-disjoint
scores. Launch remains conditional on closing the pending 4B protected product
comparison.

## Question

Qwen3.5-4B proved that a model can improve difficult reasoning by generating
one complete internal draft and then using a separately trained state of the
same backbone to revise it. IDR08 asks whether that exact mechanism remains
causally useful at 0.8B parameters. It is a capacity-boundary experiment that
selects the scratch Shohin scale, not a new mechanism search.

## Frozen Intervention

- Host: pinned `Qwen/Qwen3.5-0.8B` revision
  `2fc06364715b967f1860aea9cf38778875588b17`.
- Warm start: the existing final-four-layer rank-8 B1 checkpoint produced by
  the matched QST1 campaign; its path and SHA-256 must be recorded before
  admission.
- Bank: the exact 8,392 source-disjoint identities used by IDR1/IDR4: 4,096
  MATH, 4,096 logic/science, and 200 execution-verified code cases.
- Draft: one source-only B1 trajectory, greedy, no thinking mode, batch 4,
  maximum 768 new tokens, seed `2026080818`.
- Revision data: the same verified IDR targets, train/development/holdout
  split, and source-plus-internal-draft presentation contract as IDR4.
- Training: warm-start the 0.8B B1 state; 256 AdamW updates; batch 1 with
  gradient accumulation 8; 4,096-token context; learning rate `2e-5`; final
  four layers; rank 8; alpha 16; seeds `2026080815/2026080814`.
- Control: the unchanged 0.8B B1 state performs the identical second pass over
  the same question/draft inputs, evaluator, token budget, batches, and seeds.

No external proposal model, verifier, solver, tool, answer router, or task
router is present at inference.

## Decision Gate

Development and holdout are conjunctive. Each split must independently show:

1. at least `+0.05` absolute overall accuracy for trained revision versus the
   unchanged second pass;
2. nonnegative exact-answer delta on MATH;
3. nonnegative exact-answer delta on logic/science;
4. nonnegative exact-answer delta on executable code; and
5. complete identity, model, checkpoint, prompt, generation, and evaluator
   receipts.

`pipeline/compare_idr_scale.py` applies this rule once to the four complete
merged reports. A pass selects `shohin_390m`; any miss selects
`shohin_920m`. There is no seed, rank, duration, layer, prompt, decoding, or
threshold rescue family.

## Cost And Sequencing

Expected cost is approximately `4--8` H100-hours for 17 independent draft
shards and `1--3` H100-hours for revision fit plus sharded treatment/control
evaluation. Record actual aggregate and critical-path GPU time. Draft shards
may run concurrently, but data construction requires complete unique identity
coverage and all downstream jobs remain dependency-gated.

The result chooses a plausible scratch capacity. It does not authorize the
large scratch run until the fresh-corpus contract, multi-H100 transport canary,
checkpoint/resume test, and 5B-token capability canary also pass.
