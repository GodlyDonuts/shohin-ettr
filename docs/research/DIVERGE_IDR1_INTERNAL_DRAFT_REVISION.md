# DIVERGE-IDR1: Internal Draft Revision

Status: draft-generation stage frozen before any IDR1 draft or score on
2026-08-08.

## Hypothesis

VCR1 proves that candidate-conditioned revision can exceed its two source
attempts. SDR1 proves that the same verified targets without candidate
trajectories lose 153 holdout answers. IDR1 preserves temporal computation
while removing external proposal models: one pinned Qwen3.5-9B B1 owner first
generates a complete draft from the source, and a later 9B revision owner sees
`source + internal draft` and emits the final solution.

The model family owns both passes. No 4B B1, 4B QPT1, task router, gold answer,
correctness bit, evaluator feedback, or external tool is available at
inference. The proposal and revision owners share the exact pinned 9B
backbone, but use distinct small LoRA states. This is a deployable two-pass
reasoning architecture rather than source-only SFT.

## Frozen Draft Collection

The source bank is the exact 8,392-row CVG1 bank already partitioned by the
NUL-delimited source identity split. Draft generation uses:

- pinned `Qwen/Qwen3.5-9B@c202236...337b9a`;
- exact 9B B1 checkpoint SHA-256
  `854a7cc44fbc2b54418f4e5bd09b7efeed0da44fc9ce217b0bb6b1997b722971`;
- one greedy sample, no thinking mode, maximum 768 new tokens;
- reasoning prompt style, prompt batch 4, seed `2026080818`;
- eight 512-row math shards, eight 512-row science/logic shards, and one
  200-row execution-verified code shard;
- 17 independent one-H100 jobs with isolated atomic outputs.

Every shard must bind the immutable runtime, model, checkpoint, and source
bank. Merge must prove 8,392 unique identities, exact bank coverage, one draft
per identity, and unchanged source scoring. No draft score is a gate by itself;
drafts become model-visible training inputs only after the complete receipt
passes.

Projected draft-generation charge is `3--6` H100-hours. The later revision fit
and source-disjoint evaluation are separately expected to consume `2--4`
H100-hours. These estimates are reported before launch.

## Conditional Revision Gate

After complete draft custody, IDR1 will use the exact SDR1/VCR1 verified
targets and split. The only new input is the frozen model-owned 9B draft.
Training geometry remains the 256-update pinned 9B B1 LoRA schedule. Holdout
must retain VCR1 within two percentage points (`>=618/1,279`), meet the same
fixed MATH/logic/code and both-wrong floors, and reproduce on development
before the product board opens.

Failure closes exact IDR1 without draft sampling, prompt, seed, LR, duration,
rank, layer, context, decoding, or threshold variants. Success establishes
that the decisive intermediate trajectory can be generated internally by one
host family.
