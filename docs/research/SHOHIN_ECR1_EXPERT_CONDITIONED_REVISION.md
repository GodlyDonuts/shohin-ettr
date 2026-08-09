# ECR1: Expert-Conditioned Revision Residual

Status: closed negative on OLMoE, 2026-08-09. Holdout and larger-MoE scaling
remain unauthorized.

## Why DREM1 is not decisive

DREM1 is preserved as an unmatched upper-bound diagnostic, not a capability
gate. Its controller is recurrent across controlled layers but static across
generated tokens; its old draft-masked arm removed only a pooled draft vector;
it added a full frozen-backbone prepass; it had about 6.84M trainables versus
MTR1's 524,288; and its 1,024-token default risked left truncation. DREM1 job
`747085` was canceled before allocation and charged zero H100 time.

## Fixed evidence and hypothesis

The host, data, drafts, prompts, decoding, and evaluator remain pinned to
OLMoE-1B-7B-0125-Instruct at revision
`b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e`. Unchanged scores `191/1,289`;
MTR1 shared-attention rank 8 scores `204`; router-only RCR1 scores `194`.
At least 11 of MTR1's 16 repairs are serialization-only. Holdout is sealed.

ECR1 tests whether frozen native expert identity contains useful information
for a small same-pass correction that generic post-MoE capacity cannot recover.
It neither changes native routing nor adds a model prepass.

For each of the final four sparse layers, hidden width `h=2048`, 64 experts,
native top 8, and rank `r=31`:

```text
(logits, p, S) = frozen_router(h)
y_native = frozen_experts(h, S, p)
q = stopgrad(p / sum(p))
c = sum_(e in S) q_e tanh(C[e])
delta = B((A h) * (1 + c))
y = y_native + (alpha/r) delta, alpha=r
```

`A` is `[31,2048]`, `B` is `[2048,31]`, and `C` is `[64,31]`.
`B=C=0` initially, so initial behavior is bit-exact with the frozen base.
Only A/B/C train. Total trainables are exactly `515,840`. Additional work is
`127,224` MACs (`254,448` FLOPs) per token per controlled layer.

## Matched arms

1. **ECR1 treatment:** equation above, final four sparse layers.
2. **Shared residual:** same post-MoE location, `delta=B A h`, rank 32,
   exactly `524,288` trainables and `131,072` MACs/token/layer.
3. **Draft-unavailable ECR1:** identical ECR parameters, input IDs, positions,
   and training rows; draft-token keys receive zero attention in the complete
   second pass. This is a full-model causal key mask, not pooled-vector zeroing.
   Generation passes explicit original prompt position IDs so internal zero
   attention entries cannot compress positions.

Every arm receives 256 AdamW updates, LR `2e-5`, seed `2026080901`, and data
seed `2026080814`. Exhaustive pre-output custody found four copies of one
duplicate source identity require 4,332 tokens, beyond pinned OLMoE's hard
4,096 context. Those four occurrences are deterministically excluded from all
new arms before training; the remaining 9,651-row population is otherwise
unchanged and its longest row is 2,616 tokens. No source, exact draft, or target
token may be truncated. The admitted corpus and exclusion report are immutable
and hash-bound before fits. This four-row difference from historical MTR1 is a
declared limitation; the decisive ECR-versus-shared and draft-causality arms
use identical admitted rows.

## Frozen gates

Development promotion is conjunctive:

1. treatment at least `256/1,289`;
2. treatment at least 39 answers above the stronger newly trained control;
3. no regression versus unchanged in MATH (`>=40`), logic/science (`>=145`),
   or MBPP (`>=5`);
4. treatment at least 13 answers above draft-unavailable treatment;
5. at least 25 repairs not certified serialization-only, and possible-semantic
   repairs minus semantic breaks at least 20;
6. normal expert codes beat each of zero, mean, and deterministic within-layer
   permutation by at least 13 answers. If permutation does not hurt, expert
   identity is decorative and the expert-conditioned claim fails;
7. complete protected hashes, source/draft retention, parameter/FLOP, peak
   memory, latency, generated-token, native route, and residual receipts.

Metrics distinguish aggregate `load_entropy` from mean per-token entropy and
also report top-8 set/order overlap, top-8/top-9 margin, later-layer route
drift, residual norms by outcome class, expert-code cosine/effective rank, and
expert utilization. Routers and experts must have zero trainable tensors.

Only a complete development pass opens one sealed holdout comparison. Holdout
requires at least +5 points over unchanged, at least +3 points over shared,
nonnegative domain deltas, and the same causal/semantic conditions.

## Conditional depth follow-up

One follow-up is predeclared now: only when both final-four ECR1 and shared
residual improve unchanged by at least 13 answers but miss promotion, test all
16 layers at rank 8. The receipts are fixed at 532,480 ECR trainables and
524,288 shared trainables. Stage 1 contains only these two fits and their exact
development evaluations. It must score at least 256/1,289, beat shared by at
least 39 answers, and preserve the frozen 40/145/5 math/logic-code floors.
Only a conjunctive Stage-1 pass may spend compute on draft-removal and
expert-code interventions; it never directly opens holdout. Otherwise it is
unauthorized. If Stage 1 fails, ECR1 closes on OLMoE and does not scale to 35B.

Interpretation is fixed: shared passing without an ECR margin supports generic
MLP-side correction and rejects expert identity; ECR passing with its matched
margin and causal code interventions supports sparse expert conditioning;
neither passing closes the mechanism.

## Implementation

Core: `train/ecr1_moe_revision.py`. Trainer:
`train/train_ecr1_product.py`. Slurm entry point:
`train/jobs/train_ecr1_product.sbatch`. Autonomous loading and exact draft-key
masking are integrated into `train/hf_product_reasoning_eval.py`.

## Results and decision

Final-four development scores are ECR `221/1,289`, shared `223`, and true
draft-unavailable `224`, versus unchanged `191` and MTR1 `204`. Zeroing,
averaging, or permuting the learned expert codes leaves ECR exactly unchanged
at `221`; expert identity is decorative in this parameterization. The exact
comparison SHA-256 is
`0b10336326f87b0e1b07114b98a6dde6cd02c0a06140d43502199f408a92ab72`.

The predeclared all-layer follow-up was eligible and completed. Rank-8 ECR
uses exactly `532,480` trainables and scores `240/1,289 = 18.6191%`; rank-8
shared uses `524,288` and scores `239 = 18.5415%`. ECR domain counts are
math `55`, logic/science `180`, and code `5`. Domain floors pass, but ECR
misses the `256` floor and beats shared by one answer rather than 39. Exact
comparison SHA-256 is
`320076637af912e194b315ab6e7589240602126b8ab505184bdf2f499c602166`.

Therefore Stage 2, holdout, and 35B scaling are blocked. ECR1 is closed on
OLMoE without rank, layer, seed, duration, or loss rescue. The supported
boundary is generic post-MoE correction capacity, not expert-conditioned
sparse correction.
