# ECR1: Expert-Conditioned Revision Residual

Status: frozen implementation and one-update mechanics contract, 2026-08-09.
No 256-update fit or holdout access is authorized until the mechanics receipt
passes unchanged.

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

Every arm receives 256 AdamW updates, LR `2e-5`, seed `2026080901`, data seed
`2026080814`, and the exact MTR1/RCR1 9,655-row training population. Context is
4,096. Tokenization must fail closed if any complete source, exact draft, or
target would be truncated. Original and retained counts must match exactly.

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
524,288 shared trainables. Otherwise it is unauthorized. If both depths fail,
ECR1 closes on OLMoE and does not scale to 35B.

Interpretation is fixed: shared passing without an ECR margin supports generic
MLP-side correction and rejects expert identity; ECR passing with its matched
margin and causal code interventions supports sparse expert conditioning;
neither passing closes the mechanism.

## Implementation

Core: `train/ecr1_moe_revision.py`. Trainer:
`train/train_ecr1_product.py`. Slurm entry point:
`train/jobs/train_ecr1_product.sbatch`. Autonomous loading and exact draft-key
masking are integrated into `train/hf_product_reasoning_eval.py`.
