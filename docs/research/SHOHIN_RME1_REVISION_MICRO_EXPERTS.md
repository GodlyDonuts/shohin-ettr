# RME1: Routed Revision Micro-Experts

Status: frozen pre-mechanics contract, 2026-08-09. Development only; holdout
and larger-MoE transfer remain sealed.

The one-update mechanics job `747303` completed cleanly before capability
evaluation: exact `2,228,224` trainables, `73,728` adapter MAC/token/layer,
`15.996 GiB` peak allocation, zero protected trainables, complete sequence
retention, all four revision routes active in all 16 layers, and minimum load
entropy `0.99536`.

## Why this is structurally different

OLMoE's pretrained native routes do not separate corrected from persistent
revision failures. ECR1's expert codes are causally decorative, and SER1's
native-expert-owned residuals fragment supervision and collapse logic. RME1
therefore does not reuse native expert identity. It adds a small dedicated bank
whose routes are learned solely for revision, while the complete original MoE,
router, attention, embeddings, and language head remain frozen.

At each of all 16 sparse locations, for hidden width `h=2048`, four revision
experts, top two active, and rank `r=8`:

```text
y_native = frozen_native_moe(h)
pi = softmax(R h)
(q, S) = normalized_top2(pi)
delta = sum_(e in S) q_e B_e A_e h
y = y_native + delta
```

`R` is `[4,2048]`; every `A_e` is `[8,2048]`; every `B_e` is `[2048,8]`.
`B_e=0` initially, preserving exact base behavior. `R` receives a fixed
`0.01 * 4 * sum(mean_token(pi)^2)` load-balancing objective so all experts can
receive data before language gradients differentiate their roles. No native
router or expert parameter is trainable.

Treatment has exactly `2,228,224` trainables. Per layer, routing costs 8,192
MACs and two active rank-8 experts cost 65,536, totaling exactly 73,728
MAC/token/layer.

## Matched controls

1. **Active-FLOP shared:** rank-18 shared post-MoE residual at the same 16
   locations: exactly `1,179,648` parameters and `73,728` MAC/token/layer.
2. **Total-parameter shared:** rank-34 shared residual: exactly `2,228,224`
   parameters and `139,264` MAC/token/layer.

All arms use pinned OLMoE revision `b89a7c4...c4650e`, the same complete
model-owned drafts, immutable 9,651-row complete-context corpus, 256 AdamW
updates, LR `2e-5`, seed `2026080901`, data seed `2026080814`, exactly 338,620
charged target tokens, and the unchanged 1,289-row autonomous evaluator. No
source, draft, or target token may be truncated.

## Staged gate

One treatment mechanics update must be finite and prove exact parameter/FLOP,
protected-weight, route-use, sequence-retention, and memory receipts. Stage 1
then trains treatment and both controls exactly once. Promotion requires:

1. treatment at least `280/1,289`;
2. treatment at least 26 answers above the stronger shared control;
3. treatment domain counts at least math `55`, logic/science `180`, code `5`;
4. all four revision experts used in every controlled layer, normalized load
   entropy at least `0.80`, and complete compute/latency/memory receipts.

Only Stage-1 success authorizes evaluation-only zero-residual, uniform-route,
and within-layer whole-expert permutation controls plus one true
draft-unavailable fit. Normal must beat each by at least 13 answers. Only that
conjunctive Stage 2 can open one sealed holdout. Stage-1 failure closes RME1
without expert-count, top-k, rank, depth, duration, seed, or balance-weight
variants.

Interpretation is fixed. Beating active-FLOP but not parameter-matched shared
means sparse conditional computation has not earned inclusion. Beating both,
using all experts, and collapsing under route/expert interventions supports a
revision-specific conditional-compute effect. Otherwise retire RME1.
