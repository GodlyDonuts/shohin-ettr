# SER1: Selected-Expert Revision Residual

Status: closed negative on OLMoE, 2026-08-09. Holdout and larger-MoE transfer
remain sealed.

## Causal hypothesis

ECR1 failed because one shared transform dominated and its low-dimensional
expert codes became causally decorative. SER1 removes that parameterization.
The untouched frozen OLMoE router selects complete expert-owned residual
transforms, so expert identity changes the computation rather than merely
scaling a shared basis.

For each of all 16 sparse layers, hidden width `h=2048`, 64 native experts,
native top 8, and expert rank `r=1`:

```text
(logits, p, S) = frozen_router(h)
y_native = frozen_experts(h, S, p)
q = stopgrad(p / sum(p))
delta = sum_(e in S) q_e B_e A_e h
y = y_native + delta
```

Each `A_e` is `[1,2048]` and `B_e` is `[2048,1]`; every `B_e` starts at zero.
The base router and experts remain frozen. SER1 has exactly `4,194,304`
trainables total, but only eight rank-1 transforms are active per token. Its
adapter compute is `32,768` MAC/token/layer.

## Matched controls

1. **Active-FLOP control:** one shared rank-8 post-MoE residual at the same 16
   locations, `524,288` trainables and `32,768` MAC/token/layer.
2. **Total-parameter control:** one shared rank-64 residual at the same
   locations, `4,194,304` trainables and `262,144` MAC/token/layer.

All three fits use the same pinned OLMoE revision, exact model-owned drafts,
9,651-row complete-context corpus, seed/order, 256 AdamW updates, LR `2e-5`,
338,620 charged target tokens, and unchanged autonomous evaluator. No source,
draft, or target truncation is allowed. Routers and native experts must expose
zero trainable tensors.

## Staged gate

One treatment mechanics update must be finite, preserve protected hashes, and
match the exact parameter/FLOP receipt. Then Stage 1 trains all three arms and
evaluates the full 1,289-row development board. Promotion requires jointly:

1. SER1 at least `280/1,289`;
2. SER1 at least 26 answers above the stronger new shared control;
3. no regression below math `55`, logic/science `180`, or code `5`;
4. complete parameter, token, latency, memory, load-entropy, mean token-
   entropy, expert-use, and residual receipts.

Only a Stage-1 pass may run the same checkpoint under zero, mean-bank, and
within-layer expert-bank permutation interventions plus a true draft-
unavailable fit. Normal must beat every intervention and draft-unavailable by
at least 13 answers. These Stage-2 controls may authorize one sealed holdout;
Stage 1 alone cannot. If Stage 1 fails, close SER1 without rank, depth,
duration, seed, or loss variants.

Interpretation is fixed. Beating the active-FLOP control but not the
parameter-matched control means sparse storage has not earned a capability
claim. Beating both controls and collapsing under expert-bank permutation
supports useful expert-specific revision computation. Failing either score
gate closes the mechanism.

## Result and decision

Mechanics passed exactly. Treatment has `4,194,304` trainables and `32,768`
MAC/token/layer with zero trainable native router/expert tensors. All three
fits completed the same 256 updates and 338,620 charged target tokens.

Full development scores are selected-expert treatment `201/1,289 = 15.5935%`,
active-FLOP shared control `236 = 18.3088%`, and total-parameter shared control
`241 = 18.6967%`. Treatment domain counts are math `56`, logic/science `139`,
and code `6`; logic fails its floor. Exact comparison SHA-256 is
`a44fc09f3e46db195afc0bf32f58e05db0a5e7e030510678c4dbe80bcb7d1fb6`.

SER1 is therefore closed without Stage 2, holdout, or nearby rescue. Native
expert assignment fragments revision learning and is substantially worse than
both shared controls. This rejects post-MoE expert-owned residual banks; it
does not test new dedicated revision experts or adapters inside native expert
nonlinearities.
