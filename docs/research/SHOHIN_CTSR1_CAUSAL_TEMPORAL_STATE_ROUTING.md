# CTSR1: Causal Temporal State Routing

Status: frozen pre-implementation contract, 2026-08-09. Development only.
Holdout and large-MoE transfer remain sealed.

## Hypothesis

MTR1, RCR1, ECR1, SER1, and RME1 show that static native expert identity is
not predictive of revision success. The missing variable may be temporal:
revision requires a diagnosis that survives the draft prefill and changes as
the model emits a replacement trajectory. CTSR1 tests whether a small causal
state, updated across actual tokens and persisted through cached generation,
can make later sparse computation useful.

At each of all 16 sparse layers, a single GRU shared across layers reads the
pre-MoE hidden sequence. Each layer owns only a 64-dimensional initial code
and persistent inference state. Two tied 64-to-32-to-64 heads read the causal
state. Every layer also owns the already useful rank-18 shared post-MoE
residual.

```text
s[l,t] = shared_GRU(h[l,t], s[l,t-1])
u1, u2 = tied_heads(s[l,t])

treatment:
  router_logits = frozen_logits + tanh(u1)
  residual_gate = P[l] tanh(u2)

matched temporal control:
  router_logits = frozen_logits
  residual_gate = P[l] tanh((u1 + u2) / 2)

delta = B[l] (A[l] h * (1 + residual_gate))
output = frozen_selected_experts(h, router_logits) + delta
```

`P[l]` is a deterministic fixed projection, never trained. During training,
the GRU is reset for each complete source+draft+target sequence and is causal
over tokens. During greedy generation it consumes the left-padded prompt once,
stores the final state separately at every layer, then advances that state on
every generated token. No frozen-backbone prepass, pooled draft vector, source
KV side channel, or host state exists.

## Exact accounting

Geometry is fixed: hidden 2,048, state 64, head width 32, residual rank 18,
16 layers, 64 native experts, native top-8.

- shared GRU: `405,888` parameters;
- 16 layer initial codes: `1,024`;
- two tied state heads: `8,192`;
- 16 rank-18 residuals: `1,179,648`;
- total: exactly `1,594,752` trainables in either arm;
- adapter MAC/token/layer: exactly `488,576` in either arm.

All native attention, routers, experts, embeddings, norms, and LM head remain
frozen. Both arms use identical heads and execute identical GRU/head/residual
work. The changed factor is whether `u1` changes expert routing or supplies a
second generic residual-state signal.

## Development gate

One-update mechanics must prove exact base parity before optimization, exact
parameter/MAC receipts, finite gradients, left-padding equivalence, streaming
prefill/token equivalence to one-shot causal recurrence, protected hashes, and
complete 4,096-token sequence custody.

Only then train treatment and temporal control once for the existing 256
updates, LR `2e-5`, seed/data seed, 9,651 complete rows, and 338,620 charged
target tokens. Reuse static shared rank-18 `248/1,289` read-only as an external
floor. Promotion requires all of:

1. treatment at least `280/1,289`;
2. treatment at least 26 answers above the newly trained temporal control;
3. treatment at least 13 answers above static shared rank-18;
4. domain counts at least math `55`, logic/science `180`, code `5`;
5. finite state/route receipts, all 64 native experts used, load entropy at
   least `0.80`, and top-1 route-change rate at least `0.01`.

Only a conjunctive pass opens evaluation-only state reset, route-head zero,
state transplant, and true draft-unavailable controls. Normal must beat each
by at least 13 answers before one holdout. Failure closes CTSR1 and the
small-OLMoE MoE-native route without width/depth/rank/duration/seed variants.

