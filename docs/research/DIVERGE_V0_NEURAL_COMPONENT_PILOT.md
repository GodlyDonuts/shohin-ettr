# DIVERGE-v0 Neural Component Pilot

Status: the tiny source compiler is closed as a robustness negative; one frozen
SmolLM2 residual compiler gate is the active successor. This is not the frozen
A--G architecture promotion experiment.

Date: 2026-08-05

## Purpose

The exact CPU packet proved that DIVERGE can preserve and refine a compact
version space without losing coherent worlds. It did not prove that a learned
source compiler can identify the fault lines and construct that packet. This
pilot isolates four learned operations while retaining syntactic candidate
record and option-span scaffolds:

1. classify candidate records versus distractors;
2. compile each option into one of four finite ordered programs;
3. compile the deliberately wrong support prior; and
4. bind delayed evidence to one source-sealed option key.

The exact packet, verifier, conflict refinement, execution, and late reader are
unchanged. The result is deliberately narrower than unrestricted language
compilation or general reasoning.

## Tiny compiler result

The from-scratch compiler has 243,319 trainable parameters. It uses local
lexical convolutions for record classification, a bidirectional GRU for option
program/prior compilation, and a character encoder for exact delayed-evidence
binding. Counterfactual wrapper pairs prevent a record position from serving as
the candidate label.

Five matched seeds completed 600 updates and 9,600 charged episodes each.
Confirmation is perfect for all five seeds, but the development renderer is
not stable:

| Seed | Development joint packet+answer | Development failure | Confirmation joint packet+answer | Report SHA-256 |
|---:|---:|---|---:|---|
| 2026080517 | 100% | none | 100% | `7ad063ebae69758a46c5394efcae8f1eb804ed73eff3131ac4615c84c57c5a70` |
| 2026080518 | 100% | none | 100% | `c6f7deb064011cb6ad6a276d9eb1a65f88a5f0546bc4f128de69c7b8bd4d7351` |
| 2026080519 | 0% | prior compilation 60% | 100% | `75acd4800a89887c986c420590d1bb9f63a683dd002e557e1c009df870f5c094` |
| 2026080520 | 0% | distractor false-positive rate 100%; DIVERGE answer 10.156% | 100% | `b221e2caa3c2b312c5d3c9cf72e9b2efb6fb65e1b55bbd2a0df5c5eef2641959` |
| 2026080521 | 0% | distractor false-positive rate 100%; DIVERGE answer 11.328% | 100% | `f62195b62a80c09867c1378f94dd29eb4eade535c6e127a3477f3db644ce4f78` |

Only two of five seeds satisfy strict joint development and confirmation. The
frozen four-of-five-seed requirement therefore fails. The tiny compiler is
closed; its perfect confirmation split is not a promotion result.

## Diagnosis and one successor

The exact program and delayed-alias fields transfer reliably, while candidate
record detection and one seed's prior semantics do not. This pattern is
consistent with inadequate lexical semantics in a 243k-parameter compiler,
not an exact-packet failure. The one allowed successor changes only the source
representation: frozen layer-17 SmolLM2-135M residuals feed a roughly
million-parameter projection, two-layer local Transformer, and the same finite
heads. The packet runtime, data budget, split, causal release, evaluator, and
strict joint metric remain unchanged.

The real-backbone interface smoke completed on Stokes in 28 seconds with a
finite update, gradient, report, and checkpoint. The first 600-update H100 seed
then completed on `evc30` in 2m54s, training 1,068,775 parameters over 9,600
charged episodes at 91.07 episodes/s. It fails decisively:

| Split | Gold support | Exact packet | DIVERGE answer | Joint packet+answer | Program | Prior | Evidence binding |
|---|---:|---:|---:|---:|---:|---:|---:|
| Development | 99.609% | 0% | 88.672% | 0% | 60.000% | 100% | 100% |
| Confirmation | 62.891% | 0.781% | 62.891% | 0.781% | 100% | 79.134% | 100% |

Development also has 12.694% distractor false positives; confirmation
fault-line recall is only 93.164%. Report/checkpoint SHA-256 values are
`6b9ecebcf94cb5bbdab62b7d29dddd00bdd6b85f5867628b767da48cd87ff6ca` /
`246a4600f1ed9c3d8269582be10c56baadb30ed2943c4a11c8cba0726a9fd42f`.
The remaining four seeds are not launched because seed one misses the strict
joint gate by orders of magnitude.

The pooled-residual compiler is closed. Its renderer-dependent inversion
(development program order versus confirmation prior/candidate support) calls
for an interface redesign, not another tiny width, threshold, wrapper, loss,
duration, or seed. The next implementation must preserve token-level semantic
roles and source order through constrained copy/closure rather than compressing
an entire option into one pooled vector. A read-only fieldwise error audit is
permitted before freezing that replacement.

## Token-role/source-copy qualification

The replacement preserves the semantic boundary at token resolution. Separate
learned roles identify candidate/background cues, favored/reserve support,
four finite action identities, and their source order. A constrained decoder
copies those roles into one complete option before packet closure. It never
averages program fields or consumes the late query. The adapter has 1,013,962
trainable parameters, fewer than the failed pooled compiler.

Five matched seeds each trained for 600 updates / 9,600 charged episodes. Every
seed reaches 100% on both the 256-episode development split and the 256-episode
held-renderer/held-ontology confirmation split for gold-support recall, exact
record set, program, prior, evidence binding, exact packet, DIVERGE answer, and
strict joint packet-plus-answer. Distractor false-positive rate is zero. The
immediate top-1 and no-conflict diagnostics remain zero as predicted.

| Seed | Development joint | Confirmation joint | Report SHA-256 | Checkpoint SHA-256 |
|---:|---:|---:|---|---|
| 2026080522 | 100% | 100% | `2ac508b6456499658c71eae7a34213cfe372b24e33d16a498a944ba5cad35118` | `d614690f6446bf1635fc474d4ae941677b01ac465c0c60381c4b11bbe189826f` |
| 2026080523 | 100% | 100% | `0f6096f6112a103f0d750c3163a05b8c166b9db7846530cb86f2b9ea20a5047f` | `81d3747ef6981524f58e60ca3f4cff377e32b96d3fb3a964bad35bbd9fe9058d` |
| 2026080524 | 100% | 100% | `74d9582b9e68da23d269f740f097e1cbf25b8ad4eb56f7add74fc9ee5e2c6919` | `d53ffc0bce3f58132e8932235c43588ee89a5b6881e289934dae32a73316b742` |
| 2026080525 | 100% | 100% | `ccaade86b32d0cda97f2d8e60f7bf91c3464c74615c578299bad18cf037ba524` | `71c04b7054d821e310c282475ab96d83258b529506e979b452b56246e71791cd` |
| 2026080526 | 100% | 100% | `b91198584f2ed3e60b419f297a5ba2c0f9770d2c65a1ac09bf74db95c2f431a6` | `20e34d06c56b469f80171f8e3f156ef6ea60ddfbb9d5c81f9cbbfe8dedd9d23a` |

Runtime r2 is pinned to private commit `e1bdf8b` at tar SHA-256
`21c9179727d2a721b281b336d6abd18e28ec1ff8bcb22e0fa13170f45c4d8c12`.
All reports/checkpoints are read-only and hash-verified locally and on Newton.
This clears only the learned source-boundary prerequisite. The next gate must
implement the complete matched A--G arms, three late-query types, resource
receipts, and causal interventions already frozen in the promotion contract.

## Claim boundary

The scores named A--G in these pilots are mechanism diagnostics, not complete
parameter/FLOP-matched neural controls. In particular, A--D currently share
the compiler's immediate top-1 decision. No result from this component pilot
can promote DIVERGE. Promotion still requires the frozen complete controls and
resource receipts in `DIVERGE_V0_NEURAL_PROMOTION_GATE.json`.
