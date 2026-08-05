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

## Claim boundary

The scores named A--G in these pilots are mechanism diagnostics, not complete
parameter/FLOP-matched neural controls. In particular, A--D currently share
the compiler's immediate top-1 decision. No result from this component pilot
can promote DIVERGE. Promotion still requires the frozen complete controls and
resource receipts in `DIVERGE_V0_NEURAL_PROMOTION_GATE.json`.
