# DIVERGE-SDR1: Source-Only Verified Reasoning Distillation

Status: completed once and closed on 2026-08-08. Development and holdout both
fail the frozen retention gate, so the product board remains sealed and SDR1
receives no variants.

## Capability Question

VCR1 reached `643/1,279` on source-disjoint holdout and `72.302%` product
macro, but requires two complete 4B candidate solutions at inference. SDR1
tests the shortest path to a standalone reasoner: can the identical verified
repair supervision teach the pinned 9B host to solve directly from the source
problem, without either candidate?

This is not a VCR1 rescue variant. It is a matched causal control and a
different deployable system boundary. If SDR1 retains VCR1 capability, the
verified target curriculum rather than candidate-conditioned revision is the
main transferable mechanism. If it collapses, the candidate trajectories are
causally necessary and must remain inside the architecture.

## Single Changed Factor

SDR1 matches VCR1's:

- exact pinned Qwen3.5-9B B1 warm start;
- `5,824/1,289/1,279` source split and row order;
- 9,655 training presentations and every response target;
- outcome-class weighting through deterministic duplicate presentations;
- 256 updates, batch 1, accumulation 8, context 4,096, LR `2e-5`;
- four LoRA layers, rank 8, alpha 16, optimizer, schedule, and seeds;
- greedy source-disjoint evaluator and assessor.

The only changed factor is `question`: VCR1 receives the source plus Candidate
A and Candidate B; SDR1 receives only the original task prompt. Candidate
text, correctness, task label, outcome class, gold answer, execution result,
and assessor fields are unavailable to the runtime. An independent audit
proves exact response-target/order parity for all 9,655 presentations and
finds zero `Candidate A/B` markers in any SDR1 prompt. Training prompt bytes
fall from `40,823,625` to `6,059,006`.

## Frozen Data

| Artifact | Rows | SHA-256 |
|---|---:|---|
| Train | 9,655 | `a97344d4339d2687b6ed861844294efd14c575962811811ee74b97f70a7771d6` |
| Development | 1,289 | `aa0e700069a4f066cc16514d7db4f53d95ebaf6911cd4118f55cd8a09fecf5f0` |
| Holdout | 1,279 | `3e623da7b11a0bc36adbbfd0be845e3310b0125744eae09d37e8575ba2f696ce` |
| Report | - | `62c4eb5d9a1d1b19d9ea8b954dec4d5a2c09ea4c1791e4c176e4605d2bded97b` |

Immutable runtime `runtime/sdr1_d11b231_r1` has manifest SHA-256
`bffddf1ef4d8eb25b97065ccddd81f4190fd633ed56d280c34e8bcdb47351b6b`.

## Frozen Gates

Development must satisfy all:

- at least `550/1,289` overall;
- MATH at least `224/623`;
- BBH logic at least `294/637`;
- MBPP at least `19/29`;
- at least 75 correct cases from the both-source-wrong stratum.

Holdout must satisfy all:

- at least `618/1,279`, retaining VCR1 within two percentage points;
- MATH at least `255/621`;
- BBH logic at least `327/625`;
- MBPP at least `24/33`;
- at least 100 correct cases from the both-source-wrong stratum.

Both splits must pass before one unchanged 568-row product evaluation may
open. A qualified standalone product model must retain code `>=30/40`, reach
at least `70%` five-domain macro and `350/538` solved, improve at least three
domains over QPT1, and regress no domain by more than two points versus QPT1.

Failure closes exact SDR1 without prompt, target, seed, LR, duration, rank,
layer, context, decoding, or threshold variants. A result below VCR1 but above
QPT1 may be recorded as a standalone specialist, but does not establish that
candidate-conditioned reasoning was removed without capability loss.

## Budget

Projected total charge before launch is `2--4` H100-hours for one fit, two
source-disjoint evaluations, and one conditional product evaluation. Current
monthly use is `409.9/2000.0` H100-hours.

## Result

Training job `745625` completes all 256 matched updates in `00:12:33`, charging
365,028 target tokens at 511.70 target tok/s. Checkpoint and report SHA-256
values are
`8947a593793d42321e76f671cefd8c28ad34c94cc005ad91ac5ce8c756697195`
and
`c15ff1dbe6ab7fbcfa2062f925efbbb0d3fb765518e9cb294c54e6361309fa7d`.
The shorter source-only context makes fit `1.72x` faster than VCR1 but does not
preserve its capability.

| Split | QPT1 | VCR1 | SDR1 | SDR1 both-wrong solved |
|---|---:|---:|---:|---:|
| Development | 453/1,289 | 575/1,289 | **448/1,289** | 126 |
| Holdout | 471/1,279 | 643/1,279 | **490/1,279** | 137 |

On holdout, SDR1 scores MATH `142/621`, BBH logic `321/625`, and MBPP
`27/33`. Code and the both-wrong floor pass, but overall, MATH, and science
retention fail. Development independently agrees: MATH `109/623`, logic
`319/637`, MBPP `20/29`, and overall `448/1,289`. Development/holdout report
SHA-256 values are
`64282b672a7fe3c644f823bbfd8933594c600f2aaa200fc3a665c0e75ca756f3`
and
`516f244896dee9c1ad1e08049c55bf3f43a507d085c0d606413ba7268a1cc481`.

This is a clean causal result. The verified target curriculum alone does not
explain VCR1: source-only holdout is 153 answers below candidate-conditioned
VCR1. Candidate trajectories are especially important for MATH, where VCR1
solves 273 and SDR1 only 142. SDR1 still demonstrates some direct transfer
over QPT1 on holdout (`+19`) and strong code, but it is not a standalone
replacement. The prebuilt product board stays unopened. The next architecture
must generate a draft internally and revise it in a later model-owned pass;
one-pass source-only distillation is closed.

Training plus both evaluations consumed `0.676` H100-hours, below budget.
