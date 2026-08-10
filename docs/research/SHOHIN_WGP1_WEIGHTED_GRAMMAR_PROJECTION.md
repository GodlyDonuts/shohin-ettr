# WGP1: Weighted Grammar Projection

Status: development PASS; confirmation CLOSED on source admission before model scoring

Date: 2026-08-10

Parent: frozen BTT1 checkpoint
`2283c86b1a640c9d5c02ffbc70b4646a0a1ad3538c4e1d4ab4ea3b164363e278`

Holdout: sealed

## Hypothesis

BTT1 is exact on `98.8512%` of the source-disjoint board, and every one of its
45 errors is an invalid byte-role sequence. WGP1 changes no weights or logits.
It uses a fixed width-64 beam to select the maximum-log-probability complete
path satisfying only the generic expression grammar:

- numeric bytes are contiguous and contain at least one digit and at most one
  decimal point;
- operands and binary operators alternate;
- unary negation occurs only where an operand is expected;
- parentheses balance and close only after an operand;
- exactly one nonempty expression is selected from the raw tape.

The projection sees frozen per-byte role logits and raw bytes only to validate
copied numeric characters. It does not inspect an answer, target, verifier,
family, source template, or arithmetic value. The unchanged generic
shunting-yard executor materializes the projected roles.

## Frozen controls and gate

Run normal, same-family/depth source shuffle, zero-byte input, and flat
execution under the same frozen checkpoint and width-64 projection. Compare
row-wise against immutable BTT1 top-1 output.

The zero-byte arm may memoize exact projection results by tape length and the
per-position digit/dot/other mask. Under zeroed inputs, logits are identical
for equal lengths and grammar transitions inspect raw bytes only through that
mask, so this is exact execution reuse rather than an approximation.

The conjunctive development gate requires:

- complete and selected role exact `>=99.5%`;
- valid program `=100%`;
- exact skeleton `>=99.5%` and every family `>=99%`;
- mixed precedence, unary groups, and three-plus-parenthesis each `>=95%`;
- at least 30 of BTT1's 45 failures repaired and at most two previously exact
  rows broken;
- source shuffle and zero-byte controls each `<=25%`, with normal margins
  `>=74` points;
- flat execution loses `>=35` points on hierarchical rows;
- zero search exhaustion and no fallback to top-1.

One pass opens exactly one sealed holdout. Failure closes WGP1 without beam,
grammar, score, checkpoint, seed, or threshold variants. A pass establishes a
qualified source-to-program compiler, not learned arithmetic execution or a
novel parsing algorithm.

## Development result

Jobs `749707`, `749708`, `749710`, and exact memoized zero-byte replacement
`749711` completed. Normal projected output is `3,917/3,917 = 100%` for
complete byte roles, selected lexemes, action sequence, validity, and exact
skeleton. It repairs all 45 immutable BTT1 failures with zero breaks and zero
search exhaustion. Every family and every frozen difficult slice is `100%`.
Source-shuffled and zero-byte controls are both zero exact; flat execution
loses `62.2244` points on hierarchical rows. Every conjunctive gate passes.

The original zero-byte job `749709` was canceled after `4m23s` without a
report when it became clear that repeated identical zero-input searches would
consume its window. Replacement `749711` used the prospectively documented
exact cache and completed all rows; no score, beam, grammar, or checkpoint
changed. Result SHA-256 is
`113dffc54e30ef90d7c36662c8f4cef445d5afc6d16e422719bdc13d8553f060`.

Exactly one confirmation is now open on the immutable Reasoning-Gym held-out
seed file `rg_v4/rg_eval.jsonl`, SHA-256
`35a2625a44bd42161ac9ee562fb57a6f69ab9771747ef4166d0042d925b46435`.
The five supported arithmetic families contribute 500 source-disjoint rows
each before fail-closed exact-program admission. Confirmation thresholds are
frozen before construction: at least 99% admission per family, `>=99%` exact
skeleton overall, every family `>=98%`, valid program `=100%`, source-shuffled
and zero-byte exact each `<=25%`, and zero exhaustion. A miss closes WGP1.

## Confirmation source admission correction

CPU job `749717` failed closed in three seconds before producing a holdout
artifact or accessing model output. A read-only audit found that the legacy
`rg_eval.jsonl` is not source-disjoint from BTT1 train/development: it contains
14 protected `chain_sum` questions, 138 protected `products` questions, and
one internal `products` duplicate. This invalidates that file as a WGP1
confirmation source; it is preserved as a data-custody failure and will never
be scored.

Before any replacement data or model output exists, the replacement source is
frozen as follows: `reasoning-gym==0.1.25`, package wheel SHA-256
`7f17a3eddb13c015d7d4a755ed576a061df889faf9468bcc2cca334ebe9e0435`,
seed `20260810`, an ordered pool of 8,192 generated rows per family, and the
first 500 nonempty-answer rows per family whose normalized question SHA-256 is
unique and absent from immutable BTT1 train/development. The five families,
compiler, checkpoint, projection, controls, and all numeric gates remain
unchanged. The generated source must contain exactly 2,500 unique,
source-disjoint rows; independent exact-program admission must still retain at
least 495/500 per family before any model evaluation. Failure at either stage
closes confirmation without another seed, package, pool, or filtering retry.

## Confirmation result

Replacement-source job `749728` completed in five seconds with exactly 2,500
unique, protected-disjoint rows. Source/report SHA-256 values are
`fcc8579857138be82ecfc149b9c268bcd4a0c38c3419f1313a7d4b64f4246689`
and `3d3d817531b867074b3292a51d80938efc5d6e5aab134e8c3624d377e4ac28a2`.

Admission job `749730` then failed closed in two seconds before creating an
output directory or accessing the BTT checkpoint. Exact admitted counts are:

| Family | Admitted | Rate |
|---|---:|---:|
| basic arithmetic | 493/500 | 98.6% |
| chain sum | 500/500 | 100% |
| decimal arithmetic | 226/500 | 45.2% |
| decimal chain sum | 500/500 | 100% |
| products | 500/500 | 100% |

The seven basic-arithmetic exclusions are generated integer answers rounded
from nonintegral division results, while all 274 decimal-arithmetic exclusions
are precision-rounded answers that intentionally differ from exact-rational
execution. This is a semantic incompatibility between the frozen admission
assessor and Reasoning-Gym 0.1.25, not a model score. Nevertheless the
prospective gate is conjunctive, so exact WGP1 confirmation closes without a
different seed, package, source filter, or evaluator. The 100% development
compiler remains a valid development result but is not holdout-qualified.
Receipt: `SHOHIN_WGP1_HOLDOUT_ADMISSION_RESULT.json`.
