# R12 Source-Deleted Multi-Family Machine Qualification

## Status

CPU mechanics and leakage gates pass. Neural fitting is not yet authorized.
Confirmation generation is unavailable. Continuation pretraining remains on
user hold.

This qualification is the direct successor to
`R12_EXPEDITED_REASONING_CONCLUSION.md`. It tests whether one shared
architecture compiles and executes episode-local laws across genuinely
different law families. It does not permit a general-reasoning claim from a
single family or from the exact CPU compiler.

## Board

`pipeline/source_deleted_multifamily_machine_board.py` defines three law
families behind one anonymous finite-machine interface:

1. affine maps over a modular ring;
2. bitwise rotate/xor maps; and
3. unconstrained permutations.

Every episode uses one role-neutral `h<20 hex>` codec for fresh action and
state names. No lexical prefix distinguishes their semantic class. The source
contains a shuffled complete transition presentation. The
late query supplies a start state and an ordered action composition only after
the source has been compiled and deleted.

All families use cardinality 8 during fitting and 16 under law/scale holdout.
This prevents family identity from being inferred from tensor geometry.

The four renderer orbits are:

- semicolon-delimited fields;
- prose;
- tuples; and
- reverse-arrow notation.

Renderers 0--2 are available for fitting. Renderer 3 is held out.

## Frozen Splits

The current development board has 1,344 rows:

| Split/cell | Rows |
|---|---:|
| fit | 576 |
| unseen law/scale | 192 |
| unseen composition length | 192 |
| unseen renderer | 192 |
| joint law + composition + renderer | 192 |

Each family contributes exactly 448 rows. Fitting compositions have length
1--4; composition and joint holdouts have length 5--8. All 576 fitting laws
and all 768 development laws are unique and disjoint.

Three leave-one-family-out folds are frozen. In each fold, two complete
families are available for fitting and the third is score-only. No
family-specific executor, output head, prompt, or optimizer state is allowed.

## Candidate Boundary

The candidate compile process receives only raw source bytes. The supervisor
owns family, split, renderer, law hash, answer, and all gold transition
indices. After compilation:

1. raw source bytes are deleted;
2. source-token residuals and KV state are deleted;
3. the persistent packet contains only the candidate's anonymous machine;
4. the late query is disclosed;
5. execution uses the packet and one shared recurrent executor; and
6. no oracle, search, verifier, exact parser, or host transition function is
   available.

`compile_source` is an exact CPU upper bound and audit oracle. It is forbidden
from candidate inference.

## Candidate Architecture

The first neural treatment must adapt the retained full-trajectory recurrent
mechanism rather than introduce a family-specific controller:

- immutable 125,081,664-parameter Shohin trunk;
- one raw-record source encoder;
- one anonymous key-binding/object-file compiler;
- one tied recurrent transition/execution core;
- one anonymous next-state readout;
- at most 74,918,336 added parameters;
- fewer than 200,000,000 complete parameters.

Training must expose complete episodes in source order and execute complete
action words. Resetting recurrent state at every labeled transition is not an
admissible treatment.

## Matched Arms

Five model seeds must train these exact-budget arms:

1. full-trajectory recurrent treatment;
2. recurrence disabled with equal depth and FLOPs;
3. action-to-law binding shuffled within each episode;
4. equal-parameter record classifier with no persistent execution state; and
5. randomized labels.

The treatment and controls must share candidate bytes, initialization lineage,
parameter count, optimizer updates, batch schedule, precision, and evaluation
budget. A deterministic exact compiler/executor is reported only as a
non-neural upper bound.

## Promotion Gates

Every gate is required:

1. 100% source-deletion and custody checks;
2. at least 95% fitting exactness;
3. at least 85% exactness in every family and every development cell;
4. at least 85% in every leave-one-family-out fold;
5. at least 80% on the joint unseen-law, longer-composition, held-out-renderer
   cell for every family;
6. treatment exceeds every matched learned control by at least 10 percentage
   points overall and 5 points within every family;
7. every one of five seeds passes the direction of effect;
8. action-order reversal and law-swap interventions produce the predicted
   counterfactual answer changes; and
9. family/renderer renaming leaves exact behavior invariant.

Passing these gates supports bounded systematic machine reasoning across the
three frozen families. It does not by itself establish unrestricted natural
language or general reasoning.

## CPU Audit Receipt

The independent audit is
`pipeline/audit_source_deleted_multifamily_machine_board.py`.

Frozen receipt:
`artifacts/r12/source_deleted_multifamily_machine_board_v1/audit_receipt.json`

- board seed: 20260725
- rows: 1,344
- exact source-deleted execution: 1,344/1,344
- source deletion passes: 1,344/1,344
- family-name leaks: 0
- role-neutral opaque-key codec: 1,344/1,344
- renderer orbits exact and packet-identical: 192/192
- law-swap answer changes: 1,176/1,344 = 87.5%
- action-order answer changes: 629/909 = 69.1969%
- candidate manifest SHA-256:
  `8033200439b4ea30a626c0e6b3af56037b340feda2d281c74f04023d35c56ca3`
- supervisor manifest SHA-256:
  `9343cd91d3484dfbe220b92072f88cc002b7fdea20237fe47bc4f3de22f62930`
- payload SHA-256:
  `c3b3936fedd1e9b606818822838c5a3a8609ddc62e9abdf84cc7a75a7f6c1163`
- receipt file SHA-256:
  `36c64c735c0da6c987eab0872449d3f352e6dfe0f06da025f3f500b9cc5ea0e1`

The focused mechanics/audit suite is 18 passed with clean Ruff and byte
compilation.

## Standalone Learned Smoke

`train/multifamily_raw_machine_compiler.py` implements a candidate that sees
only role-neutral key equality, record boundaries, masked raw bytes, and late
query bytes. A shared bidirectional recurrent encoder predicts all source and
query semantic roles. Constrained hard sealing emits a canonical source-free
machine wire. The candidate imports no exact board parser.

A 300-update CPU smoke uses 152,933 learned parameters:

| Cell | Exact |
|---|---:|
| fitting renderers 0--2 | 36/36 |
| unseen laws | 6/6 |
| longer compositions | 6/6 |
| unseen reverse renderer | 0/6 |
| joint law + composition + renderer | 0/6 |

Fitting source/query role accuracy is 100% and loss falls from 1.763858 to
0.00000696. The unseen renderer produces invalid key partitions: development
source-role accuracy is 50% overall because the two renderer-0 cells are
exact and the two renderer-3 cells fail. Candidate-time oracle, search, and
verifier calls are zero.

Decision:
`standalone_byte_compiler_optimizes_but_does_not_transfer_renderer_semantics`.
Do not scale its width. The justified next treatment supplies frozen,
hash-verified Shohin residual features to the same compiler while preserving
all hard custody and matched controls.

Smoke report SHA-256:
`e4ff0d09abb9533e7d8c16b213f9b6a073e672572a4799c94803f7eb0c539ff3`.

## Authorization Boundary

Before an H100 run, the raw-token candidate package, process-level deletion
boundary, exact parameter/FLOP ledger, five-seed schedule, and independent
assessor must be implemented and frozen. Until then, this is a qualified
board and upper-bound receipt, not a learned result.
