# DIVERGE-IEM1 Integrated Epistemic Machine Result

Status: **FAIL** on the one frozen run, 2026-08-06.

## Question

Can one shared trainable whole-mention encoder, one learned latent-to-primitive
transport, and one optimizer replace the separately qualified TOL3 source,
NVE1 evidence, and typed-query interfaces while retaining exact factorized
execution and delayed recovery?

IEM1 is a controlled integration gate. It is not a public benchmark,
continuation-pretraining result, or test of unrestricted language reasoning.

## Frozen lineage and data

- frozen specification commit: `6f75eca`;
- implementation commit: `058d2ad`;
- hash-bound Newton launcher commit: `8fd48c5`;
- corrected immutable TOL3 artifact paths: `112f5d7`;
- empty-evaluation-batch fail-closed repair: `fb23f6b`.

The repair adds only an empty-input return to the query batch compiler. It was
made after training, changes no model, data, prediction, denominator, or gate,
and allowed the already-trained checkpoint to report a valid component miss.

IEM1 reuses 50,000 NVE1 evidence statements and adds 50,000 deterministic
natural-query statements. The fresh confirmation board contains 256 programs,
3,072 evidence items, 768 late queries, 12 binary fault lines and 4,096 worlds
per program, or 1,048,576 represented worlds. Training and confirmation have
zero exact evidence, query, or program-identity overlap. The immutable hashes
are:

- query training data: `8dc6085f7632b56563416fa75c13ff611344ad53c7c2ca4350c54bbcc17301fa`;
- confirmation board: `4294799cace78330eabdbba369d383feac870329edd00e15ba53d3a24cc9ce8a`;
- data report: `2e516549ce5d7fb4ece0967c2255d8c272cad0e98750bb33710f0f83e87d8f6c`;
- inherited evidence data: `7eb27276332a14dcbf57c651c8823393fa7eb18b4122cff30c07b961db285b35`.

## Model and training

The 550,343-parameter IEM1 starts from the exact NVE1 byte encoder. The same
encoder serves source operations and predicates, evidence roles, and query
roles. Four operation channels and six comparator channels are transported
through learned doubly normalized matrices into fixed rational primitives.
Only the composed semantic distribution is supervised; hard inference commits
to one latent channel and one primitive.

Newton job `744075` ran on one H100 for exactly 1,000 AdamW updates at batch
256 and learning rate `1e-3` with cosine decay. It consumed 43,176,637 source
bytes in 119.504 seconds, sustained 361,298 source bytes/second, used
497,242,624 peak allocated bytes, and recorded no skipped or non-finite
updates. Training fit was:

- evidence numeric, symbol, and joint roles: **50,000/50,000** each;
- natural query role: **50,000/50,000**;
- local operation phrases: **48/50**;
- local comparator phrases: **11/18**.

The final operation and comparator train accuracies were 96.0% and 61.1%.
The final total loss was 0.294324; evidence loss was zero and query loss was
effectively zero. The learned transport remained diffuse. Direct hard/soft
inspection showed `SET` at 0/2 and `LT/GE` at 0/3 each, while the remaining
direct operations were exact.

Artifact hashes are:

- checkpoint: `c7560eb5c0bb08a8bbe0f15b1127790859fb2a12bf80d5a89f292be30789e84a`;
- model state: `6552b6fbd6e736338ae2b6a583ab11f36ea141a66f5b925454791fe7f2f21738`;
- training report: `1bb8861ce8567215034936860c1ce39c376b27b12f5b3be2bc55c9d4339519dd`.

## Confirmation result

Official evaluation job `744089` completed cleanly on H100 `evc31`. The
learned source compiler produced **0/256** exact fresh programs. Every failure
was `declaration operation differs`: the failed learned `SET` mapping invalidated
every program before a sealed packet existed. In contrast, the immutable
separate source ceiling compiled **256/256** programs.

Because no IEM1 packet was valid, no fresh evidence receipt, query binding,
factorized execution, or protected joint recovery received an evaluable
denominator. The evaluator therefore reports all matched arms as zero:

| Arm | Exact |
|---|---:|
| integrated IEM1 | 0/256 |
| immutable separate ceiling in the joint loop | 0/256 |
| premature top-1 | 0/256 |
| equal-memory particles | 0/256 |
| no-evidence abstention | 0/256 |

The table's denominator is the frozen board size; internally the end-to-end
loop evaluated zero valid IEM1 packets. The separate source compiler's
independent 256/256 result and the previously frozen NVE1 256/256 result remain
the valid modular ceilings. Fresh IEM1 evidence/query generalization must not
be described as passing: their training sets fit exactly, but source failure
prevented confirmation evaluation. Vacuous zero-denominator integrity flags
are not positive evidence.

Joint-training regression controls also fail closed: the IEM1 source interface
is 0/1,024 on protected TOL3 semantic programs, and the IEM1-composed path is
0/256 on the protected NVE1 board. No invalid receipt, false commitment,
malformed packet, gold deletion, or overflow is accepted because execution is
never entered.

Official H100 evaluation SHA-256 is
`afe52c7ec3f68da8aa01b5999a1479ff13aed50f268aec1efb884a309e9804bf`.
An independent CPU run gives identical component counts, matched arms,
controls, protected regressions, and promotion conditions; its diagnostic
SHA-256 is
`9ee1baef29cd53e11b8817dd3750f3d9e46a2cabdd3bf4a63b8a0a8986cef674`.

## Decision and architectural implication

IEM1 fails the first component condition by the maximum possible margin. The
frozen conjunctive promotion gate fails. Per preregistration, there will be no
IEM1 width, duration, seed, transport, renderer, optimizer, or loss variant.

The negative is specific and useful. The factorized runtime did not fail;
execution was never reached. Evidence and query ownership were learnable in
isolation and fit jointly, while anonymous shared semantic transport lost a
single mandatory structural primitive and thereby destroyed every complete
program. A universal encoder plus symmetric latent transport is therefore too
brittle for transaction-critical semantic ownership.

The next architecture must preserve complete, qualified specialists and join
them through a typed, provenance-bound transaction bus. Source declarations,
state updates, predicates, evidence receipts, and late queries need distinct
owners with explicit contracts, while one model-owned controller coordinates
their recurrent use. This is not an IEM1 repair: it rejects universal shared
encoding and anonymous transport as the integration mechanism. The next gate
must first prove that one composite checkpoint retains the immutable TOL3 and
NVE1 component ceilings before measuring autonomous composition. No
continuation pretraining is authorized by this result.
