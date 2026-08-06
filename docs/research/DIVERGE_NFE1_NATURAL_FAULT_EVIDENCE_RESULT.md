# DIVERGE-NFE1 Natural Fault Evidence Result

Status: **PASS** on the one frozen run, 2026-08-06.

## Question

Can a learned whole-mention interface compile source-bound numeric evidence
from a dataset-disjoint natural arithmetic source and use that packet to refine
the protected DIVERGE version space, without exposing raw source after sealing?

NFE1 is an interface/mechanism gate. It is not a public benchmark, continuation
pretraining result, or claim of unrestricted reasoning.

## Frozen inputs

- implementation commit: `f9a70b7`
- source corpus SHA-256:
  `2461d6f70b44a142854d56c24e1fb42d600065e5788a2c4e055ba47b12696549`
- 2,179 deduplicated verified `reasoning_gym_trace` training equations:
  `e05d05abb4fa01c4a38cd3126ec9e09887fcd1c8fcc291b23341c586a30fa7e9`
- 96 source-disjoint `augmented_gsm8k` confirmation episodes, 222
  transactions:
  `854395449c78eaf65e07d6e428336045e8604dcd9d4a6bced130def97d1f9858`
- data report SHA-256:
  `dc85f6b833d48313451f4f4af27543d7ed1cbf8c57a14f9d876a6db04d802c95`
- unchanged FTA1 checkpoint SHA-256:
  `9321b78372d9926930d4de073d70e82c94e8360a69e09be695bab91b2e479f2d`

The exact-overlap audit finds 109 eligible confirmation rows with depth counts
`85/20/2/2`. Selection retains all 24 depth-three-or-greater rows and the first
72 hash-sorted depth-two rows. The final board has depth counts `72/20/2/2`
and operation counts 58 add, 88 subtract, and 76 multiply.

## Learned interface

The only newly trained component is a 183,043-parameter, two-layer,
bidirectional byte GRU with width 128 and no position embeddings. A lexical
scanner proposes three complete signed-integer mentions. The model assigns a
hard one-to-one `LHS/ARGUMENT/RHS` permutation over those complete mentions.

The frozen schedule is seed `2026080608`, 1,000 AdamW updates, batch 256,
learning rate 0.003 with cosine decay, and zero dropout. CPU training consumes
4,543,408 source bytes in 154.078 seconds. Training assignment/value accuracy
is 2,179/2,179. Checkpoint SHA-256 is
`c7ef7b4c0dd0b3738a9356bded7d9aa6f1eaff08da24677197388f3474c68764`;
model-state SHA-256 is
`ad5923dc78ed5c4462b90ed8b90b86bd32e8a308cc170adc5c578ea6db0d41cb`;
training-report SHA-256 is
`16350f099eebd4f21ad53277d78c7ed9b5af4562a0475b3abc155e2bce0857f8`.

## Confirmation result

The source-disjoint component gate is exact:

- learned whole-mention assignment: **222/222**;
- all-three-operation support: **222/222**;
- distinct immediate operation outcomes: **222/222**;
- initially wrong FTA1 top operation: **221/222 transactions** and **96/96
  episodes**.

The end-to-end source-sealed results are:

| Arm | Exact |
|---|---:|
| premature highest-support top-1 | 0/96 |
| equal-memory complete particles | 0/96 |
| no-evidence factorization | 96/96 abstain |
| posterior aggregation after evidence | 96/96 |
| **hard-evidence factorized DIVERGE** | **96/96** |

Full DIVERGE recovers **96/96** episodes whose initial top-1 assignment is
wrong. Exact state-group execution matches independent assignment enumeration
with and without evidence on all 96 episodes. Every depth and every operation
slice is exact.

## Causal and integrity controls

- shuffled evidence: 0/96;
- initial-state reset: 0/96;
- operand-semantic shift: 0/96;
- packet/query swap: 96/96 reject;
- post-seal source poison: 96/96 bit-invariant;
- wrong-source, wrong-step, wrong-mention, or wrong-value evidence accepted:
  zero;
- false commitments, malformed accepted packets, and overflow: zero.

Canonical complete-particle storage is 3,654,126 bytes versus 244,490 bytes
for factorized packet/execution/evidence/query receipts, a 14.946x ratio. The
equal-memory control funds one highest-ranked complete particle per episode.
Factorized execution performs 2,610 logical applications through 666 unique
state-group applications, a 3.919x sharing ratio. Peak surviving state groups
after each evidence step is one.

Evaluation SHA-256 is
`ad77a3b2d6b32cf2ad10727a6e85cfbcbc811d06f4ab2ff93c62269f2b22eb7f`.

## Decision and boundary

NFE1 passes the intended boundary: learned, source-disjoint, whole-mention
evidence packets can drive exact delayed version-space commitment without raw
source access after sealing. This materially extends TFS1 beyond assessor-
issued typed evidence.

The claim remains narrow. The lexical scanner still proposes numeric spans;
the operation domain is three engineered scalar functions; FTA1 supplies the
operation support; and the arithmetic verifier/executor is exact code. NFE1
does not establish general language reasoning or justify continuation
pretraining. Freeze this result with no seed, width, duration, renderer, or
loss variants. The next admissible boundary is one learned natural
program/evidence compiler with variables, predicates, and noncommuting stateful
updates, using the same source-sealed factorized runtime and explicit
single-state/particle controls.
