# DIVERGE-CCR1 Result: Candidate-Relative Encoding Is Not Semantic Grounding

Status: closed negative at the opened development gate. Sealed confirmation
was never evaluated.

## Frozen execution

- implementation commits: `ec6d2f3`, then pre-result audit-alias correction
  `c2ee33f`;
- training job: `744334` on one Newton H100;
- development evaluation: `744340`;
- read-only attribution: `744352`;
- training seed/updates: `2026080623`, exactly 1,000;
- trainable/total parameters: `546,433 / 1,009,618`;
- charged source bytes: `21,349,174` in `56.415` training seconds;
- checkpoint SHA-256:
  `dd5287181308b3c8aae3a21cba8a12f873d5294f63158313a4bd6eececa3dc1d`;
- training-report SHA-256:
  `2695cd21a8af40cc914e721f7e23d08c13cb116a4dcd359ba9b96c1bf8f1997b`.

All 50,000 evidence and 50,000 query training assignments fit exactly. WORLD
and numeric-EVIDENCE owner hashes remained bit-identical.

The fresh 256-episode confirmation board was generated before training from
commit `ec6d2f3`. It contains 3,072 evidence items, 768 balanced queries, and
1,048,576 represented worlds. Source, query, identity, and entity overlap
with training and SRP1 are all zero. Board/report SHA-256 values are
`29923706...1ebf7` and `77ce453b...042d`. Because development failed, this
board remained unopened.

## Development result

CCR1 reaches:

| Measure | Result | Required |
|---|---:|---:|
| Evidence assignment | 2,097 / 3,072 | >= 3,070 |
| Query assignment | 695 / 768 | >= 765 |
| Fully sealed episodes | 0 / 256 | >= 255 |

Query renderer exact counts are `55, 128, 128, 128, 128, 128` of 128.
Development evaluation SHA-256 is
`c87a51ed08a103a73a63f6446554d9a6cea57dee3fe2fb975a785ab92cf85d5f`.
The evaluation wrapper exited nonzero, so dependency-held confirmation job
`744343` became unsatisfiable and was canceled.

## Read-only attribution

The one allowed attribution crosses every development item with every held
renderer:

- EVIDENCE renderer 0: `159/3,072`; renderers 1 and 2: `3,072/3,072` each;
- QUERY renderer 0: `351/768`; renderers 1--5: `768/768` each;
- marker swap changes query accuracy only from `695/768` to `683/768`, so the
  SELF/OTHER marker identity is not the causal decision variable;
- canonical entity renaming changes 122 of 3,840 assignments;
- renderer-0 signed margins are negative on average (`-16.581` evidence,
  `-0.834` query), while every passing renderer has a large positive margin.

Diagnostic SHA-256 is
`86eae09bde4123ef0edf4552753d5dc5dc5f97dfa170e61ffbbacf46336b7178`.

An independent data audit exposes the identifiability failure. In both
50,000-row source sets, every renderer has exactly one role order: even
renderers always label first mention TARGET and odd renderers always label it
DISTRACTOR. A model can therefore fit every training label by recognizing
renderer/order without learning the semantic relation.

## Decision

Close CCR1 without width, marker, duration, seed, renderer, warm-start, loss,
or optimizer variants. Independent candidate encoding does not force
candidate-relative semantics and retains mention-length leakage. It does not
qualify raw-language PL1.

The successor must remove the confound before fitting and change the
representation boundary: collapse every full mention to one anonymous,
length-free token; encode one sentence once; jointly match two explicit role
slots to the two mention contexts; and train on paired role orders within each
lexical family. That successor is DIVERGE-RRG1.
