# DIVERGE-RRG1 Result: Query Semantics Transfer, Shared Evidence Does Not

Status: closed negative after one frozen fit, one opened-board admission, and
one read-only attribution. Confirmation was never opened.

## Frozen execution

- implementation commits: `8246fcb`, `f23d032`;
- H100 job: `744379` on `evc22`;
- schedule: exactly 2,000 updates, 128 EVIDENCE plus 128 QUERY rows per update;
- trainable/total parameters: 733,249 / 1,196,434;
- fit elapsed: 85.681 seconds;
- checkpoint SHA-256:
  `05c7b6c7ee68e14133231d572b8c66aced5e3a72a6ba25827498560f723336ae`;
- training-report SHA-256:
  `0e0b681f3bd65c876fbdbc094de3385de6369352f934fac75ebf540c2cff57db`.

The immutable corpus contains 100,000 EVIDENCE and 100,000 QUERY rows. Every
one of 50,000 semantic items per stage has TARGET-first and DISTRACTOR-first
realizations. Every family/form/order cell is exact and all overlap receipts
are zero. Data/report SHA-256 values are `56f0e6e5...d67fe`,
`2d325c86...78c1c`, and `7439147d...36b3`.

Training reaches 100,000/100,000 exact in both stages and 50,000/50,000 exact
counterfactual pairs in both stages. WORLD and numeric-EVIDENCE owner hashes
remain bit-identical.

## Development result

The opened SRP1 board rejects promotion:

| Gate | Result | Floor |
|---|---:|---:|
| QUERY | **768/768** | 765 |
| EVIDENCE | **2,048/3,072** | 3,070 |
| Fully sealed episodes | **0/256** | 255 |
| WORLD | **256/256** | 256 |

Every QUERY mode is 256/256 and every QUERY renderer is 128/128. Every episode
fails sealing because exactly one of the three evidence renderers is inverted.
Development evaluation SHA-256 is
`6ca42dabb4da119e26d7159c340750a4f01161efe65d634ceba0b2f865fd96db`.

## Read-only attribution

Diagnostic commit `7038868` changes no weight, threshold, source, split, or
prediction. It scores the frozen checkpoint on every established NVE1
training and confirmation surface:

| Surface | Exact |
|---|---:|
| confirmation renderer 0 | **0/3,072** |
| confirmation renderers 1/2 | **3,072/3,072 each** |
| legacy training renderer 0 | **0/3,072** |
| legacy training renderers 1--5 | **3,072/3,072 each** |

Renderer 0's mean signed margin is -59.806 on confirmation and -2.774 on the
legacy training surface. All passing surfaces have positive margins. Arbitrary
entity renaming changes zero assignments and produces bit-identical logits
with maximum absolute difference 0.0. Owner hashes remain exact. Diagnostic
SHA-256 is
`0095b8ca08ed14e2e26534b00eb6da0a4fb1ef10305c1514d458f5a91e865409`.

## Decision

RRG1 disproves the universal shared REFERENT owner. The representation removes
identity and order shortcuts and produces a qualified natural QUERY owner, but
EVIDENCE semantics remain stage/syntax dependent. Do not run RRG1 width,
duration, seed, grammar, renderer, loss, marker, or optimizer variants.

Preserve the RRG1 QUERY owner as a qualified component. The structurally
different successor is STI1: immutable NVE1 owns all EVIDENCE bindings and the
immutable RRG1 owner handles QUERY, joined only by typed transactions. STI1
uses no new learning and must independently pass development before opening
confirmation.

