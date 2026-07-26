# R12 Episodic-Generator Constraint Result

## Decision

`bounded_program_induction_demonstrated_but_fail_closed_promotion_withheld`

Shohin's reasoning sidecar now demonstrates exact, source-deleted induction of
sparse unseen action laws from unfamiliar generators supplied inside each
episode. This is the first retained result that constructs its finite
hypothesis space from the episode rather than selecting from a fixed global
law library.

The preregistered confirmation does not formally pass every promotion
condition. Deranging one support generator produces zero correct answers and
zero correct maps, but one of 11 altered episodes still seals a wrong packet;
the frozen confirmation required zero seals. Preserve this as a formal gate
miss rather than weakening the criterion after seeing the result.

This is a bounded program-induction result, not demonstrated general language
reasoning. The ontology is still a finite permutation machine; each episode
provides two complete support-generator tables; closure depth is at most six;
constraint intersection, consensus sealing, and categorical execution are
fixed algorithms; and the mechanism remains a sidecar rather than a learned
capability inside the Shohin transformer trunk.

## Protected Base

- checkpoint: `train/flagship_out/ckpt_0300000.pt`
- protected parameters: 125,081,664
- SHA-256:
  `211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6`
- confirmation-time local/Newton hash match: yes
- continuation pretraining: held by explicit user instruction

The checkpoint was not modified or trained during this experiment.

## Mechanism

Each episode contains four opaque actions:

1. two complete support-generator transition tables; and
2. two sparse target actions composed from those support generators.

A 232,065-parameter byte reader learns record direction. Complete actions are
identified as episode-local supports. The compiler constructs the 127
syntactic words over two generators through depth six, keeps each sparse
target record as a separate logical factor, and intersects the surviving
programs. A map seals only when every surviving program agrees on every state
transition. The deployed packet contains only the two inferred target maps;
raw source bytes and support-generator keys are deleted before query
execution.

There is no global operation-law bank, family label, target program label,
candidate-time oracle, search call, solver call, or verifier call.

## Data and Audit

The confirmation board, seed `20260726`, contains 26 frozen episodes:

- 15 fitting episodes across cyclic, dihedral, and bitwise families;
- 11 development episodes covering unseen laws, deeper compositions,
  cardinality 16, held-out renderers, joint shifts, and a completely held-out
  random-permutation generator family;
- 52 unique target laws;
- zero training/development target-law overlap;
- zero raw target-map overlap;
- 89 sparse target records, all inclusion-minimal;
- 99 hidden query transitions and zero target transitions visible at query
  time;
- 8/8 renderer orbits compiling to identical packets;
- 26/26 law swaps changing the answer;
- 16/16 eligible action-order reversals changing the answer; and
- 52/52 support keys deleted from deployed packets.

The independent audit receipt SHA-256 is
`596039c38bbaecdd646ef8bea78c7263ea61b43bba13b7d1b067e36d35ae2e02`.

Training used 375 rows:

- 15 frozen fitting rows;
- 60 counterfactual direction rows; and
- 300 auxiliary rows.

One model received 1,000 optimizer updates. All controls reused the same
weights with zero additional updates. Training loss fell from 7.5843468 to
0.0000001813.

## Confirmation

Newton job `704792` completed on one H100 in 5 minutes 38 seconds.

| Arm | State accuracy | Complete maps | Exact queries | Invalid packets |
|---|---:|---:|---:|---:|
| Treatment | 100.0000% | 11/11 | 11/11 | 0 |
| Record order reversed | 100.0000% | 11/11 | 11/11 | 0 |
| Support order recoded | 100.0000% | 11/11 | 11/11 | 0 |
| Direction negated | 33.5417% | 2/11 | 2/11 | 0 |
| Observations shifted | 11.4583% | 0/11 | 1/11 | 0 |
| Observations zeroed | 9.1667% | 0/11 | 0/11 | 11 |
| Support semantics deranged | 51.8750% | 0/11 | 0/11 | 10 |
| Necessary target witness deleted | 54.5833% | 0/11 | 0/11 | 11 |

Treatment is exact in every development cell and family:

- cyclic: 3/3;
- dihedral: 3/3;
- bitwise: 3/3;
- held-out random permutation: 2/2;
- composition: 3/3;
- renderer: 3/3;
- topology: 3/3;
- unseen law: 1/1; and
- joint shift: 1/1.

Every treatment packet passes source and support deletion. Record reordering
and consistent opaque support recoding are exactly invariant. Removing one
necessary target witness forces all 11 packets to fail closed.

The deranged-support arm is causally decisive for accuracy but misses the
strong fail-closed criterion: 10/11 packets reject, while one internally
consistent but wrong altered episode seals. The treatment capability is
therefore demonstrated, but promotion as a universally fail-closed mechanism
is withheld.

An independent exact identifiability audit explains this miss. Under the
perturbed abstract support/observation interface, six episodes are ambiguous,
four are contradictory, and the held-out random-permutation law episode is a
coherent alternate world: each target has exactly one surviving program and
neither is the original target map. Therefore a constraint compiler operating
only on that abstract interface cannot reject all deranged episodes without
also rejecting evidence-consistent worlds. Raw-source redundancy remains
available and could detect the injected internal corruption, but that would
measure a parser checksum rather than law induction. The frozen formal miss
is preserved.

The identifiability report is
`artifacts/r12/episodic_generator_constraint/deranged_identifiability_seed20260726.json`;
its receipt SHA-256 is
`4e48a77c7f3ea9995a36094467585fd99ff2164abc4c89fa0761c3950eb5d8af`
and file SHA-256 is
`a58f64bec1f61eb8e9e611e45e525feac54a95b93b211dbb2c1d437800e30292`.

## Receipts

- report:
  `artifacts/r12/episodic_generator_constraint/seed20260726.json`
- report SHA-256:
  `226a36d9156101617b769f698550eb51ebec57a8ffa01464bdd7a64d8805caad`
- model:
  `artifacts/r12/episodic_generator_constraint/seed20260726.pt`
- model SHA-256:
  `37a28b0f9ab401d369e481f99c62393dee99125250dbd6ce1ba37c07e360512f`
- focused tests: 14 passed
- Ruff: clean
- byte compilation: clean
- learned compiler parameters: 232,065
- complete conceptual parameters: 125,313,729
- remaining headroom under 200M: 74,686,271

## Pre-Hardening Canary

Job `704786`, seed `20260725`, first established 11/11 treatment exactness,
including 2/2 held-out random-permutation episodes. Its ambiguous posterior
could still accidentally seal 4/11 witness-deletion packets. That result is
retained as a mechanics measurement only. Consensus sealing and a regression
test corrected the defect before the confirmation run.

## Scientific Conclusion

The campaign no longer supports the statement that Shohin has no sparse
unseen-law induction. It has exact bounded evidence for:

1. reading unfamiliar demonstrations;
2. constructing a temporary program space from episode-local generators;
3. intersecting independently necessary constraints;
4. refusing to guess when evidence is insufficient;
5. deleting the demonstrations; and
6. executing hidden compositions from the sealed result.

The result does not establish genuine general reasoning because the generic
algorithms and finite ontology are supplied by the architecture. The learned
component discovers direction semantics, while closure, logical conjunction,
consensus, and execution are engineered inductive biases. The next scientific
boundary is not another parameter sweep on this board. It is whether the same
constraint-carrying architecture can learn new object types, relations,
operators, and executors across genuinely different task families.

Under the present usage constraint, stop here. Preserve this mechanism as the
best reasoning baseline, preserve the formal fail-closed miss, keep
continuation pretraining held, and do not claim that the 125M language model
itself has acquired general reasoning.
