# R12 Episodic-Generator Constraint Result

## Decision

`reject_architecture_native_shohin_reasoning_retain_neurosymbolic_solver`

The candidate reaches 11/11 development queries, but adversarial audit rejects
promotion as Shohin reasoning. The actual learned system is a 232,065-
parameter record-direction classifier feeding exact regex parsing, exhaustive
127-word enumeration, host-side sealing, and a Python executor. The protected
Shohin checkpoint is not loaded or called.

The preregistered confirmation does not formally pass every promotion
condition. Deranging one support generator produces zero correct answers and
zero correct maps, but one of 11 altered episodes still seals a wrong packet;
the frozen confirmation required zero seals. Preserve this as a formal gate
miss rather than weakening the criterion after seeing the result.

The defensible classification is a bounded neuro-symbolic permutation solver
with a learned syntax-orientation frontend. It is not architecture-native
program induction and not a Shohin capability.

## Protected Base

- checkpoint: `train/flagship_out/ckpt_0300000.pt`
- protected parameters: 125,081,664
- SHA-256:
  `211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6`
- confirmation-time local/Newton hash match: yes
- continuation pretraining: held by explicit user instruction

The checkpoint was not modified, trained, loaded, or executed during this
experiment. Adding its parameter count to the conceptual ledger does not make
it part of the candidate.

## Mechanism

Each episode contains four opaque actions:

1. two complete support-generator transition tables; and
2. two sparse target actions composed from those support generators.

A 232,065-parameter byte reader learns record direction. Exact regex code
extracts cardinality, action equality, numeric states, record boundaries, and
query actions. Fixed code identifies complete supports by record count,
constructs all 127 syntactic words through depth six, applies a finite-
temperature match softmax, seals two target tables, and executes the query in
Python.

The report's zero search/solver counters do not describe the end-to-end
system: 127-word enumeration is exhaustive bounded search and the sealer plus
executor are symbolic host algorithms.

## Data and Audit

The confirmation board, seed `20260726`, contains 26 frozen episodes:

- 15 fitting episodes across cyclic, dihedral, and bitwise families;
- 11 development episodes covering unseen laws, deeper compositions,
  cardinality 16, held-out renderers, joint shifts, and a completely held-out
  random-permutation generator family;
- 52 unique target laws;
- zero training/development target-law overlap;
- zero raw target-map overlap;
- 22 development target-word instances, of which 14 overlap training;
- all four abstract length-two words present in both splits;
- 89 sparse target records, all inclusion-minimal;
- 99 hidden query transitions and zero target transitions visible at query
  time;
- 8/8 renderer orbits compiling to identical packets;
- 26/26 law swaps changing the answer;
- 16/16 eligible action-order reversals changing the answer; and
- 52/52 support keys deleted from deployed packets.

The updated mechanics audit receipt SHA-256 is
`b29101d13632f1406d51088909607e5a3aa0af043103e51fa8df62c3dd9a0e6d`.
It is not implementation-independent because it imports the same generator,
compiler, decoder, and executor family.

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

Every serialized treatment packet excludes literal source and support bytes.
This is not process-level deletion: source rows and parsed tensors remain live
during query execution and the packet contains a source-derived digest.

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
- actual trained candidate parameters: 232,065
- protected Shohin parameters participating in the runtime: 0
- conceptual ledger sum only: 125,313,729

## Pre-Hardening Canary

Job `704786`, seed `20260725`, first established 11/11 treatment exactness,
including 2/2 held-out random-permutation episodes. Its ambiguous posterior
could still accidentally seal 4/11 witness-deletion packets. That result is
retained as a mechanics measurement only. Consensus sealing and a regression
test corrected the defect before the confirmation run.

## Scientific Conclusion

The 11/11 score establishes that a learned renderer-direction classifier plus
a bounded symbolic permutation solver works on this board. It does not
establish that Shohin learned parsing, law induction, program construction,
constraint conjunction, source-deleted execution, or reasoning.

Do not retain this as Shohin's best reasoning baseline. Keep it as a
neuro-symbolic mechanics control. The next admissible claim-bearing path must
load Shohin, consume raw tokens, eliminate semantic host parsing/search/
execution, enforce process-level deletion, hold out abstract programs, and
cross genuinely different ontologies. Full hostile findings are in
`R12_EPISODIC_GENERATOR_ADVERSARIAL_AUDIT.md`.
