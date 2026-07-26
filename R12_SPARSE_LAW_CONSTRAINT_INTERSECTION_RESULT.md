# R12 Sparse-Law Constraint-Intersection Result

## Decision

The architecture demonstrates exact bounded sparse-law identification, but
misses one preregistered promotion criterion because that criterion was
mathematically impossible on the frozen board.

Newton job `704761` completed 2,000 updates on one H100 in 3 minutes 7
seconds. Treatment reached:

- 2,400/2,400 = 100% development transition states;
- 60/60 complete hash-disjoint unseen maps;
- 60/60 exact hidden queries;
- 60/60 source-free deployed packets; and
- zero training/development action-map overlap.

## Same-Weight Controls

| Arm | Transition accuracy | Complete maps | Exact queries |
|---|---:|---:|---:|
| Constraint intersection | 100.0000% | 60/60 | 60/60 |
| Direction negated | 36.0833% | 6/60 | 20/60 |
| Observation targets shifted | 8.1667% | 0/60 | 1/60 |
| Observations zeroed | 8.1667% | 0/60 | 0/60 |

The treatment beats the strongest control by 40 exact queries. The frozen
preregistration required 45. Independent exact analysis shows that reversing
every observed permutation and executing the original query has exactly
20/60 answer collisions on this board. Therefore no perfect treatment could
exceed the direction-negated control by more than 40. The result must not be
reported as passing every frozen gate, but the control does not contradict
the capability result.

## Mechanism

The learned byte encoder predicts source-versus-target direction. Every
observation then remains a separate compatibility factor over a fixed
finite-domain program bank:

`V(D) = intersection_i {p : p(source_i) = target_i}`.

A dense tensor energy evaluates all factors in parallel. The unique surviving
map is sealed, source bytes are discarded, and a fixed categorical executor
answers the late query. Candidate inference calls no host parser, callback,
solver, search routine, oracle, or verifier.

This fixes the specific failure of direct attention, generic generators, and
supervised microcode: those candidates averaged the individually necessary
records before predicting a program.

## Receipts

- learned parameters: 232,065
- conceptual complete system: 125,313,729
- report SHA-256:
  `22d22d8ac9079ee1722ce971c50d2307c5ad1d9132d49c329ef779f01adb2ede`
- model SHA-256:
  `b54fc0d2c113fff07a5a70629b5499ecc8412b6ecc2732ef3728359a2dc3de89`
- optimizer updates: 2,000
- training rows: 3,300
- development rows: 60
- candidate-time oracle/search/verifier calls: 0/0/0

## Boundary And Successor

Constraint intersection and the complete operation library are fixed
algorithmic priors. Dense parallel evaluation is semantically equivalent to
exhaustive version-space elimination. This is architecture-native execution,
but it is not learned ontology discovery or general reasoning.

The only justified successor removes the global program bank. A fresh episode
must provide unfamiliar complete support generators; the same architecture
must construct an episode-local closure, infer sparse target programs, seal
them, delete source, and execute. Development must include unseen generator
families and longer target programs with matched support-shuffle, witness-
deletion, contradiction, direction, and record-order controls.
