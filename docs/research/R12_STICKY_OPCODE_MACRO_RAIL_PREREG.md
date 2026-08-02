# R12 Sticky Opcode-Macro Rail Preregistration

**Status:** implementation complete; GPU launch withheld until the fixed
syntax-graph 5k endpoint is read.

**Date:** 2026-08-02 EDT

**Claim boundary:** architecture diagnostic only. This experiment cannot be
called native reasoning unless the unchanged source-deleted evaluator improves
both strict WORLD and strict COMMAND causal pairs and the result replicates.

## Causal hypothesis

The parallel syntax-graph compiler can learn locally plausible transaction
fields but often combines them into an incoherent program. The full-corpus
audit identifies a lower-entropy reusable variable: the complete opcode
skeleton. Training contains 2,530 opcode sequences, and those sequences cover
99.275% of development instances. In contrast, exact and structural complete
programs cover only 60.07% and 77.99% of development instances.

The treatment therefore selects one complete opcode skeleton once per episode
and keeps that identity sticky across all transaction steps. Source, target,
relation, type, and value operands remain dynamically predicted from the exact
public syntax graph and the autonomous initial typed state. This tests whether
whole-program coherence, rather than width or another fieldwise loss, is the
missing architectural variable.

## Frozen evidence and registry

- Audit schema: `r12-ettr-program-template-audit-v2`.
- Registry artifact:
  `artifacts/r12/ettr_program_template_audit_full_e868d3c_r3/report.json`.
- File SHA-256:
  `03fc92829bc4a1c9f9e8381953ac506e04afeef746871a60ebfca1e482cbafcc`.
- Payload SHA-256:
  `d58185b4a5c7b28e54cd9497215dd8d5f0e52f7339a968f10facbc6669497b4b`.
- Registry population: 2,530 distinct train-only opcode sequences.
- Development instance coverage: 99.275%.
- The registry is generated from admitted training traces only. Development
  rows are used solely to report coverage and evaluate held-out behavior.

The training and evaluation processes must load the registry through the
hash-bound loader. The exact registry bytes are copied into each output and
included in `SHA256SUMS`. A missing, mutated, noncanonical, out-of-range, or
internally inconsistent registry fails closed.

## Architecture treatment

Starting from contract-v8 exact syntax graphs:

1. Pool the encoded COMMAND graph and autonomous initial state.
2. Predict a categorical distribution over the 2,530 frozen opcode skeletons.
3. Make one straight-through hard selection per episode.
4. Broadcast the selected program embedding to all parallel step queries.
5. Use the selected complete template as the hard applied opcode sequence.
6. Predict source, target, relation, type, and value operands dynamically with
   the existing syntax-graph-conditioned heads.
7. Apply the resulting schedule only through the fixed audited transaction
   algebra.

The selector receives an explicit whole-sequence categorical loss in addition
to the existing field losses. This prevents per-step opcode marginals from
forming a hybrid sequence that belongs to no selected program.

The treatment adds 1,942,873 parameters to the 169,421,167-parameter contract-
v8 system, for 171,364,040 total parameters. Parameter count is not an
advancement metric.

## Prohibited information

At inference the selector and operand heads may receive only:

- frozen Shohin hidden states for the initial packet and COMMAND bytes;
- the autonomous initial typed state;
- public syntax-graph topology and opaque-token equality; and
- the frozen train-only opcode registry.

They may not receive QUERY bytes, answer labels, terminal targets, oracle
programs, development-derived templates, candidate scores, a host semantic
solver, or teacher-forced intermediate states. Source bytes are deleted under
the existing evaluator boundary before late query scoring.

## Fixed experiment

The first gate uses the same data population, architecture seed 31, data seed
11, start position 0, learning rate `3e-4`, evaluation batch count 32, and
1,000-update budget as the completed contract-v8 syntax-graph endpoint. The
matched control is contract v8 without a registry. A longer treatment is not
authorized by a lower loss alone.

The first report must include:

- selector exact-class accuracy and candidate-use entropy;
- opcode, source, target, relation, type, value, and joint schedule accuracy;
- oracle-initial and autonomous exact terminal packets;
- autonomous factual top-1;
- strict, margin-1, and intervention-DID WORLD and COMMAND paired gates;
- unknown-template development rows as a separate slice; and
- complete artifact and contract hashes.

## Advancement rule

The treatment advances beyond 1,000 updates only if it beats the matched
contract-v8 endpoint on a coherence metric without regressing factual behavior,
or crosses at least one previously zero strict causal axis. It becomes a native
reasoning candidate only after all of the following:

1. strict WORLD and strict COMMAND both improve on the unchanged evaluator;
2. the gain appears on at least two held-out population orderings;
3. the gain replicates across at least two architecture seeds;
4. source deletion and all custody hashes pass; and
5. matched-compute contract-v8 controls do not explain the gain.

Best-of-K selection, oracle initial state, oracle programs, registry membership,
training accuracy, and exact packets alone are diagnostics and cannot satisfy
this rule.

## Grokking interpretation

The concurrent 1k/5k/15k contract-v8 trajectory tests two distinct hypotheses:

- **Classical grokking:** training behavior saturates first, followed by a
  delayed held-out causal/generalization jump.
- **Discrete coherence transition:** local categorical probabilities improve
  gradually, but exact schedules or packets jump when all coupled hard choices
  cross their argmax thresholds.

Only the first is evidence of delayed generalization. Neither counts as native
reasoning unless both source-deleted causal axes improve and replicate.
