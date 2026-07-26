# R12 Endogenous Typed Theory Reactor Preregistration

## Status

Frozen successor protocol; no implementation or capability claim.

The user pretraining hold remains active. This protocol may train only isolated
reasoning components or adapters against the immutable step-300k checkpoint.

## Objective

Test whether one actual-Shohin, raw-token, source-deleted architecture can
infer and execute a previously unseen typed theory rather than operate inside
a supplied finite-machine ontology.

The candidate must induce an anonymous episode object

```text
Theta = (Q, tau, R, F, Gamma, Omega)
```

where:

- `Q` binds physical mentions to episode-local objects;
- `tau` assigns latent object types;
- `R` defines relation symbols, arities, argument roles, and initial facts;
- `F` defines operator preconditions and graph effects;
- `Gamma` defines sequential, synchronous, saturation, branching, and halt
  semantics; and
- `Omega` defines observers available to a late query.

## Behavioral Identifiability Gate

For evidence `D`, bounded theory class `H`, and admissible late challenges
`Q`, define:

```text
V(D) = {Theta in H : Theta satisfies D} / isomorphism
```

Two theories are behaviorally equivalent when they answer every admissible
late challenge identically. Exact deterministic reasoning is identifiable if
and only if the behavioral quotient of `V(D)` has one class.

Every scored episode must receive an independent exact version-space receipt:

- singleton behavioral class: candidate must commit and answer;
- multiple behavioral classes: candidate must abstain;
- empty version space: candidate must reject; and
- coherent alternate singleton: candidate must commit to the alternate
  world's behavior, not reject it.

## Three Ontologies

Three leave-one-ontology-out folds are mandatory.

| Ontology | Hidden structure | Required execution |
|---|---|---|
| Horn closure | objects, typed predicates, asymmetric roles | monotone least fixed point |
| Typed term rewriting | constructors, variables, ordered child roles | deletion, replacement, branching normal form |
| Guarded resource process | places, resource types, multiplicities | guarded consume/produce, sequence, deadlock/halt |

These families may share only the generic typed-transaction substrate. No
family identifier, family head, domain opcode, or host semantic callback may
enter the candidate.

## Architecture

### Raw-token compiler

- Load and execute the protected Shohin checkpoint.
- Accept only tokenizer output and masks from raw source text.
- Use one shared compiler/adaptor across all ontologies and renderers.
- Emit an immutable anonymous typed-theory packet.
- Do not receive exact spans, numbers, entity equality, relation roles,
  family labels, program graphs, schedules, answers, or assessor products.

### Generic reactor

One recurrent controller emits only:

```text
ALLOC WRITE CLEAR LINK UNLINK SET_ROOT COMMIT HALT
```

A rule-blind committer may enforce bounds, pointer validity, type shape,
capacity, and transaction atomicity. It may not match a semantic rule, choose
a redex, compute closure, perform arithmetic, repair a transaction, select an
answer, or retry after assessor feedback.

### Late-query reader

The reader receives only the committed terminal object and raw late-query
tokens. It cannot access source tokens, compiler residuals, KV state, parser
state, execution trajectory, or assessor data.

## Four-Process Custody

1. **Compiler process:** reads source and writes an immutable packet.
2. **Executor process:** starts fresh, receives only packet and command stream,
   commits terminal state, then exits.
3. **Query process:** starts fresh, receives terminal state and raw late query,
   writes an answer or abstention, then exits.
4. **Assessor process:** starts only after all candidate processes exit and
   uses an independent implementation.

Packets may contain no source-derived digest, source offsets, raw names,
residuals, hidden caches, answer labels, or executable host callbacks.
Post-seal source poisoning must be bit-invariant.

## Smallest Decisive Board

- 3 leave-one-ontology-out folds;
- 8 independently generated held-out theories per fold;
- 7 versions per theory:
  base, alpha/reorder, alias split, relation reification, type twin,
  execution-semantics twin, and ambiguity-deleted twin;
- 16 independently generated post-seal challenges per version;
- 2,688 primary scored executions;
- at most 6 objects, 3 inferred types, 3 relations of arity at most 3,
  3 opaque operators, depth 8, and branch width 2;
- at least 4 renderers, with one fully held out;
- disjoint canonical and isomorphism hashes across splits; and
- no abstract operator/effect/control program overlap between fitting and the
  held-out ontology.

Hybrid confirmation must include at least:

- arithmetic index selecting a rewrite location;
- relation result selecting a resource operator; and
- resource state controlling a Horn query.

## Matched Controls

1. actual Shohin trunk;
2. zeroed trunk residuals;
3. parameter-permuted trunk;
4. example-swapped frozen trunk;
5. equal-parameter generic recurrent classifier;
6. fixed-ontology typed reactor;
7. family-routed executors with matched aggregate parameters;
8. random-label control;
9. ambiguous, contradictory, and coherent-alternate evidence; and
10. independent type, role, effect, control-semantic, state, order, and query
    transplants.

Every learned arm must share update count, data access, initialization lineage,
and parameter budget where structurally possible.

## Promotion Gates

- actual checkpoint loaded and hash-verified;
- fewer than 200,000,000 unique participating parameters;
- actual Shohin treatment beats every zeroed/randomized/swapped-trunk control;
- raw-token end-to-end custody passes with no semantic host parser;
- 100% independent oracle agreement and packet-schema validation;
- at least 95% exactness on identifiable cases in every held-out ontology;
- 100% abstention on behaviorally ambiguous cases;
- 100% rejection on contradictory cases;
- 100% coherent-alternate-world behavior;
- 100% alpha/reorder/alias/reification invariance after canonical alignment;
- at least 95% execution-twin and noncongruent-twin separation;
- at least 20 points over every qualified matched learned control;
- all three leave-one-ontology-out folds pass individually; and
- hybrid confirmation reaches at least 85%.

A pass establishes bounded cross-ontology typed-theory induction. It does not
establish unrestricted natural-language reasoning. Natural-language and public
benchmark promotion remains governed by G4 in
`R12_GENERAL_REASONING_GATE.md`.
