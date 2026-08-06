# DIVERGE-FTA1-AC1: Autonomous Contradiction Replay

Status: frozen before implementation result on 2026-08-06.

## Hypothesis

FTA1 now compiles every held source step into an exact typed packet containing
the claimed left state, operation, arguments, and claimed right state. The
algebraic transaction layer can independently compute the right state. AC1
tests whether one source-sealed contradiction circuit can locate the first
invalid transaction, commit once, and replay all remaining operations in one
coherent state lineage without answer labels or an oracle error index.

The runtime starts from the first compiled left state. Before commitment, it
compares the computed successor with the source-owned claimed successor. The
first mismatch is the discrete fault line. After that event, every later right
claim is ignored and only computed state is carried forward. This prevents the
trivial strategy of always restarting at step one: exact first-error selection
remains a required output and promotion condition.

## Frozen inputs

- FTA1 checkpoint SHA-256
  `9321b78372d9926930d4de073d70e82c94e8360a69e09be695bab91b2e479f2d`;
- passing FTA1 component-gate SHA-256
  `dd5f0d261fdf88f18bbcf7aee6574d46f5d48a5d31e438bfcf0cc8181f3dfa72`;
- unchanged 480-row CRP1 OOD board SHA-256
  `db0bde0c22afe3d25f4f1f578249bf67156f4115d38a094c07f1ea36f6be6849`;
- unchanged FTA1 compiler, typed packets, transaction algebra, and one seed;
- no additional training and no CRP1 language-generator or oracle selection.

## Frozen gate

All conditions are conjunctive:

- compiler role, operation, LHS, RHS, and arguments each >=99%;
- exact first-conflict selection, terminal, and complete trajectory each
  >=432/480;
- each of scalar/register/symbolic has >=136/160 exact selections, terminals,
  and complete trajectories;
- zero invalid packets;
- trusting source claims or ignoring the first conflict each costs >=384
  answers;
- initial-state packet swap and operation shift each cost >=240 answers.

A pass authorizes one natural verified-trace transfer gate. It does not by
itself establish open-domain language reasoning: the operation vocabulary and
typed algebra are still closed and engineered. A failure closes this exact
autonomous composition without selector, threshold, or control repair.
