# DIVERGE-CCR1: Counterfactual Candidate Referent

Status: frozen after the oracle-typed PL1 pass and before CCR1 implementation,
data materialization, training, or model result.

## 1. Capability hypothesis

SOT1 assigns referent semantics independently by stage and catastrophically
inverts one renderer. SRP1 shares one pair scorer across stages and reaches
`753/768`, but gains only 11 transactions over frozen SOT1, misses one
renderer, and loses three complete episodes through evidence errors.

Both models encode the raw sentence once and pool hidden states at the two
entity mentions. The candidate identity, lexical surface, and role context are
therefore entangled before the hard two-way assignment.

CCR1 changes that representation boundary:

> Encode one counterfactual sentence per candidate after replacing that
> candidate by a learned SELF marker and the other candidate by a learned
> OTHER marker. Score SELF with one shared encoder, then commit the only legal
> complete role permutation.

Swapping candidate groups swaps the two canonicalized inputs by construction.
Names cannot drive the score because their embeddings are replaced before the
encoder. Word order and role predicates remain visible. This is not an SRP1
width, renderer, duration, warm-start, or optimizer retry.

CCR1 is an interface qualification step, not the project novelty claim. It
does not reopen the earlier unrestricted whole-mention span-quotient family.
The exact symbol groups are already supplied by the qualified controlled
source interface; CCR1 tests only semantic role grounding.

## 2. Frozen owner

The referent owner has:

- byte width 192;
- one learned SELF vector and one learned OTHER vector;
- one shared two-layer bidirectional GRU;
- LayerNorm;
- one shared scalar scorer over CLS, global mean, SELF mean, and their
  pairwise difference/product; and
- one exact two-way permutation projection.

For candidate `i`, every byte in group `i` receives SELF, every byte in group
`1-i` receives OTHER, and all remaining bytes keep their ordinary byte
embedding. The two candidate-conditioned sequences are encoded independently
with tied weights. The scorer emits `s_0,s_1`; `s_0-s_1` defines the hard
`TARGET/DISTRACTOR` permutation.

Qualified TOL3 WORLD and NVE1 numeric EVIDENCE owners remain immutable and
hash-checked. CCR1 is the only plastic owner. It starts from the frozen random
seed rather than NVE1 or SRP1 weights.

## 3. Frozen fit

- seed: `2026080623`;
- exactly 1,000 updates;
- each update: 128 immutable natural-evidence rows plus 128 immutable
  natural-query rows;
- AdamW, LR `3e-3` cosine to zero, betas `(0.9,0.95)`, weight decay `0.01`,
  gradient clip `1.0`;
- equal-weight two-way role cross entropy;
- no answer, operation, state, comparator, confirmation, or PL1 label enters
  fit.

The opened SRP1 board is development-only. It can reject CCR1 before fresh
confirmation but cannot promote it.

## 4. Fresh confirmation

One 256-episode board at seed `2026080622` uses a new 32-name bank, new program
identities, and six query renderers composed from training-supported semantic
anchors in unseen orders. Every mode crosses every renderer. Evidence keeps
the three qualified held layouts so the numeric owner remains a protected
control. Exact source/query/identity/entity overlap with SRP1 and training is
zero. No model score enters board selection.

Frozen SRP1 is evaluated on the same fresh board. Controls include complete
SELF/OTHER marker swap, marker deletion, entity-name permutation, packet/query
swap, shuffled evidence, state reset, operation shift, and post-seal source
poison.

## 5. Pass/kill gate

Development admission requires at least `765/768` query assignments,
`3,070/3,072` evidence assignments, and `255/256` sealed episodes on the
opened SRP1 board.

Fresh promotion then requires every condition:

1. WORLD `256/256`, EVIDENCE at least `3,070/3,072`, and sealed episodes at
   least `255/256`;
2. QUERY at least `765/768`, every mode at least `254/256`, and every renderer
   at least `127/128`;
3. sensitive answers, extensional parity, no-evidence abstention, invariant
   answers, and partial-evidence underdetermined abstention each at least
   `254/256`;
4. CCR1 improves frozen SRP1 by at least four exact query transactions on the
   same board and has no worse evidence count;
5. marker swap loses at least 90 points, marker deletion loses at least 49
   points, and entity-name permutation changes no assignment after masks are
   permuted consistently;
6. shuffled evidence, state reset, and operation shift each lose at least 50
   points; packet/query swaps all reject and post-seal poison is invariant;
7. zero invalid transaction, false commitment, malformed packet, gold
   deletion, overflow, or protected-owner hash change.

A miss closes CCR1 without width, layer, marker, seed, duration, renderer,
warm-start, loss, or optimizer variants. A pass qualifies one natural PL1
integration; it does not authorize continuation pretraining or establish
open-domain reasoning.
