# DIVERGE-SOT1: Stage-Owned Epistemic Transactions

Status: frozen before SOT1 implementation, data materialization, training, or
neural result on 2026-08-06.

## 1. Decision

IEM1 disproved the universal-owner integration hypothesis. One shared encoder
fit the evidence and query training interfaces but lost mandatory source
semantics: `SET` transferred at 0/2 and invalidated all 256 programs. A
read-only splice restored the immutable TOL3 source owner and immediately
recovered 3,072/3,072 evidence receipts, 256/256 sealed episodes, and 256/256
sensitive answers. The IEM1 query owner remained incomplete at 280/768 exact
natural query transactions.

DIVERGE-SOT1 tests a materially different architecture:

> Give each semantic stage an isolated parameter owner. Owners exchange only
> typed, provenance-bound transactions through a small epistemic microkernel.
> A parameter update can alter one owner but cannot silently rewrite another
> owner's transaction vocabulary.

SOT1 is not an IEM1 transport, width, loss, duration, seed, or renderer repair.
IEM1 remains closed. SOT1 does not authorize continuation pretraining.

## 2. Architecture

### Stage owners

One `StageOwnedEpistemicMachine` checkpoint contains three disjoint neural
owners and an owner manifest:

1. **WORLD source owner:** the exact frozen TOL3 local semantic anchor at
   checkpoint SHA-256
   `b8b9dfe54b7ab4a31a74739625b8650fa4ee93a41221ab5d82610ebc1c030328`.
   It owns declaration, update, predicate, swap, and structural query tokens.
2. **EVIDENCE owner:** the exact frozen NVE1 whole-mention compiler at
   checkpoint SHA-256
   `1610815471c695b0d2d198922dd99369e1f45a5dabc1b1c5d8e986b30fd200ff`.
   It owns `STEP/VALUE` and `TARGET/DISTRACTOR` evidence assignments.
3. **QUERY owner:** one fresh position-free two-layer bidirectional byte GRU,
   width 192 and zero dropout, with one whole-group permutation head. It owns
   only the hard `TARGET/DISTRACTOR` query assignment. It shares no embedding,
   encoder, normalization, or head parameter with the other owners.

The transaction microkernel is deliberately semantically small. It verifies
schema, owner-state hash, source/packet commitment, complete mention identity,
and monotone stage order. It can route a valid transaction to the existing
factorized rational state engine or reject it; it cannot infer, repair, or
average semantic fields. Every transaction records the owner that produced it
and the exact owner-state commitment.

### Two-phase plasticity

SOT1 uses owner-local optimization. A proposed update is written to a candidate
owner state first. The composite checkpoint commits only after:

- all non-target owner hashes remain bit-identical;
- the target owner's training receipt is complete;
- transaction schemas and manifest hashes validate; and
- the candidate contains no cross-owner parameter alias.

This first gate updates only the fresh QUERY owner. The qualified WORLD and
EVIDENCE owners are immutable. Later architecture-aware training may make an
owner plastic only through a separately frozen retention gate; SOT1 does not
silently unfreeze them.

### Execution boundary

The runtime path is:

```text
natural WORLD -> WORLD owner -> sealed program/fault-line packet
natural EVIDENCE -> EVIDENCE owner -> sealed monotone receipts
packet + receipts -> factorized recurrent state engine
natural QUERY -> QUERY owner -> sealed late read transaction
terminal support -> exact answer if invariant, otherwise ABSTAIN
```

After the WORLD owner seals its packet, raw WORLD bytes and hidden states are
absent from evidence, execution, and query stages. A query cannot alter support
or execution. Incompatible worlds remain complete and are never averaged.

## 3. Data and training

The QUERY owner trains on the immutable 50,000-row IEM1 query corpus at
SHA-256
`8dc6085f7632b56563416fa75c13ff611344ad53c7c2ca4350c54bbcc17301fa`.
The one fit uses seed `2026080616`, 1,000 AdamW updates, batch 256, learning
rate `3e-3` with cosine decay, betas `(0.9, 0.95)`, weight decay `0.01`, and
gradient clipping at 1.0. Loss is class-balanced cross entropy over one hard
two-group permutation. No answer, state, evidence, operation, or comparator
label reaches the QUERY owner.

The opened IEM1 board is development-only for integration debugging. The
scientific SOT1 result requires one fresh deterministic 256-program board at
seed `2026080617`, materialized after this specification and before training.
It retains 12 binary fault lines and three query modes per program but uses
new program identities and at least three query surface templates not present
in either query training or IEM1 confirmation. Exact WORLD, evidence, query,
and identity overlap with training and prior confirmation must be reported and
zero where applicable. No model score may select rows.

## 4. Controls

- immutable separate TOL3 + NVE1 + typed-query ceiling;
- closed IEM1 shared-owner checkpoint;
- premature highest-support top-1;
- equal-memory complete particles;
- no-evidence factorized support;
- query-owner role swap;
- evidence-owner numeric and symbol role swaps;
- shuffled complete evidence sets;
- declaration-state reset;
- owner-manifest swap and packet/query provenance swap; and
- post-seal WORLD/evidence/query poisoning.

The source and evidence owner state hashes are checked before and after query
training and after composite serialization. Any difference is a fatal
two-phase-commit violation.

## 5. Frozen pass/kill gate

All conditions are conjunctive:

1. source programs are at least 250/256 exact on the fresh board and at least
   1,000/1,024 exact on protected TOL3;
2. natural evidence is at least 3,041/3,072 exact and protected NVE1 recovery
   remains at least 250/256;
3. natural queries are at least 752/768 exact, every accepted query is valid,
   and all three query modes are at least 245/256 exact individually;
4. at least 245/256 sensitive answers are exact, including at least 95% exact
   conditional on initially wrong top-1, with extensional parity on every
   answered episode;
5. no-evidence sensitive queries abstain at least 245/256, invariant queries
   answer at least 245/256, and partial-evidence underdetermined queries
   abstain at least 245/256;
6. top-1 and equal-memory particles trail SOT1 by at least 50 points;
7. relevant owner-role swaps, shuffled evidence, and state reset each reduce
   exactness by at least 50 points;
8. every manifest or provenance swap rejects, post-seal poisoning is
   bit-invariant, and zero invalid transaction, false commitment, malformed
   packet, gold deletion, overflow, or valid-execution rejection is accepted;
9. non-target owner hashes are bit-identical before/after training and after
   composite checkpoint reload; and
10. parameters per owner, examples, bytes, updates, checkpoint/data hashes,
    wall time, peak memory, canonical storage, transaction counts, and
    logical/unique state applications are reported.

A QUERY owner miss closes this exact isolated-query design without a width,
duration, seed, renderer, optimizer, or loss variant. A component-retention or
composition miss closes SOT1. A pass qualifies the transaction architecture
for one broader free-form board and architecture-aware training; it does not
itself prove general reasoning.

## 6. Claim boundary

Owner-specialized encoders, typed messages, version spaces, exact ALUs,
two-phase commit, and global-workspace routing all have prior analogues. The
candidate contribution is their source-sealed conjunction as a learning
architecture: semantic stages have cryptographically accountable parameter
ownership, plasticity commits atomically against interface contracts, and a
factorized recurrent state machine consumes only coherent transactions. The
first SOT1 result remains controlled-language evidence even if it passes.
