# DIVERGE-PL1: Branch-Local Verified Plasticity

Status: oracle-typed mechanics lane only. SRP1 closed negative, so raw-language
PL1 is not admitted. The exact gate below may test branch-local plasticity with
an oracle-typed referent interface, but cannot qualify a language architecture,
Shohin integration, or continuation pretraining.

## 1. Question and prior-art boundary

PL1 asks one narrow question:

> Can a small model-owned policy state improve later, context-free attempts by
> committing only verifier-certified, branch-local eligibility writes, while
> every protected semantic and execution owner remains bit-identical?

Inference-time mutability, Hebbian updates, eligibility traces, fast weights,
metaplasticity, test-time training, transient LoRA, and multi-timescale memory
are established families. PL1 does not claim any of them individually. The
only candidate contribution is the conjunction of coherent DIVERGE branches,
copy-on-write policy state, verifier-gated local credit, exact rollback, and
measured improvement after the demonstrations and feedback text are removed.

PL1 is not a replacement for source compilation. Plasticity cannot recover a
referent that the compiler discarded. It is also not continuation pretraining,
durable consolidation, structural growth, pruning, or full-backbone updating.

## 2. Admission boundary

Before PL1 can consume natural text, SRP1 must satisfy its frozen gate and the
following owners must be hash-bound:

1. WORLD compiler;
2. numeric EVIDENCE compiler;
3. semantic REFERENT owner shared by EVIDENCE and QUERY; and
4. exact typed executor/verifier.

All four are immutable throughout PL1. Only `POLICY_PLASTIC` may change. A
pre-write manifest binds every protected model-state hash. Every attempted
commit recomputes those hashes and fails closed on any difference.

SRP1 missed. `REFERENT_ORACLE` therefore replaces only item 3. A result may
qualify the plastic update mechanics, but cannot qualify natural-language PL1
or a Shohin integration.

## 3. Episode family

Each episode defines a new eight-symbol mini-language. The symbols denote a
hidden permutation of eight noncommuting, invertible transforms over a pair of
registers modulo 97. The transform set is fixed, but symbol aliases, mapping,
initial registers, programs, and surface order are episode-local and disjoint
across train, development, and confirmation.

An episode contains:

- eight shallow acquisition programs of depth 3--5;
- twelve feedback attempts, each with eight complete branch proposals;
- sixteen transfer programs of depth 12--20;
- four poison/rollback probes; and
- four unrelated-task interference probes.

The verifier exposes only `PASS` or the index of the first invalid transition.
It never exposes the correct transform, state, answer, mapping, or gradient.
Each branch therefore receives delayed, deterministic, executable feedback,
but no answer label.

The transfer programs use the same episode-local rules at held-out depths and
compositions. Before transfer, all demonstrations, branch transcripts,
verifier messages, source residuals, and textual context are deleted. The
runtime retains only the fixed owners and the bounded plastic policy state.

## 4. Plastic owner

The owner has a fixed symbol feature map, fixed transform feature map, and a
mutable low-rank score correction:

```text
logit(symbol, transform) = base(symbol, transform) + u(symbol)^T F v(transform)
```

`u` and `v` are immutable. `F` is the only mutable session tensor. Version 0
uses rank 8 and one owner, not six layer-local ranklet banks. This is the
smallest test of the proposed causal mechanism.

At attempt `t`, branch `b` receives copy-on-write `F[t,b]`. It samples one
complete bijective symbol-to-transform assignment and executes every
acquisition program coherently under that assignment. Its eligibility record
contains the chosen symbol/transform pair for each executed transition and
the branch-local state hash at that transition.

On `PASS`, the write proposal potentiates all used assignments. On a verified
first-error certificate, it depresses only the assignment active at that
transition. Unverified, malformed, stale, or cross-branch certificates cannot
write. Branch proposals are combined only after verification; fields from
different complete assignments are never averaged into a synthetic mapping.

The commit is:

```text
proposal -> certificate check -> write-budget projection -> protected-hash check
         -> atomic session commit or exact rollback
```

The Frobenius norm of each attempt write is capped, the cumulative session
write budget is fixed, and the singular values of `F` are clipped to the
frozen homeostatic envelope. The transaction log stores pre-state hash,
certificate, branch identity, eligibility hash, update norm, effective rank,
and post-state hash.

## 5. Matched arms

Every arm receives the same episodes, branch count, proposal count, verifier
calls, mutable-scalar allowance, and total update FLOPs. No arm may retain raw
episode text at transfer.

1. `STATIC`: frozen policy; matched no-op update compute.
2. `CONTEXT_ONLY`: feedback remains available during acquisition but is
   deleted with all text before transfer.
3. `DIVERGE_ONLY`: eight coherent complete branches per attempt with no
   persistent policy write.
4. `FAST_WEIGHT`: one session matrix updated from all branch outcomes without
   branch-local eligibility or first-error localization.
5. `TRANSIENT_GRAD`: the same mutable-scalar budget updated by a standard
   transient policy-gradient/LoRA-style step on complete branch reward.
6. `PL1`: branch-local eligibility, verified local credit, bounded atomic
   commit, and rollback.

The static arm receives extra recurrent policy evaluations so total forward
FLOPs match PL1. Update FLOPs, verifier calls, wall time, peak memory, and
mutable bytes are reported separately rather than hidden in a single total.

## 6. Causal controls

The unchanged PL1 checkpoint is evaluated under:

- shuffled verifier outcomes;
- correct outcome assigned to the wrong branch;
- plastic-state reset before transfer;
- unrelated-episode plastic-state transplant;
- one poisoned certificate followed by atomic rollback;
- eligibility trace removed;
- write/homeostatic budget removed; and
- protected-owner mutation injection, which must fail closed.

The poison probe records behavior before poison, after poison, and after
rollback. Rollback must restore the exact pre-poison plastic hash and outputs.

## 7. Data and split custody

The CPU builder first materializes train, development, and unopened
confirmation episodes from separately frozen seeds. It reports exact identity,
alias, mapping, program, transition, and composition overlap. Development is
used only to verify that the board is nondegenerate and to measure random and
oracle ceilings. Quantitative gates are frozen after that calibration and
before fitting or opening confirmation.

The independent assessor re-enumerates every hidden mapping and executes every
program without importing candidate runtime code. It verifies unique episode
solutions, verifier certificates, first-error indices, transfer depth, poison
validity, and split disjointness.

## 8. CPU calibration

The fixed development board contains 256 oracle-typed episodes. The runtime
uses eight complete branches, twelve acquisition attempts, seed `2026080799`,
rank 8, per-write norm cap 4, score clip 8, and sixteen context-free transfer
programs per episode. Before confirmation, development transfer counts are:

| Arm | Exact transfer programs | Exact mappings |
| --- | ---: | ---: |
| STATIC | `1/4,096` | `0/256` |
| CONTEXT_ONLY | `1/4,096` | `0/256` |
| DIVERGE_ONLY | `3/4,096` | `0/256` |
| FAST_WEIGHT | `0/4,096` | `0/256` |
| TRANSIENT_GRAD | `190/4,096` | `11/256` |
| PL1 | `3,657/4,096` | `228/256` |

PL1's assessor-only context-free transfer probe rises from `1/4,096` after
attempt 1 to `3,657/4,096` after attempt 12. Reset, shuffled credit,
wrong-branch credit, and no eligibility score `1`, `2`, `0`, and `1` of 4,096.
Removing the write cap and score clip is identical to PL1 on development, so
homeostasis is retained as a safety envelope but explicitly excluded from the
candidate causal claim.

## 9. Frozen confirmation gate

The five confirmation seeds contain exactly 256 episodes each. All conditions
are conjunctive:

1. PL1 transfer exactness is at least 85% aggregate and 80% on every seed;
2. PL1 exact mapping recovery is at least 80% aggregate and 75% on every seed;
3. PL1 exceeds every matched arm by at least 10 absolute transfer points in
   aggregate and by at least 5 points on every seed;
4. deterministic 5,000-resample paired 95% bootstrap intervals for PL1 minus
   every matched arm have a lower bound above zero;
5. assessor-only transfer exactness after attempt 12 exceeds attempt 1 by at
   least 50 points;
6. plastic-state reset loses at least 25 points and does not beat STATIC by
   more than 3 points;
7. shuffled reward, wrong-branch credit, and unrelated transplant do not beat
   STATIC by more than 3 points;
8. poisoned state differs from pre-poison behavior in at least 95% of episodes,
   and rollback restores the exact pre-poison hash and outputs in every case;
9. removing eligibility loses at least 5 points; the no-homeostasis arm is
   stable and fully reported, but is not required to lose;
10. protected mutation injection fails closed on every seed and normal runs
    have zero protected-owner hash changes, unauthorized writes,
    stale-certificate commits, cross-branch commits, false verifier
    acceptances, or rollback mismatches; and
11. all arms report per-attempt accuracy, transfer exactness by depth, update
    norm, write budget, verifier calls, mutable bytes, operation receipts, and
    wall time.

A failure closes this exact PL1 update rule. Do not retry rank, width, layer,
branch count, seed, duration, optimizer, or write budget. Attribution may be
read-only. A mechanics pass under oracle referents authorizes one natural
integration only after a semantic compiler separately qualifies; it is not a
reasoning or language claim by itself.
