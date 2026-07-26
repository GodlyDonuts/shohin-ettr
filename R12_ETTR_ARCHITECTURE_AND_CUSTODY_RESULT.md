# R12 ETTR Architecture and Custody Result

## Decision

`architecture_continuation_contract_complete_h100_profile_pending_pretraining_held_capability_unproven`

Shohin now has a concrete architecture core, a complete falsification matrix,
and an exact causal continuation/training contract for later pretraining and
post-training. It does not yet have demonstrated general reasoning. The user
has explicitly held continuation pretraining until the architecture is
qualified.

## Architecture

The Endogenous Typed Theory Reactor has four causal stages:

1. **World compiler:** actual Shohin residuals from raw world tokens
   cross-attend anonymous object slots and produce categorical value codes,
   latent types, a capped sparse relation ledger, activity, root, commit, and
   halt state.
2. **Command-conditioned reactor:** after world sealing, a separate raw-token
   command stream enters a shared recurrent controller. The controller emits
   only `ALLOC`, `WRITE`, `CLEAR`, `LINK`, `UNLINK`, `SET_ROOT`, `COMMIT`, and
   `HALT`, plus a distinct terminal `REJECT`.
3. **Typed graph update:** an edge-aware typed message bus preserves the
   identities and directions of relation endpoints before each transaction;
   each transaction changes only generic state. No
   family opcode, arithmetic routine, rewrite matcher, resource scheduler,
   search, repair, or answer callback exists in the runtime.
4. **Late-query reader:** after execution, a causally masked reader receives
   every declared terminal-state field and late-query residuals.

Hard transactions are bit-exact in the forward pass, while their
pre-discretization probabilities remain available for corrective gradients.
The two persistent status bits encode four always-visible dispositions:
`OPEN`, `ANSWER`, `ABSTAIN`, and `REJECT`. Deployed state rejects continuous
value channels, non-one-hot codes/types, non-binary control state, and
relation ledgers above 256 edges. Any terminal disposition freezes subsequent
structural writes. The production packet has 64 slots, 16 typed relation
roles, and 256 categorical symbols. It represents ordered hyperedges and
multi-byte values by reifying their role/value nodes rather than collapsing
them into one scalar label.

## Parameter Receipt

| Component | Parameters |
|---|---:|
| World compiler | 21,466,377 |
| Command-conditioned reactor | 29,757,217 |
| Late-query reader | 16,474,177 |
| Added architecture | **67,697,771** |
| Protected Shohin | 125,081,664 |
| Complete system | **192,779,435** |
| Remaining below 200M | **7,220,565** |

The protected step-300k checkpoint hash matches
`211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6`
and strictly loads with zero missing or unexpected tensors.

## Exact Offline Ontology Mechanics

| Ontology | Theories | Held-out structure | Independent comparisons |
|---|---:|---|---:|
| Typed Horn closure | 20 | entire ontology in leave-one-out fold | 7,560 |
| Typed term rewriting | 15 | 8 unseen rule combinations | 960 |
| Guarded resource process | 60 | 36 unseen length-2/3 programs | 174,960 |
| **Total** | **95** | rules, compositions, roles, halt/deadlock | **183,480** |

All independent oracle comparisons agree. The boards contain four opaque
renderers and 352 exact singleton, ambiguous, contradictory, or coherent-
alternate evidence episodes. These are assessor mechanics, never candidate
runtime imports.

## Four-Process Custody

State crosses processes only through an immutable safetensors file with seven
allowlisted tensors and three metadata fields. It contains no source text,
token offsets, source hash, residual cache, KV cache, parser state, executable
callback, or assessor product.

The serial test executes:

```text
compiler -> physical compiler-directory deletion
executor -> physical executor-directory deletion
late query -> physical query-directory deletion
independent assessor
```

World/compiler artifacts are absent before execution. Query inputs do not
exist during execution. Expected answers do not exist until candidate exit.
Every output is write-once and read-only.

## Frozen Seven-Variant Matrix

The three boards now materialize real alpha/reorder, alias-split, relation-
reification, type-twin, execution-semantics-twin, and ambiguity-deletion
transformations rather than relabeling renderer variants. The joined matrix
contains 3 folds, 24 held-out theories, 168 source worlds, 384 canonical
challenges, and 2,688 primary executions. It audits 1,472 invariant cases,
750 separating outcomes/directives, 384 abstentions, zero family-label leaks,
24 disjoint theory hashes, and 2,688 unique row hashes. Payload SHA-256 is
`d1904b54a0fab8e59cfcb0b0dd464f5c8778e5b828907028ec8614aeae76d5d5`.

## Causal Continuation Contract

The frozen architecture source adds:

1. `CausalETTREpisodeRunner`, with independent batch rows and explicit
   `WORLD`, `COMMAND`, and `QUERY` reset boundaries;
2. reset-safe token targets plus packet, transaction, equivariance,
   commit/halt, sparsity, and anti-bypass losses;
3. a canonical continuation batch and immutable manifest that reject live
   writers, family labels, malformed geometry, and snapshot drift;
4. disjoint protected-base and architecture optimizer groups, with base
   freezing and an embedded WSD update cursor;
5. atomic no-replace checkpoints covering exact model/optimizer/schedule/RNG/
   data state and protected-checkpoint provenance, admitted only at a complete
   optimizer and between-episode boundary; and
6. a bounded accumulation/update component that has no filesystem, shard,
   launcher, or network access.

The complete ETTR/cross-ontology architecture and custody inventory passes
**163/163**.
Reset-boundary tests include interior segment starts, exact native
Muon/AdamW resume, and next-update equivalence after restore. A
degree-preserving edge-swap falsifier holds every per-slot in/out relation
count fixed while changing endpoints; both the reactor and query reader change
their outputs. The previous degree-summary architecture was invariant to this
necessary distinction and is superseded.

The training boundary additionally proves exact optimizer/model parameter
identity, rejects mutable or forged causal targets, binds batches to immutable
manifest and dataset hashes, prevents scheduler steps past the frozen horizon,
and uses pre-discretization policy probabilities for hard-forward
supervision. Redundant per-segment LM losses are disabled in the composite
train step.

## Cross-Ontology Hybrid Receipt

The frozen hybrid board contains exactly three couplings:

- arithmetic result selects a rewrite location;
- Horn relation selects a guarded resource operator; and
- resource state selects a Horn query.

Each coupling has 16 factual/counterfactual cases. Independent executors agree
on **96/96** executions, and interventions change both the coupling signal and
final output on **48/48** cases. Candidate payloads expose no ontology label.
Payload SHA-256:
`d155f868494f9379b214028c8d7475cc2cde08192c9b3a5bbdea5a73b29f98e2`.

## Remaining Architecture Gate

- Complete the isolated BF16 H100 eager-versus-compiled memory/throughput
  profile from the exact corrected source commit.
- Record strict checkpoint load, nonzero architecture gradients, peak memory,
  measured throughput, compiled-arm status, and unchanged checkpoint hash.
- If profiling finds an OOM or systems defect, revise only the architecture
  implementation and repeat the same synthetic gate.

After that gate, the architecture can be called technically ready for the
user's later pretraining decision. Capability still requires future
pretraining, matched causal controls, unseen-ontology qualification, and
post-training. Only the user may lift the continuation-pretraining hold.
