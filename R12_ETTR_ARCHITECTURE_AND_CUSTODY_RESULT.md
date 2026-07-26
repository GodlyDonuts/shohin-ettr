# R12 ETTR Architecture and Custody Result

## Decision

`architecture_mechanics_hardened_primary_matrix_frozen_pretraining_interface_pending_capability_unproven`

Shohin now has a concrete architecture core and a complete falsification
matrix for later continued pretraining and post-training. It does not yet
have demonstrated general reasoning, and its continuation-pretraining
interface is not yet complete.

## Architecture

The Endogenous Typed Theory Reactor has four causal stages:

1. **World compiler:** actual Shohin residuals from raw world tokens
   cross-attend anonymous object slots and produce categorical value codes,
   latent types, a capped sparse relation ledger, activity, root, commit, and
   halt state.
2. **Command-conditioned reactor:** after world sealing, a separate raw-token
   command stream enters a shared recurrent controller. The controller emits
   only `ALLOC`, `WRITE`, `CLEAR`, `LINK`, `UNLINK`, `SET_ROOT`, `COMMIT`, and
   `HALT`.
3. **Typed graph update:** each transaction changes only generic state. No
   family opcode, arithmetic routine, rewrite matcher, resource scheduler,
   search, repair, or answer callback exists in the runtime.
4. **Late-query reader:** after execution, a causally masked reader receives
   every declared terminal-state field and late-query residuals.

Hard transactions are bit-exact in the forward pass and use straight-through
gradients during training. Deployed state rejects continuous value channels,
non-one-hot codes/types, non-binary control state, and relation ledgers above
96 edges. `COMMIT` freezes subsequent structural writes.

## Parameter Receipt

| Component | Parameters |
|---|---:|
| World compiler | 17,153,097 |
| Command-conditioned reactor | 21,174,360 |
| Late-query reader | 7,994,433 |
| Added architecture | **46,321,890** |
| Protected Shohin | 125,081,664 |
| Complete system | **171,403,554** |
| Remaining below 200M | **28,596,446** |

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

## Remaining Before Continuation Pretraining

- Implement a standard causal autoregressive episode interface with explicit
  segment/reset lifecycle and full-token language-model loss.
- Freeze packet/transaction/equivariance/commit objectives and anti-bypass
  controls.
- Freeze an ETTR-aware checkpoint/resume schema, optimizer groups, scheduler,
  RNG state, and protected-base provenance.
- Profile BF16 H100 memory and throughput after removing validation
  synchronizations from the recurrent hot path.
- Freeze hybrid compositions and their independent assessor receipts.
- Add packet/transaction supervision and the continuation-pretraining data
  contract.
- Run matched actual-Shohin, zeroed, permuted, swapped, generic-recurrent,
  fixed-ontology, and family-routed controls.
- Demonstrate unseen-ontology capability rather than optimizer health.
- Only the user may lift the continuation-pretraining hold.

The architecture core is ready for isolated causal qualification. It is not
yet authorized or technically ready for continuation pretraining. Its
reasoning capability and scientific novelty remain empirical hypotheses.
