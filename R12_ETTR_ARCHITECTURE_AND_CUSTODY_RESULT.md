# R12 ETTR Architecture and Custody Result

## Decision

`trainable_cross_ontology_architecture_ready_capability_unproven`

Shohin now has a concrete architecture for later continued pretraining and
post-training. It does not yet have demonstrated general reasoning.

## Architecture

The Endogenous Typed Theory Reactor has four causal stages:

1. **World compiler:** actual Shohin residuals from raw world tokens
   cross-attend anonymous object slots and produce latent types, relation
   tensors, activity, values, root, commit, and halt state.
2. **Command-conditioned reactor:** after world sealing, a separate raw-token
   command stream enters a shared recurrent controller. The controller emits
   only `ALLOC`, `WRITE`, `CLEAR`, `LINK`, `UNLINK`, `SET_ROOT`, `COMMIT`, and
   `HALT`.
3. **Typed graph update:** each transaction changes only generic state. No
   family opcode, arithmetic routine, rewrite matcher, resource scheduler,
   search, repair, or answer callback exists in the runtime.
4. **Late-query reader:** after execution, a separate reader receives only
   terminal state and late-query residuals.

Hard transactions are bit-exact in the forward pass and use straight-through
gradients during training.

## Parameter Receipt

| Component | Parameters |
|---|---:|
| World compiler | 17,120,265 |
| Command-conditioned reactor | 21,369,880 |
| Late-query reader | 7,947,329 |
| Added architecture | **46,437,474** |
| Protected Shohin | 125,081,664 |
| Complete system | **171,519,138** |
| Remaining below 200M | **28,480,862** |

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

## Remaining Before Qualification

- Materialize all seven preregistered variants for eight theories per fold.
- Freeze canonical/isomorphism split hashes and hybrid compositions.
- Add packet/transaction supervision and the continuation-pretraining data
  contract.
- Run matched actual-Shohin, zeroed, permuted, swapped, generic-recurrent,
  fixed-ontology, and family-routed controls.
- Demonstrate unseen-ontology capability rather than optimizer health.
- Only the user may lift the continuation-pretraining hold.

The architecture is ready to be trained. Its reasoning capability and
scientific novelty remain empirical hypotheses.
